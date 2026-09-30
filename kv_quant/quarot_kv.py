from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .bitpack import pack_bits, packed_bytes, unpack_tensor_state
from .incremental import IncrementalKVQuantizer
from .utils import (
    COMPARISON_GROUP_SIZE,
    SCALE_STORAGE_BYTES,
    SCALE_STORAGE_DTYPE,
    ZERO_STORAGE_BYTES,
    ZERO_STORAGE_DTYPE,
    fwht_last_dim,
    quarot_dequantize,
    quarot_find_params,
    quarot_quantize,
    round_scale_to_storage,
    timed,
)


class QuaRotKVQuantizer(IncrementalKVQuantizer):
    """KV-cache-only QuaRot, following spcl/QuaRot.

    The reference (``fake_quant/rotation_utils.py::QKRotationWrapper`` and
    ``fake_quant/rotation_utils.py::rotate_ov_proj``) does three things that
    together define the method:

    * Q and K are rotated by the same per-head Hadamard **after RoPE** and
      both stay rotated; the K cache holds the rotated keys.  QuaRot refuses
      pre-RoPE K quantization outright (``k_pre_rope`` raises).
    * V is produced already rotated, because a head-sized Hadamard is folded
      into ``v_proj``'s output weights and its inverse into ``o_proj``'s
      input weights.  With BF16 weights the identical, weight-free form is to
      cache the rotated V and undo the rotation on the attention output --
      attention is linear in V, so ``H(sum_j a_j v_j) == sum_j a_j (H v_j)``.
    * Both caches are quantized per token (``ActQuantizer`` only supports
      per-token), over one group of ``head_dim`` channels or over the whole
      hidden size.  K always goes through ``find_params`` with
      ``groupsize=-1`` (head-wise K is reshaped to ``[-1, head_dim]`` rows
      first), whose range is widened to include zero.  V is ``v_proj``'s
      output quantizer, so head-wise V takes
      ``find_params_per_token_groupwise`` and its range is *not* widened.

    Defaults follow the paper's KV setting (Section 5: "asymmetric
    quantization with a group size 128 with a constant clipping ratio of
    0.95"), with group size ``head_dim`` (128 for Wan and LongCat).

    ``materialize_kv`` therefore returns tensors in *attention space* when the
    state says so: the caller must rotate Q with :meth:`prepare_attention_qk`
    and pass the attention output through :meth:`restore_attention_output`.
    A caller that does neither gets the cache rotated back to the original
    basis instead, which stays numerically correct but is a plain
    rotate-quantize-unrotate round trip.
    """

    #: QuaRot asserts ``k_groupsize in [-1, head_dim]``.
    TOKEN_WISE_GROUP = -1
    #: Channels per scale in the shared comparison setting;
    #: ``channel_group_size=head_dim`` reproduces QuaRot's own KV setting.
    COMPARISON_GROUP = COMPARISON_GROUP_SIZE

    def __init__(
        self,
        bits: int = 4,
        channel_group_size: int | None = None,
        asym: bool = True,
        clip_ratio: float = 0.95,
        rotate_value: bool = True,
    ) -> None:
        if not 0.0 < float(clip_ratio) <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        super().__init__(bits=bits)
        self.channel_group_size = channel_group_size
        self.asym = bool(asym)
        self.sym = not self.asym
        self.clip_ratio = float(clip_ratio)
        self.rotate_value = bool(rotate_value)
        self.requires_special_attention_backend = True

    # -- rotation ---------------------------------------------------------

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        return fwht_last_dim(x)

    #: The normalized Hadamard is involutory, so one transform is its inverse.
    _inv_rotate = _rotate

    def prepare_attention_qk(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate a post-RoPE Q/K pair, exactly as ``QKRotationWrapper`` does."""
        return self._rotate(roped_query), self._rotate(roped_key)

    def prepare_value(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the ``v_proj`` output rotation that QuaRot folds into weights."""
        return self._rotate(value) if self.rotate_value else value

    def restore_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Undo the V rotation, standing in for ``o_proj``'s fused inverse."""
        return self._inv_rotate(attn_output) if self.rotate_value else attn_output

    # -- quantization -----------------------------------------------------

    def _resolve_group(self, head_dim: int, num_heads: int) -> int:
        """Channels that share one scale.

        The Hadamard is always one ``head_dim`` block, which is what makes the
        rotation exact; the quantizer's group is a separate choice.  The
        default 64 is the shared comparison group, and ``head_dim`` reproduces
        QuaRot's own KV setting.
        """
        group = self.channel_group_size
        if group is None:
            return int(min(self.COMPARISON_GROUP, head_dim))
        if group == self.TOKEN_WISE_GROUP:
            return int(num_heads * head_dim)
        if group == num_heads * head_dim:
            return int(group)
        if group <= 0 or head_dim % group:
            raise ValueError(
                "QuaRot groups must be token-wise (-1) or divide head_dim; got "
                f"channel_group_size={group} for head_dim={head_dim}"
            )
        return int(group)

    def _grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, L, H, D]`` so the last dim is one quantization group."""
        b, l, h, d = x.shape
        group = self._resolve_group(d, h)
        return x.reshape(b, l, (h * d) // group, group)

    #: Values rotated and quantized per chunk.  The FWHT works in float32 and
    #: allocates per level, so a whole long cache at once is ~10x its size.
    _CHUNK_VALUES = 1 << 24

    def _quantize_segment(self, x: torch.Tensor, is_value: bool, rotate=None) -> Dict[str, Any]:
        """Quantize ``x`` per token, applying ``rotate`` chunk by chunk first.

        Every step is per token (the Hadamard acts on the last dimension and
        the ranges are per token group), so chunking along the sequence gives
        the same codes and scales as one full-size pass.
        """
        # V is ``v_proj``'s output quantizer: head-wise groups skip widening
        # the range to include zero (``find_params_per_token_groupwise``).
        widen_to_zero = not (is_value and self.channel_group_size != self.TOKEN_WISE_GROUP)
        b, l, h, d = x.shape
        group = self._resolve_group(d, h)
        groups = (h * d) // group
        q = torch.empty((b, l, groups, group), dtype=torch.int8, device=x.device)
        scales, zeros = [], []
        tensor_dtype = x.dtype
        chunk = max(self._CHUNK_VALUES // max(b * h * d, 1), 1)
        for start in range(0, l, chunk):
            xc = x[:, start:start + chunk]
            if rotate is not None:
                xc = rotate(xc)
                tensor_dtype = xc.dtype
            xg = self._grouped(xc)
            scale, zero = quarot_find_params(
                xg, self.bits, self.sym, self.clip_ratio, widen_to_zero=widen_to_zero
            )
            scale = round_scale_to_storage(scale)
            q[:, start:start + chunk] = quarot_quantize(xg, scale, zero, self.bits, self.sym)
            scales.append(scale.to(SCALE_STORAGE_DTYPE))
            if not self.sym:
                zeros.append(zero.to(ZERO_STORAGE_DTYPE))
            del xc, xg, scale, zero
        empty = torch.empty((b, 0, groups, 1), dtype=SCALE_STORAGE_DTYPE, device=x.device)
        state = {
            "q": pack_bits(q, self.bits, signed=self.sym),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "signed": self.sym,
            "scale": torch.cat(scales, dim=1) if scales else empty,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "sym": self.sym,
            "channel_group_size": int(group),
            "axis": "post_rope_token_group",
            "rotated": True,
            "tensor_dtype": tensor_dtype,
        }
        if not self.sym:
            state["zero"] = torch.cat(zeros, dim=1) if zeros else empty.to(ZERO_STORAGE_DTYPE)
        return state

    def _decode_segment(self, state: Dict[str, Any]) -> torch.Tensor:
        sym = state["sym"]
        zero = state["zero"] if not sym else torch.zeros_like(state["scale"])
        x = quarot_dequantize(
            unpack_tensor_state(state), state["scale"], zero, sym,
            dtype=state.get("tensor_dtype", torch.bfloat16),
        )
        return x.reshape(tuple(int(dim) for dim in state["orig_shape"]))

    # -- cache hooks ------------------------------------------------------

    def _prepare_append(self, new_k, new_v, meta: Dict[str, Any]):
        already_rotated = bool(meta.get("already_rotated", False))
        # A generic caller gets the original basis back from ``materialize_kv``.
        # The model runtime opts into attention space explicitly once it
        # rotates post-RoPE Q/K together and compensates the attention output.
        meta["attention_space"] = bool(meta.get("attention_space", already_rotated))
        if already_rotated:
            return new_k, new_v
        return self._rotate(new_k), self.prepare_value(new_v)

    def _output_basis(self, k, v, state: Dict[str, Any]):
        if state.get("attention_space", False):
            return k, v
        return self._inv_rotate(k), (self._inv_rotate(v) if self.rotate_value else v)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        meta = dict(meta or {})
        already_rotated = bool(meta.get("already_rotated", False))
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            # Rotation runs inside the chunked quantization loop, so the full
            # rotated float32 cache never exists at once.
            k_state = self._quantize_segment(
                k, is_value=False, rotate=None if already_rotated else self._rotate,
            )
            v_state = self._quantize_segment(
                v, is_value=True, rotate=None if already_rotated else self.prepare_value,
            )
            tensor_dtype = meta.get("tensor_dtype", k.dtype)
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            state = {
                "k": k_state,
                "v": v_state,
                # A caller may ask the one-shot cache to stay rotated without
                # having rotated the input itself, which is what LongCat's
                # condition cache does.  This mirrors ``append_kv``.
                "attention_space": bool(meta.get("attention_space", already_rotated)),
            }
        self.stats.record_quantize(timer)
        self.stats.quantize_calls += 1
        return state

    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        group = self._resolve_group(head_dim, num_heads)
        values = batch_size * active_tokens * num_heads * head_dim
        groups = batch_size * active_tokens * (num_heads * head_dim) // group
        # A BF16 scale, plus a BF16 zero point when asymmetric.
        param_bytes = groups * (SCALE_STORAGE_BYTES + (ZERO_STORAGE_BYTES if self.asym else 0))
        return int(2 * (packed_bytes(values, self.bits) + param_bytes))


class HadamardKQuantizer(QuaRotKVQuantizer):
    """QuaRot with the V-side rotation removed.

    Kept as the K-only ablation of :class:`QuaRotKVQuantizer`: it isolates how
    much of QuaRot's benefit comes from the shared post-RoPE Q/K rotation
    alone.  It is not QuaRot and must not be reported as such.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["rotate_value"] = False
        super().__init__(*args, **kwargs)
