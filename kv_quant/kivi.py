from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from .bitpack import pack_bits_rows, packed_bytes, unpack_tensor_state
from .incremental import (
    IncrementalKVQuantizer,
    concat_parts,
    new_state,
    quantized_tensor_bytes,
    state_device,
    tensor_bytes,
    window_parts,
)
from .utils import (
    COMPARISON_GROUP_SIZE,
    SCALE_STORAGE_BYTES,
    _reshape_blocks,
    _unshape_blocks,
    dequantize_asym_min_offset,
    quantize_asym_min_offset,
    timed,
)


class KIVIQuantizer(IncrementalKVQuantizer):
    """KIVI-style asymmetric quantization with faithful K/V axes.

    K is quantized per channel over sequence groups.  V is quantized per
    token, per head, and per channel group; heads therefore never share a
    value scale.  The incremental path keeps a configurable recent BF16
    residual and migrates only complete sequence blocks.

    Official KIVI (``models/llama_kivi.py``) quantizes ``key_states`` after
    ``apply_rotary_pos_emb``, so integrations must cache post-RoPE keys.  The
    group size is the comparison setting's 64 so that every row is charged
    the same parameter budget; the 128-token BF16 residual and the K/V axes are
    KIVI's own.  A BF16 scale and a BF16 minimum are stored per group.
    """

    cache_space = "post_rope"
    #: Recent tokens KIVI keeps in BF16 (paper: 128).
    DEFAULT_RESIDUAL_LENGTH = 128

    def __init__(
        self,
        bits: int = 4,
        block_size: int = COMPARISON_GROUP_SIZE,
        residual_length: int | None = None,
        key_group_size: int | None = None,
        value_group_size: int | None = None,
    ) -> None:
        super().__init__(bits=bits)
        key_group_size = block_size if key_group_size is None else int(key_group_size)
        # KIVI keeps a full residual window, not one group: the paper runs
        # group size 32 with 128 recent tokens in BF16.  A group that does not
        # divide 128 rounds up, because the residual must stay a whole number
        # of groups.
        if residual_length is None:
            residual_length = math.ceil(self.DEFAULT_RESIDUAL_LENGTH / key_group_size) * key_group_size
        if residual_length % key_group_size != 0:
            raise ValueError("residual_length must be a multiple of key_group_size")
        self.residual_length = int(residual_length)
        self.key_group_size = key_group_size
        self.value_group_size = value_group_size

    def _resolve_value_group_size(self, head_dim: int) -> int:
        return int(self.value_group_size or self.key_group_size)

    def _quantize_keys(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.key_group_size)
        # KIVI K: per channel across the sequence tokens in each block.
        q, scale, zero = quantize_asym_min_offset(xb, bits=self.bits, reduce_dims=(2,))
        if pad_len:
            valid_tokens = self.key_group_size - pad_len
            tail_q, tail_scale, tail_zero = quantize_asym_min_offset(
                xb[:, -1:, :valid_tokens], bits=self.bits, reduce_dims=(2,),
            )
            q[:, -1:, :valid_tokens] = tail_q
            q[:, -1:, valid_tokens:] = 0
            scale[:, -1:] = tail_scale
            zero[:, -1:] = tail_zero
        return {
            "q": pack_bits_rows(q, self.bits, signed=False, row_dims=2),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed_rows": 2,
            "signed": False,
            "scale": scale,
            "zero": zero,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "key_group_size": self.key_group_size,
            "axis": "sequence_group_per_channel",
            "tensor_dtype": x.dtype,
        }

    def _quantize_values(self, x: torch.Tensor) -> Dict[str, Any]:
        b, length, h, d = x.shape
        group_size = self._resolve_value_group_size(d)
        xg = x.reshape(b, length, h, d // group_size, group_size)
        # KIVI V: each token/head/channel-group has its own scale/min-offset.
        q, scale, zero = quantize_asym_min_offset(xg, bits=self.bits, reduce_dims=(-1,))
        return {
            "q": pack_bits_rows(q, self.bits, signed=False, row_dims=2),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed_rows": 2,
            "signed": False,
            "scale": scale,
            "zero": zero,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "value_group_size": group_size,
            "axis": "token_head_channel_group",
            "tensor_dtype": x.dtype,
        }

    def _dequantize_keys(self, state: Dict[str, Any]) -> torch.Tensor:
        dtype = state.get("tensor_dtype", torch.bfloat16)
        x = dequantize_asym_min_offset(unpack_tensor_state(state), state["scale"], state["zero"], dtype=dtype)
        return _unshape_blocks(x, int(state["pad_len"]), int(state["orig_shape"][1]))

    def _dequantize_values(self, state: Dict[str, Any]) -> torch.Tensor:
        dtype = state.get("tensor_dtype", torch.bfloat16)
        xg = dequantize_asym_min_offset(unpack_tensor_state(state), state["scale"], state["zero"], dtype=dtype)
        return xg.reshape(tuple(int(dim) for dim in state["orig_shape"]))

    # -- state: independent K and V streams -----------------------------

    def _segment_count(self, state: Dict[str, Any]):
        return len(state["k_segments"]), len(state["v_segments"])

    def _sync_state(self, state: Dict[str, Any]) -> None:
        quantized_k = sum(int(item["length"]) for item in state["k_segments"])
        quantized_v = sum(int(item["length"]) for item in state["v_segments"])
        residual_k = state.get("residual_k")
        write_k = state.get("write_k")
        residual_k_tokens = int(residual_k.shape[1]) if residual_k is not None else 0
        write_tokens = int(write_k.shape[1]) if write_k is not None else 0
        state["quantized_k_tokens"] = quantized_k
        state["quantized_v_tokens"] = quantized_v
        state["quantized_tokens"] = quantized_k
        state["num_tokens"] = quantized_k + residual_k_tokens + write_tokens

        source = residual_k if residual_k is not None else write_k
        if source is not None:
            state["shape"] = (int(source.shape[0]), int(state["num_tokens"]), int(source.shape[2]), int(source.shape[3]))
        elif state["k_segments"]:
            shape = state["k_segments"][0]["state"]["orig_shape"]
            state["shape"] = (int(shape[0]), int(state["num_tokens"]), int(shape[2]), int(shape[3]))

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = new_state(meta)
        del state["segments"]
        state.update(
            residual_length=self.residual_length,
            k_segments=[],
            v_segments=[],
            residual_k=None,
            residual_v=None,
        )
        self._sync_state(state)
        return state

    def _commit_write(
        self,
        state: Dict[str, Any],
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        residual_k = state.get("residual_k")
        residual_v = state.get("residual_v")
        buffered_k = (
            torch.cat((residual_k, write_k), dim=1)
            if residual_k is not None and residual_k.shape[1] > 0
            else write_k
        )
        buffered_v = (
            torch.cat((residual_v, write_v), dim=1)
            if residual_v is not None and residual_v.shape[1] > 0
            else write_v
        )

        # K migrates whole residual windows, V every token beyond the window.
        k_unit = self.residual_length if self.residual_length > 0 else self.key_group_size
        k_flush = (int(buffered_k.shape[1]) // k_unit) * k_unit
        v_flush = max(int(buffered_v.shape[1]) - self.residual_length, 0)
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        for stream, tensor, quantize in (
            ("k_segments", buffered_k[:, :k_flush], self._quantize_keys),
            ("v_segments", buffered_v[:, :v_flush], self._quantize_values),
        ):
            if tensor.shape[1]:
                packed = quantize(tensor)
                packed["tensor_dtype"] = tensor_dtype
                state[stream].append({"state": packed, "length": int(tensor.shape[1])})
        # ``clone`` rather than ``contiguous``: for batch 1 the tail slice is
        # already contiguous, so ``contiguous`` would return a view that keeps
        # the whole BF16 buffer (e.g. LongCat's full condition K/V) alive.
        state["residual_k"] = buffered_k[:, k_flush:].clone()
        state["residual_v"] = buffered_v[:, v_flush:].clone()
        self._sync_state(state)

    def _window(self, state: Dict[str, Any], key: str, start: int, meta) -> torch.Tensor:
        segments = [(item["state"], int(item["length"])) for item in state[f"{key}_segments"]]
        decode = self._dequantize_keys if key == "k" else self._dequantize_values
        buffers = (state.get(f"residual_{key}"), state.get(f"write_{key}"))
        return concat_parts(window_parts(segments, buffers, decode, start), state, meta)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            state_meta = dict(meta or {})
            state_meta.setdefault("shape", (k.shape[0], 0, k.shape[2], k.shape[3]))
            state_meta.setdefault("tensor_dtype", k.dtype)
            state_meta.setdefault("device", k.device)
            state = self.init_state(state_meta)
            self._commit_write(state, k, v, state_meta)
            if state_meta.get("absolute_end") is not None:
                state["committed_end"] = int(state_meta["absolute_end"])
        self.stats.record_quantize(timer)
        self.stats.quantize_calls += 1
        return state

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        total = sum(
            quantized_tensor_bytes(item["state"])
            for stream in ("k_segments", "v_segments")
            for item in state[stream]
        )
        for name in ("residual_k", "residual_v", "write_k", "write_v"):
            total += tensor_bytes(state.get(name))
        return int(total)

    # -- eviction ---------------------------------------------------------

    def _slice_front(self, tensor_state: Dict[str, Any], tokens: int, is_key: bool) -> None:
        # K rows are key groups, V rows are tokens.
        rows = tokens // self.key_group_size if is_key else tokens
        # ``clone`` so the evicted prefix is actually released (a contiguous
        # tail slice would stay a view of the original payload).
        for name in ("q", "scale", "zero"):
            tensor_state[name] = tensor_state[name][:, rows:].clone()
        q_shape = list(tensor_state["q_shape"])
        q_shape[1] -= rows
        tensor_state["q_shape"] = tuple(q_shape)
        tensor_state["q_numel"] = math.prod(q_shape)
        orig_shape = list(tensor_state["orig_shape"])
        orig_shape[1] -= tokens
        tensor_state["orig_shape"] = tuple(orig_shape)
        if is_key:
            tensor_state["pad_len"] = 0

    def _evict_stream_prefix(self, state: Dict[str, Any], key: str, requested: int) -> None:
        items = state[f"{key}_segments"]
        remaining = requested
        while items and remaining > 0:
            length = int(items[0]["length"])
            if remaining >= length:
                items.pop(0)
                remaining -= length
                continue
            self._slice_front(items[0]["state"], remaining, is_key=key == "k")
            items[0]["length"] = length - remaining
            remaining = 0
        residual = state.get(f"residual_{key}")
        if remaining > 0 and residual is not None:
            state[f"residual_{key}"] = residual[:, remaining:].clone()

    def evict_prefix(self, state: Dict[str, Any], requested_tokens: int) -> int:
        """Evict whole key groups of committed history from both streams."""
        write_k = state.get("write_k")
        committed = int(state["num_tokens"]) - (int(write_k.shape[1]) if write_k is not None else 0)
        removable = min(max(int(requested_tokens), 0), committed)
        removable = (removable // self.key_group_size) * self.key_group_size
        if removable <= 0:
            return 0
        self._evict_stream_prefix(state, "k", removable)
        self._evict_stream_prefix(state, "v", removable)
        self._sync_state(state)
        return removable

    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        k_unit = self.residual_length if self.residual_length > 0 else self.key_group_size
        quantized_k = (active_tokens // k_unit) * k_unit
        quantized_v = max(active_tokens - self.residual_length, 0)
        key_blocks = quantized_k // self.key_group_size
        value_groups = head_dim // self._resolve_value_group_size(head_dim)
        key_payload = batch_size * key_blocks * packed_bytes(self.key_group_size * num_heads * head_dim, self.bits)
        value_payload = batch_size * quantized_v * packed_bytes(num_heads * head_dim, self.bits)
        # A BF16 scale and a BF16 minimum, as the reference stores them.
        param_bytes = 2 * SCALE_STORAGE_BYTES
        key_params = batch_size * key_blocks * num_heads * head_dim * param_bytes
        value_params = batch_size * quantized_v * num_heads * value_groups * param_bytes
        residual_values = (
            (active_tokens - quantized_k) + (active_tokens - quantized_v)
        ) * batch_size * num_heads * head_dim
        return int(key_payload + key_params + value_payload + value_params + residual_values * 2)
