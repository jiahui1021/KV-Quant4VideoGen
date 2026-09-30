from __future__ import annotations

from typing import Any, Dict

import torch

from .bitpack import pack_bits, packed_bytes, unpack_tensor_state
from .incremental import IncrementalKVQuantizer
from .utils import (
    COMPARISON_GROUP_SIZE,
    SCALE_STORAGE_BYTES,
    ZERO_STORAGE_BYTES,
    dequantize_asym_min_offset,
    quantize_asym_min_offset,
    reshape_channel_groups,
)


class RTNQuantizer(IncrementalKVQuantizer):
    """Group-wise per-token asymmetric RTN over an append-only cache.

    This is the RTN row of the comparison: round to nearest over channel groups
    of 64, asymmetric, no rotation, with a BF16 scale and a BF16 group minimum.
    QuaRot runs the same quantizer on Hadamard-rotated keys and values, so the
    two rows differ only by the rotation.
    """

    def __init__(
        self,
        bits: int = 4,
        block_size: int = COMPARISON_GROUP_SIZE,
        channel_group_size: int | None = None,
    ) -> None:
        super().__init__(bits=bits)
        # ``block_size`` is the shared CLI spelling for RTN's channel group.
        self.channel_group_size = block_size if channel_group_size is None else int(channel_group_size)

    def _resolve_group(self, head_dim: int) -> int:
        """Channels per scale, narrowed to ``head_dim`` when it is smaller."""
        return int(min(self.channel_group_size, head_dim))

    def _quantize_segment(self, x: torch.Tensor, is_value: bool) -> Dict[str, Any]:
        group = self._resolve_group(int(x.shape[-1]))
        q, scale, zero = quantize_asym_min_offset(
            reshape_channel_groups(x, group), bits=self.bits, reduce_dims=(-1,)
        )
        return {
            "q": pack_bits(q, self.bits, signed=False),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "signed": False,
            "scale": scale,
            "zero": zero,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "channel_group_size": group,
            "axis": "token_head_channel_group",
            "tensor_dtype": x.dtype,
        }

    def _decode_segment(self, state: Dict[str, Any]) -> torch.Tensor:
        dtype = state.get("tensor_dtype", torch.bfloat16)
        xg = dequantize_asym_min_offset(unpack_tensor_state(state), state["scale"], state["zero"], dtype=dtype)
        return xg.reshape(tuple(int(dim) for dim in state["orig_shape"]))

    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        group = self._resolve_group(head_dim)
        values = batch_size * active_tokens * num_heads * head_dim
        groups = batch_size * active_tokens * num_heads * (head_dim // group)
        param_bytes = groups * (SCALE_STORAGE_BYTES + ZERO_STORAGE_BYTES)
        return int(2 * (packed_bytes(values, self.bits) + param_bytes))
