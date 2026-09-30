"""Bridge LongCat's ``[B, H, S, D]`` cache layout to ``kv_quant``.

The shared quantizers deliberately use ``[B, S, H, D]`` so that their
sequence blocking is identical across models.  LongCat keeps the head axis
before the sequence axis, therefore this conversion must happen at the
adapter boundary and nowhere inside the quantizer implementations.
"""

from __future__ import annotations

from typing import Any

import torch


FORMAT_NAME = "shared_kv_quant_v1"


def _swap_seq_heads(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` <-> ``[B, S, H, D]`` as a view.

    A contiguous copy of the whole condition cache would be held for the full
    quantization call; the quantizers copy only what they reshape.
    """
    return x.permute(0, 2, 1, 3)


def encode_longcat_kv(k: torch.Tensor, v: torch.Tensor, quantizer) -> dict[str, Any]:
    # QuaRot keeps the cache in the rotated attention space; the attention
    # module rotates Q to match and undoes the V rotation on its output.
    attention_space = bool(getattr(quantizer, "requires_special_attention_backend", False))
    # The caller captures K after RoPE for QuaRot and for methods that declare
    # ``cache_space = "post_rope"`` (KIVI); the payload records which it is.
    cache_space = "post_rope" if attention_space else getattr(quantizer, "cache_space", "pre_rope")
    state = quantizer.quantize_kv(
        _swap_seq_heads(k),
        _swap_seq_heads(v),
        meta={"tensor_dtype": k.dtype, "device": k.device, "attention_space": attention_space},
    )
    return {
        "format": FORMAT_NAME,
        "state": state,
        "dtype": k.dtype,
        "shape": tuple(k.shape),
        "attention_space": attention_space,
        "cache_space": cache_space,
    }


def decode_longcat_kv(payload: dict[str, Any], quantizer) -> tuple[torch.Tensor, torch.Tensor]:
    k, v = quantizer.dequantize_kv(payload["state"], meta={"tensor_dtype": payload["dtype"]})
    return _swap_seq_heads(k), _swap_seq_heads(v)


def is_shared_quant_cache(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") == FORMAT_NAME


def move_state_to(obj: Any, device: torch.device | str) -> Any:
    """Recursively move the tensors of a quantized payload to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {name: move_state_to(value, device) for name, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_state_to(value, device) for value in obj)
    return obj
