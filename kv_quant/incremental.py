"""Append-only KV-cache states shared by the baseline quantizers.

The quantizers use different metadata and quantization axes, but they all
need the same lifetime rules in the model runtime:

* quantized segments are immutable;
* the current diffusion block stays in a write buffer so repeated denoising
  calls can replace it without re-quantizing history.

Quantized segments are stored as a list.  Concatenating packed payloads would
require unpacking/repacking whenever a segment is not byte aligned, which
would violate append invariance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch

from .base import KVQuantizer
from .utils import timed

INCREMENTAL_STATE_FORMAT = "incremental_kv_v1"


def new_state(meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta = dict(meta or {})
    return {
        "format": INCREMENTAL_STATE_FORMAT,
        "segments": [],
        "write_k": None,
        "write_v": None,
        "quantized_tokens": 0,
        "num_tokens": 0,
        "shape": tuple(int(dim) for dim in meta.get("shape", (0, 0, 0, 0))),
        "tensor_dtype": meta.get("tensor_dtype"),
        "device": meta.get("device"),
        "write_start": None,
        "write_end": None,
        "committed_end": None,
        "attention_space": bool(meta.get("attention_space", False)),
    }


def is_incremental_state(state: Any) -> bool:
    return isinstance(state, dict) and state.get("format") == INCREMENTAL_STATE_FORMAT


def append_segment(
    state: Dict[str, Any],
    k_state: Dict[str, Any],
    v_state: Dict[str, Any],
    length: int,
) -> None:
    if length > 0:
        state["segments"].append({"k": k_state, "v": v_state, "length": int(length)})


def recompute_counts(state: Dict[str, Any]) -> None:
    quantized_tokens = sum(int(segment["length"]) for segment in state["segments"])
    write_k = state.get("write_k")
    write_tokens = int(write_k.shape[1]) if write_k is not None else 0
    state["quantized_tokens"] = quantized_tokens
    state["num_tokens"] = quantized_tokens + write_tokens
    if write_k is not None:
        state["shape"] = (
            int(write_k.shape[0]),
            int(state["num_tokens"]),
            int(write_k.shape[2]),
            int(write_k.shape[3]),
        )


def prepare_write(
    state: Dict[str, Any],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    meta: Dict[str, Any],
    commit: Callable[[Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, Any]], None],
) -> str:
    """Install a new mutable write, committing the previous one if needed.

    Returns ``"replace"`` when the current absolute range was overwritten and
    ``"append"`` for a new range.  The callback moves the previous write into
    the method-specific quantized part of the state; token counts are left for
    the caller to recompute.
    """
    start = meta.get("absolute_start")
    end = meta.get("absolute_end")
    old_write_k = state.get("write_k")
    # Rewriting a committed range would leave the packed history stale.
    committed_end = state.get("committed_end") if old_write_k is None else state.get("write_end")
    same_range = (
        old_write_k is not None
        and start is not None
        and end is not None
        and state.get("write_start") == int(start)
        and state.get("write_end") == int(end)
    )
    if not same_range and start is not None and committed_end is not None and int(start) < int(committed_end):
        raise ValueError(
            "Cannot overwrite an already committed KV-cache range; "
            "only the current mutable write range may be replaced"
        )

    if old_write_k is not None and not same_range:
        commit(state, old_write_k, state["write_v"], meta)
        if state.get("write_end") is not None:
            state["committed_end"] = int(state["write_end"])

    state["write_k"] = new_k
    state["write_v"] = new_v
    state["tensor_dtype"] = meta.get("tensor_dtype", new_k.dtype)
    if not same_range:
        state["device"] = new_k.device
        state["write_start"] = int(start) if start is not None else None
        state["write_end"] = int(end) if end is not None else None
    if meta.get("attention_space") is not None:
        state["attention_space"] = bool(meta["attention_space"])
    return "replace" if same_range else "append"


def empty_from_state(state: Dict[str, Any], meta: Dict[str, Any] | None) -> torch.Tensor:
    meta = dict(meta or {})
    shape = tuple(int(dim) for dim in meta.get("shape", state.get("shape", (0, 0, 0, 0))))
    dtype = meta.get("tensor_dtype") or state.get("tensor_dtype") or torch.bfloat16
    device = meta.get("device") or state.get("device") or "cpu"
    return torch.empty((shape[0], 0, shape[2], shape[3]), dtype=dtype, device=device)


def window_parts(
    segments: list,
    buffers: tuple,
    decode: Callable[[Dict[str, Any]], torch.Tensor],
    start: int,
) -> list:
    """Decode only the parts of one stream at or after token ``start``.

    Attention reads a bounded window, so history before it is neither
    unpacked nor dequantized; a long generation otherwise decodes its entire
    (never-attended) prefix on every call.
    """
    parts = []
    cursor = 0
    for segment, length in segments:
        if cursor + length > start:
            parts.append(decode(segment)[:, max(start - cursor, 0):])
        cursor += length
    for buffered in buffers:
        if buffered is not None and buffered.shape[1] > 0:
            length = int(buffered.shape[1])
            if cursor + length > start:
                parts.append(buffered[:, max(start - cursor, 0):])
            cursor += length
    return parts


def concat_parts(parts: list, state: Dict[str, Any], meta: Dict[str, Any] | None) -> torch.Tensor:
    if not parts:
        return empty_from_state(state, meta)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return 0


def quantized_tensor_bytes(state: Dict[str, Any]) -> int:
    """Packed payload plus its scale and zero parameters."""
    return sum(tensor_bytes(state.get(key)) for key in ("q", "scale", "zero"))


def state_device(state: Dict[str, Any]) -> torch.device:
    for key in ("write_k", "residual_k"):
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            return value.device
    for segment in state.get("segments", ()):
        return segment["k"]["q"].device
    configured = state.get("device")
    return torch.device(configured) if configured is not None else torch.device("cpu")


class IncrementalKVQuantizer(KVQuantizer):
    """Shared lifecycle for the RTN and QuaRot caches.

    A subclass supplies ``_quantize_segment`` and ``_decode_segment``; KIVI
    overrides the state layout for its independent K/V streams.
    """

    def _quantize_segment(self, x: torch.Tensor, is_value: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _decode_segment(self, tensor_state: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_append(self, new_k, new_v, meta: Dict[str, Any]):
        """Hook for a transform applied to K/V before they enter the cache."""
        return new_k, new_v

    def _output_basis(self, k, v, state: Dict[str, Any]):
        """Hook for a transform applied to decoded K/V."""
        return k, v

    def _segment_count(self, state: Dict[str, Any]):
        return len(state["segments"])

    def _sync_state(self, state: Dict[str, Any]) -> None:
        recompute_counts(state)

    # -- append-only cache ------------------------------------------------

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return new_state(meta)

    def _commit_write(
        self,
        state: Dict[str, Any],
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        if write_k.shape[1] == 0:
            return
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        k_state = self._quantize_segment(write_k, is_value=False)
        v_state = self._quantize_segment(write_v, is_value=True)
        k_state["tensor_dtype"] = tensor_dtype
        v_state["tensor_dtype"] = tensor_dtype
        append_segment(state, k_state, v_state, int(write_k.shape[1]))

    def append_kv(
        self,
        state: Dict[str, Any],
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        meta = dict(meta or {})
        meta.setdefault("tensor_dtype", new_k.dtype)
        new_k, new_v = self._prepare_append(new_k, new_v, meta)
        before = self._segment_count(state)
        had_write = state.get("write_k") is not None
        with timed(new_k.device, enabled=self.stats.timing_enabled) as timer:
            result = prepare_write(state, new_k, new_v, meta, self._commit_write)
        self._sync_state(state)
        if result != "replace" and (had_write or self._segment_count(state) != before):
            self.stats.record_quantize(timer)
            self.stats.quantize_calls += 1
        return state

    def finalize_state(
        self, state: Dict[str, Any], meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Commit the mutable write buffer, e.g. before reporting resident bytes."""
        write_k = state.get("write_k")
        if write_k is None or write_k.shape[1] == 0:
            return state
        meta = dict(meta or {})
        meta.setdefault("tensor_dtype", write_k.dtype)
        with timed(write_k.device, enabled=self.stats.timing_enabled) as timer:
            self._commit_write(state, write_k, state["write_v"], meta)
        if state.get("write_end") is not None:
            state["committed_end"] = int(state["write_end"])
        state["write_k"] = None
        state["write_v"] = None
        self._sync_state(state)
        self.stats.record_quantize(timer)
        self.stats.quantize_calls += 1
        return state

    def _window(self, state: Dict[str, Any], key: str, start: int, meta) -> torch.Tensor:
        segments = [(segment[key], int(segment["length"])) for segment in state["segments"]]
        parts = window_parts(segments, (state.get(f"write_{key}"),), self._decode_segment, start)
        return concat_parts(parts, state, meta)

    def materialize_kv(
        self, state: Dict[str, Any], meta: Dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``meta["start_token"]`` limits decoding to the attention window.
        start = max(int((meta or {}).get("start_token", 0)), 0)
        with timed(state_device(state), enabled=self.stats.timing_enabled) as timer:
            k = self._window(state, "k", start, meta)
            v = self._window(state, "v", start, meta)
            k, v = self._output_basis(k, v, state)
        self.stats.record_dequantize(timer)
        self.stats.dequantize_calls += 1
        return k, v

    def evict_prefix(self, state: Dict[str, Any], requested_tokens: int) -> int:
        """Drop complete immutable segments from the logical prefix.

        A partial segment is deliberately left untouched; the caller's
        attention slice still enforces the window.  Nothing is re-packed.
        """
        remaining = max(int(requested_tokens), 0)
        removed = 0
        segments = state["segments"]
        while segments and remaining >= int(segments[0]["length"]):
            length = int(segments.pop(0)["length"])
            remaining -= length
            removed += length
        if removed:
            self._sync_state(state)
        return removed

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        if not is_incremental_state(state):
            return sum(quantized_tensor_bytes(state[key]) for key in ("k", "v"))
        total = sum(
            quantized_tensor_bytes(segment[key])
            for segment in state["segments"]
            for key in ("k", "v")
        )
        return int(total + tensor_bytes(state.get("write_k")) + tensor_bytes(state.get("write_v")))

    # -- one-shot cache ---------------------------------------------------

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        tensor_dtype = (meta or {}).get("tensor_dtype", k.dtype)
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            k_state = self._quantize_segment(k, is_value=False)
            v_state = self._quantize_segment(v, is_value=True)
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            state = {"k": k_state, "v": v_state}
        self.stats.record_quantize(timer)
        self.stats.quantize_calls += 1
        return state

    def dequantize_kv(
        self, state: Dict[str, Any], meta: Dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_incremental_state(state):
            return self.materialize_kv(state, meta)
        with timed(state["k"]["q"].device, enabled=self.stats.timing_enabled) as timer:
            k, v = self._output_basis(
                self._decode_segment(state["k"]), self._decode_segment(state["v"]), state
            )
        self.stats.record_dequantize(timer)
        self.stats.dequantize_calls += 1
        return k, v
