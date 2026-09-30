from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch


@dataclass
class QuantizerStats:
    quantize_time_s: float = 0.0
    dequantize_time_s: float = 0.0
    quantize_calls: int = 0
    dequantize_calls: int = 0
    timing_enabled: bool = False
    _pending_quantize_timers: list[Any] = field(default_factory=list, repr=False)
    _pending_dequantize_timers: list[Any] = field(default_factory=list, repr=False)

    def record_quantize(self, timer: Any) -> None:
        if not timer.enabled:
            return
        if timer.is_cuda:
            self._pending_quantize_timers.append(timer)
        else:
            self.quantize_time_s += timer.resolve()

    def record_dequantize(self, timer: Any) -> None:
        if not timer.enabled:
            return
        if timer.is_cuda:
            self._pending_dequantize_timers.append(timer)
        else:
            self.dequantize_time_s += timer.resolve()

    def resolve_timing(self, synchronize: bool = True) -> None:
        """Resolve pending CUDA events, synchronizing at most once per device."""
        pending = self._pending_quantize_timers + self._pending_dequantize_timers
        if not pending:
            return

        if synchronize:
            devices = {timer.device for timer in pending if timer.device is not None}
            for device in devices:
                torch.cuda.synchronize(device)

        for timer in self._pending_quantize_timers:
            self.quantize_time_s += timer.resolve()
        for timer in self._pending_dequantize_timers:
            self.dequantize_time_s += timer.resolve()
        self._pending_quantize_timers.clear()
        self._pending_dequantize_timers.clear()


class KVQuantizer(ABC):
    """KV-cache quantizer protocol shared by the three model integrations.

    The model runtimes use the append-only cache (``init_state``,
    ``append_kv``, ``materialize_kv``, ``finalize_state``, ``evict_prefix``);
    LongCat's condition cache and the reconstruction experiments use the
    one-shot ``quantize_kv``/``dequantize_kv`` pair.
    """

    def __init__(self, bits: int = 4) -> None:
        self.bits = bits
        self.stats = QuantizerStats()

    def reset_stats(self) -> None:
        self.stats = QuantizerStats(timing_enabled=self.stats.timing_enabled)

    def set_timing_enabled(self, enabled: bool) -> None:
        """Enable optional quantize/dequantize timing breakdowns."""
        self.stats.timing_enabled = bool(enabled)

    @abstractmethod
    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def append_kv(self, state: Dict[str, Any], new_k, new_v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def materialize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def finalize_state(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evict_prefix(self, state: Dict[str, Any], requested_tokens: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self, state: Dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        """Analytic resident bytes of ``active_tokens`` cache positions."""
        raise NotImplementedError
