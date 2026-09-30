"""Block-boundary resident-byte sampling for the Self-Forcing KV quantizers.

This repository's integration lives in the attention layer: the vendored
``third_party/Self-Forcing/wan/modules/causal_model.py`` dequantizes on read and
quantizes on write inside ``CausalWanSelfAttention``.  The pipeline's block loop
is unmodified upstream code, so there is no block hook available without
changing it as well.

The wrapper below reconstructs the boundary instead: ``quantize_kv`` (or
``append_kv`` on the incremental KIVI path) is called once per layer per
block, so a wrap of the layer counter marks a new block.
The sample is taken before the block's first layer is quantized and again
after, which is the same dense/packed state the other two repositories sample.
The equality gate in ``collect_efficiency.py`` -- every method's
``peak_bf16_equivalent_bytes`` must equal the BF16 run's
``peak_resident_bytes`` -- is what proves the reconstruction is correct.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

from . import efficiency_record as efficiency


class SamplingQuantizer:
    """Transparent quantizer wrapper that samples the cache at block edges.

    Everything except the two write calls is forwarded to the wrapped
    quantizer unchanged.
    """

    def __init__(
        self,
        inner,
        cache_getter: Callable[[], Sequence[Dict[str, Any]]],
        num_layers: int,
    ) -> None:
        self._inner = inner
        self._cache_getter = cache_getter
        self._num_layers = int(num_layers)
        self._layer_cursor = 0
        self.sampler = efficiency.ResidentSampler()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    # -- sampling ---------------------------------------------------------

    def quantize_kv(self, k, v, meta=None):
        if self._layer_cursor == 0:
            self._sample()
        result = self._inner.quantize_kv(k, v, meta=meta)
        self._advance_layer_cursor()
        return result

    def append_kv(self, state, new_k, new_v, meta=None):
        if self._layer_cursor == 0:
            self._sample()
        result = self._inner.append_kv(state, new_k, new_v, meta=meta)
        self._advance_layer_cursor()
        return result

    def _advance_layer_cursor(self) -> None:
        self._layer_cursor += 1
        if self._layer_cursor >= self._num_layers:
            self._layer_cursor = 0
            self._sample()

    def _sample(self) -> None:
        resident, equivalent = self.resident_kv_bytes()
        self.sampler.observe(resident, equivalent)

    def resident_kv_bytes(self) -> Tuple[int, int]:
        """Resident bytes and BF16 equivalent, per ``resident_analytic.v1``.

        Counts the packed ``quant_state`` and any dense ``k``/``v`` buffer
        still allocated; both are resident at the same time.
        """
        layers = list(self._cache_getter())
        if not layers:
            return 0, 0
        resident = 0
        tokens = 0
        for block in layers:
            for key in ("quant_state", "k", "v"):
                resident += efficiency.tensor_bytes(block.get(key))
            end_index = block.get("local_end_index")
            tokens = max(tokens, int(end_index) if end_index is not None else 0)
        batch, heads, head_dim = _cache_geometry(layers[0])
        equivalent = efficiency.bf16_equivalent_bytes(
            batch=batch, tokens=tokens, heads=heads, head_dim=head_dim
        ) * len(layers)
        return resident, equivalent


def _cache_geometry(block: Dict[str, Any]) -> Tuple[int, int, int]:
    """Batch, heads, and head dimension of one cache block."""
    tensor = block.get("k")
    if tensor is not None and getattr(tensor, "ndim", 0) == 4:
        batch, _, heads, head_dim = tensor.shape
        return int(batch), int(heads), int(head_dim)
    return int(block["batch_size"]), int(block["num_heads"]), int(block["head_dim"])
