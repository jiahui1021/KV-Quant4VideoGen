"""Runtime helpers for the shared KV-cache quantizers in Causal-Forcing."""

from __future__ import annotations

import torch


def _cache_lists(pipeline) -> list[list[dict]]:
    if hasattr(pipeline, "kv_cache1"):
        return [pipeline.kv_cache1] if pipeline.kv_cache1 is not None else []
    lists = []
    for name in ("kv_cache_pos", "kv_cache_neg"):
        value = getattr(pipeline, name, None)
        if value is not None:
            lists.append(value)
    return lists


def _initialize_pipeline_caches(pipeline, dtype, device) -> None:
    if hasattr(pipeline, "kv_cache1"):
        if pipeline.kv_cache1 is None:
            pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
            pipeline._initialize_crossattn_cache(
                batch_size=1, dtype=dtype, device=device
            )
        return

    if getattr(pipeline, "kv_cache_pos", None) is None:
        pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
        pipeline._initialize_crossattn_cache(
            batch_size=1, dtype=dtype, device=device
        )


def attach_quantizer_to_pipeline(
    pipeline,
    quantizer,
    num_output_frames: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Attach a shared quantizer to every causal self-attention cache."""
    _initialize_pipeline_caches(pipeline, dtype=dtype, device=device)
    required_tokens = int(num_output_frames) * int(pipeline.frame_seq_length)

    for cache_list in _cache_lists(pipeline):
        for layer_idx, block in enumerate(cache_list):
            k = block["k"]
            cache_size = int(k.shape[1])
            if pipeline.local_attn_size == -1:
                # Only the capacity is needed: dense K/V are discarded below.
                cache_size = max(cache_size, required_tokens)
            block["kv_cache_size"] = cache_size
            block["batch_size"] = int(k.shape[0])
            block["num_heads"] = int(k.shape[2])
            block["head_dim"] = int(k.shape[3])
            block["dtype"] = dtype
            block["frame_seq_length"] = int(pipeline.frame_seq_length)
            block["num_frame_per_block"] = int(pipeline.num_frame_per_block)
            block["layer_id"] = int(layer_idx)
            block["quantizer"] = quantizer
            block["quant_state"] = None
            block["k"] = torch.empty(0, dtype=dtype, device=device)
            block["v"] = torch.empty(0, dtype=dtype, device=device)


def reset_quantized_kv_cache(pipeline) -> None:
    """Reset indices and compressed state before processing a new prompt."""
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            # The diffusion pipeline keeps tensor cursors, the DMD one ints.
            for name in ("global_end_index", "local_end_index"):
                if isinstance(block[name], torch.Tensor):
                    block[name].fill_(0)
                else:
                    block[name] = 0
            block["quant_state"] = None
            if block["k"].numel():
                block["k"].zero_()
                block["v"].zero_()


def finalize_quantized_kv_cache(pipeline, quantizer=None) -> None:
    """Commit the final mutable write buffer after generation finishes."""
    if quantizer is None:
        return
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            state = block.get("quant_state")
            if state is not None:
                quantizer.finalize_state(state, meta={"tensor_dtype": block["dtype"]})


def resident_kv_memory_bytes(pipeline, quantizer=None) -> tuple[int, int]:
    """BF16 equivalent and resident bytes under ``resident_analytic.v1``.

    The equivalent is the BF16 cost of the cache positions actually held, the
    same figure the QVG breakdown and the Self-Forcing record report, so every
    method and the BF16 row answer the same question.  Counting the
    preallocated capacity instead tied the ratio to the cache allocation: a
    run whose window differed from its video length reported a compression
    ratio that had nothing to do with the quantizer.
    """
    bf16_bytes = 0
    resident_bytes = 0
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            state = block.get("quant_state")
            if quantizer is None:
                batch_size, _, num_heads, head_dim = block["k"].shape
                tokens = int(block["local_end_index"])
            else:
                batch_size, num_heads, head_dim = (
                    block["batch_size"], block["num_heads"], block["head_dim"]
                )
                tokens = 0 if state is None else int(state["num_tokens"])
            block_bf16_bytes = batch_size * tokens * num_heads * head_dim * 2 * 2
            bf16_bytes += block_bf16_bytes
            if quantizer is None:
                resident_bytes += block_bf16_bytes
            elif state is not None:
                resident_bytes += int(quantizer.memory_bytes(state))
    return int(bf16_bytes), int(resident_bytes)
