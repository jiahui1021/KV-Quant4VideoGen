# Forcing-KV integration

This directory is a source-only snapshot of the Forcing-KV code, taken
without its Git history, and trimmed to the inference path.

Generated videos, checkpoints, evaluation results, and other binary artifacts
were intentionally left out of this copy. The runnable SF/CF smoke entry point
is:

```bash
Forcing-KV/scripts/run_sf_cf_717.sh --only both
```

It runs one 180-latent-frame / 717-pixel-frame video for Self-Forcing and
Causal-Forcing, then writes one report per workload under
`efficiency/compression_ratio_0.json`.

The reported ratio compares the requested 717-frame KV-cache capacity with the
actual live grouped-cache tensors:

```text
full requested KV cache as BF16 with all heads / actual resident KV tensor bytes
```

The denominator is obtained by walking the live K/V tensors, including the
head-group and dynamic-history buffers. The numerator represents the same
180-latent-frame cache capacity with all 12 heads stored as BF16. It does not
inspect or depend on `quantization_enabled`.
