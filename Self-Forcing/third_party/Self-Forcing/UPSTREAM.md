# Upstream

- Repository: <https://github.com/guandeh17/Self-Forcing>
- Commit: `33593df3e81fa3ec10239271dd2c100facac6de1` ("Clean", 2025-09-11)
- License: see `LICENSE` (unchanged)

The tree is an export of that commit, trimmed to the inference path (training,
demo and data-preparation code removed), vendored so that the code which
runs is the code in this repository (previously it was cloned and patched on
each machine, which let server copies drift from the repository).

## Local modifications

`pipeline/causal_inference.py`: the block loop shows a `tqdm` progress bar
(one step per generated block) instead of printing every denoising timestep.

`wan/modules/causal_model.py` adds the KV-cache quantization hook to
`CausalWanSelfAttention`:

- `_attention_with_incremental_cache`: append-only cache for RTN, KIVI and
  QuaRot.  Keys are cached after RoPE; history is never re-quantized; only the
  attention window (`start_token`) is decoded.  QuaRot additionally rotates
  Q/K/V and restores the attention output.

To see the exact change:

```bash
git diff --no-index <upstream-checkout>/wan/modules/causal_model.py wan/modules/causal_model.py
```
