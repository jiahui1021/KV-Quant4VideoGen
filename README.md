# KV-Quant4VideoGen

KV-cache quantization baselines for autoregressive video diffusion.

[中文](README.zh-CN.md)

RTN, KIVI and QuaRot KV-cache quantization on **Self-Forcing**, **Causal-Forcing** and
**LongCat-Video**, plus the official **Quant-VideoGen (QVG)** on Causal-Forcing. All three
models share one quantizer implementation in [`kv_quant/`](kv_quant/).

## Methods

| `--method` | Quantizer |
| --- | --- |
| `BF16` | Uncompressed reference |
| `RTN_INT{4,2}` | Per-token asymmetric round-to-nearest, groups of 64 channels |
| `KIVI_INT{4,2}` | K per channel over 64-token groups, V per token, last 128 tokens in BF16 |
| `QUAROT_KV_INT{4,2}` | Hadamard on post-RoPE Q/K and on V, then asymmetric groups of 64, clip ratio 0.95 |
| `QVG_INT{2,4}` | Official Quant-VideoGen, Causal-Forcing only |

Every baseline stores one BF16 scale and one BF16 zero per 64 values, so INT4 is 4.5 bits
(3.56×) and INT2 is 2.5 bits (6.40×) for all of them. Generation follows the Tempokv / QVG
protocol: MovieGen-128 prompts, seed `seed + prompt_index * 1000003`, full-history
attention, 180 latent frames (717 frames).

## Setup

```bash
pip install -r Self-Forcing/requirements-inference.txt -r Causal-Forcing/requirements.txt
pip install -r Causal-Forcing/requirements-qvg.txt      # QVG and LongCat
pip install flash-attn --no-build-isolation
pip install -r Self-Forcing/requirements-eval.txt       # VBench / fidelity

bash scripts/download_checkpoints.sh                    # Wan2.1, Self-Forcing, Causal-Forcing
bash scripts/download_checkpoints.sh longcat            # LongCat-Video
bash Self-Forcing/scripts/setup_vbench.sh               # evaluation only
```

Weights go to `wan_models/` and `checkpoints/`, outputs to `results/`. Run everything from
the repository root.

## Run

```bash
# Self-Forcing: generate → VBench → fidelity vs BF16 → summary table
bash Self-Forcing/run_baseline_matrix.sh
METHODS="BF16 KIVI_INT2" GPUS=0,1 MAX_PROMPTS=4 bash Self-Forcing/run_baseline_matrix.sh

# Causal-Forcing (QVG included)
CHECKPOINT_PATH=checkpoints/causal_forcing/framewise/causal_forcing.pt \
  bash Causal-Forcing/run_baseline_matrix.sh

# LongCat: one BF16 initial video, then every method continues it
torchrun --nproc_per_node=1 LongCat/run_long_t2v.py --workload 480p_init --method BF16 \
  --quant_type none --checkpoint_dir checkpoints/LongCat-Video --output_dir results/longcat/init \
  --prompt "A person walking through a sunlit forest"
CHECKPOINT_DIR=checkpoints/LongCat-Video INIT_VIDEO_PATH=results/longcat/init/0-0.mp4 \
  bash LongCat/run_baseline_matrix.sh
```

Each launcher lists its environment variables (`METHODS`, `GPUS`, `STAGES`, `MAX_PROMPTS`,
…) in its header. Single runs: `Self-Forcing/scripts/01_generate.py`,
`Causal-Forcing/inference.py`, `LongCat/run_long_t2v.py`.

`--block_size` changes the RTN channel group and the KIVI key group; QuaRot's group is set
by `--kv_channel_group_size`.

## Outputs

Each run writes videos plus a JSON report with runtime, peak memory and cache bytes under
`results/<model>/`. The compression ratio is `resident_analytic.v1`: BF16 bytes of the cache
positions held, divided by the bytes resident at that moment.

Paper experiments: [`efficiency`](Self-Forcing/scripts/paper_experiments/efficiency/run_efficiency.sh),
[`latency`](scripts/paper_experiments/latency/README.md),
[`reconstruction`](scripts/paper_experiments/reconstruction/README.md).
