# KV-Quant4VideoGen

自回归视频扩散模型的 KV cache 量化 baseline。

[English](README.md)

在 **Self-Forcing**、**Causal-Forcing**、**LongCat-Video** 上实现 RTN、KIVI、QuaRot 三种 KV cache
量化，Causal-Forcing 上另有官方 **Quant-VideoGen（QVG）**。三个模型共用 [`kv_quant/`](kv_quant/) 里的
同一份量化器。

## 方法

| `--method` | 量化器 |
| --- | --- |
| `BF16` | 不压缩的对照 |
| `RTN_INT{4,2}` | per-token 非对称 round-to-nearest，64 通道一组 |
| `KIVI_INT{4,2}` | K 按通道在 64 token 一组上量化，V 按 token 量化，最近 128 token 保留 BF16 |
| `QUAROT_KV_INT{4,2}` | RoPE 后对 Q/K 以及 V 做 Hadamard，再按 64 一组非对称量化，clip ratio 0.95 |
| `QVG_INT{2,4}` | 官方 Quant-VideoGen，仅 Causal-Forcing |

每条 baseline 都是每 64 个值一个 BF16 scale 加一个 BF16 zero，所以 INT4 都是 4.5 bit（3.56×），
INT2 都是 2.5 bit（6.40×）。生成协议与 Tempokv / QVG 对齐：MovieGen-128 prompt、种子
`seed + prompt_index * 1000003`、全历史注意力、180 个 latent 帧（717 帧）。

## 准备

```bash
pip install -r Self-Forcing/requirements-inference.txt -r Causal-Forcing/requirements.txt
pip install -r Causal-Forcing/requirements-qvg.txt      # QVG 与 LongCat
pip install flash-attn --no-build-isolation
pip install -r Self-Forcing/requirements-eval.txt       # VBench / 保真度

bash scripts/download_checkpoints.sh                    # Wan2.1、Self-Forcing、Causal-Forcing
bash scripts/download_checkpoints.sh longcat            # LongCat-Video
bash Self-Forcing/scripts/setup_vbench.sh               # 仅评测需要
```

权重放在 `wan_models/` 和 `checkpoints/`，输出在 `results/`。所有命令在仓库根目录下运行。

## 运行

```bash
# Self-Forcing：生成 → VBench → 对 BF16 的保真度 → 汇总表
bash Self-Forcing/run_baseline_matrix.sh
METHODS="BF16 KIVI_INT2" GPUS=0,1 MAX_PROMPTS=4 bash Self-Forcing/run_baseline_matrix.sh

# Causal-Forcing（含 QVG）
CHECKPOINT_PATH=checkpoints/causal_forcing/framewise/causal_forcing.pt \
  bash Causal-Forcing/run_baseline_matrix.sh

# LongCat：先用 BF16 生成初始视频，所有方法从它续写
torchrun --nproc_per_node=1 LongCat/run_long_t2v.py --workload 480p_init --method BF16 \
  --quant_type none --checkpoint_dir checkpoints/LongCat-Video --output_dir results/longcat/init \
  --prompt "A person walking through a sunlit forest"
CHECKPOINT_DIR=checkpoints/LongCat-Video INIT_VIDEO_PATH=results/longcat/init/0-0.mp4 \
  bash LongCat/run_baseline_matrix.sh
```

各 launcher 的环境变量（`METHODS`、`GPUS`、`STAGES`、`MAX_PROMPTS` …）写在脚本开头。单次运行：
`Self-Forcing/scripts/01_generate.py`、`Causal-Forcing/inference.py`、`LongCat/run_long_t2v.py`。

`--block_size` 改 RTN 的通道分组和 KIVI 的 key 分组；QuaRot 的分组由 `--kv_channel_group_size` 决定。

## 输出

每次运行在 `results/<model>/` 下写出视频和 JSON 报告（运行时间、峰值显存、cache 字节）。压缩比用
`resident_analytic.v1` 口径：当前持有的 cache 位置换算成 BF16 的字节数，除以同一时刻的常驻字节。

论文实验：[`efficiency`](Self-Forcing/scripts/paper_experiments/efficiency/run_efficiency.sh)、
[`latency`](scripts/paper_experiments/latency/README.md)、
[`reconstruction`](scripts/paper_experiments/reconstruction/README.md)。
