#!/usr/bin/env bash
# LongCat baseline matrix: every method continues the same initial video.
#
#   CHECKPOINT_DIR=checkpoints/LongCat-Video INIT_VIDEO_PATH=results/longcat/init/0-0.mp4 \
#     bash LongCat/run_baseline_matrix.sh
#
# Environment (default):
#   CHECKPOINT_DIR     required, the LongCat-Video checkpoint directory
#   INIT_VIDEO_PATH    required, the shared initial video (see the README)
#   METHODS            methods to run (BF16 and the six baselines)
#   OUTPUT_ROOT        output root (results/longcat)
#   NUM_SEGMENTS       continuation segments (8)
#   NPROC_PER_NODE     processes for torchrun (1)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the LongCat checkpoint directory}"
: "${INIT_VIDEO_PATH:?Set INIT_VIDEO_PATH to the shared initial video}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/longcat}"
read -r -a METHODS <<< "${METHODS:-BF16 RTN_INT4 RTN_INT2 KIVI_INT4 KIVI_INT2 QUAROT_KV_INT4 QUAROT_KV_INT2}"

for method in "${METHODS[@]}"; do
  echo "===== LongCat ${method} ====="
  torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" LongCat/run_long_t2v.py \
    --workload 480p_long_gen \
    --context_parallel_size 1 \
    --method "${method}" \
    --quant_type none \
    --no_offload_kv_cache \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --init_video_path "${INIT_VIDEO_PATH}" \
    --num_segments "${NUM_SEGMENTS:-8}" \
    --output_dir "${OUTPUT_ROOT}/${method}"
done
