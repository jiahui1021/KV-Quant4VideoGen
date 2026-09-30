#!/usr/bin/env bash
# Causal-Forcing baseline matrix: one video set and one report per method.
#
#   CHECKPOINT_PATH=checkpoints/causal_forcing/framewise/causal_forcing.pt \
#     bash Causal-Forcing/run_baseline_matrix.sh
#   METHODS=QVG_INT2 CONFIG_PATH=Causal-Forcing/configs/causal_forcing_dmd_chunkwise.yaml \
#     CHECKPOINT_PATH=checkpoints/causal_forcing/chunkwise/causal_forcing.pt \
#     bash Causal-Forcing/run_baseline_matrix.sh
#
# Environment (default):
#   CHECKPOINT_PATH        required
#   CONFIG_PATH            model config (configs/causal_forcing_dmd_framewise.yaml)
#   METHODS                methods to run (BF16, the six baselines, QVG_INT2, QVG_INT4)
#   DATA_PATH              prompts (Self-Forcing/prompts/moviegen_128.txt)
#   OUTPUT_ROOT            output root (results/causal_forcing)
#   NUM_OUTPUT_FRAMES      latent frames (180)
#   LOCAL_ATTN_SIZE        attention window in frames (-1, the full history)
#   RETAIN_FINAL_CACHE=1   keep the last block's cache for the resident-byte report
#   USE_EMA=0, PROFILE_QUANT_TIMING=0, QVG_DISABLE_COMPRESSION=0, PYTHON_BIN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repository root: Wan weights load from wan_models/

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the Causal-Forcing checkpoint}"
CONFIG_PATH="${CONFIG_PATH:-Causal-Forcing/configs/causal_forcing_dmd_framewise.yaml}"
DATA_PATH="${DATA_PATH:-Self-Forcing/prompts/moviegen_128.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/causal_forcing}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a METHODS <<< "${METHODS:-BF16 RTN_INT4 RTN_INT2 KIVI_INT4 KIVI_INT2 QUAROT_KV_INT4 QUAROT_KV_INT2 QVG_INT2 QVG_INT4}"
export CAUSAL_FORCING_BENCHMARK="${CAUSAL_FORCING_BENCHMARK:-1}"

common_args=(
  --config_path "${CONFIG_PATH}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --data_path "${DATA_PATH}"
  --num_output_frames "${NUM_OUTPUT_FRAMES:-180}"
  --local_attn_size "${LOCAL_ATTN_SIZE:--1}"
  --report_timing
)
if [[ "${RETAIN_FINAL_CACHE:-1}" == 1 ]]; then common_args+=(--retain_final_cache); fi
if [[ "${USE_EMA:-0}" == 1 ]]; then common_args+=(--use_ema); fi
if [[ "${PROFILE_QUANT_TIMING:-0}" == 1 ]]; then common_args+=(--profile_quant_timing); fi

for method in "${METHODS[@]}"; do
  echo "===== Causal-Forcing ${method} ====="
  method_args=(--method "${method}")
  # QVG rows use inference.py's --qvg_* defaults, the official Quant-VideoGen settings.
  if [[ "${method}" == QVG_* && "${QVG_DISABLE_COMPRESSION:-0}" == 1 ]]; then
    method_args+=(--qvg_disable_compression)
  fi
  "${PYTHON_BIN}" Causal-Forcing/inference.py \
    "${common_args[@]}" \
    "${method_args[@]}" \
    --output_folder "${OUTPUT_ROOT}/${method}"
done
