#!/usr/bin/env bash
# Self-Forcing baseline matrix: generate, VBench, fidelity against BF16, summary table.
#
#   bash Self-Forcing/run_baseline_matrix.sh
#   METHODS="RTN_INT2 KIVI_INT2" GPUS=0,1 STAGES=generate bash Self-Forcing/run_baseline_matrix.sh
#
# Environment (default):
#   METHODS            methods to run (BF16 and the six INT4/INT2 baselines)
#   GPUS               comma-separated GPU ids; methods run in parallel, one per GPU (0)
#   STAGES             subset of generate,vbench,fidelity,summary (all four)
#   RUN_ROOT           output root (results/self_forcing)
#   BF16_DIR           fidelity reference videos ($RUN_ROOT/videos/BF16)
#   PROMPT_FILE        prompts (Self-Forcing/prompts/moviegen_128.txt)
#   MAX_PROMPTS        number of prompts (all)
#   NUM_OUTPUT_FRAMES  latent frames (180, i.e. 717 video frames)
#   SEED, BLOCK_SIZE, PYTHON_BIN, EVAL_PYTHON
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repository root: Wan weights load from wan_models/

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PYTHON="${EVAL_PYTHON:-${PYTHON_BIN}}"
RUN_ROOT="${RUN_ROOT:-results/self_forcing}"
PROMPT_FILE="${PROMPT_FILE:-Self-Forcing/prompts/moviegen_128.txt}"
BF16_DIR="${BF16_DIR:-${RUN_ROOT}/videos/BF16}"
STAGES="${STAGES:-generate,vbench,fidelity,summary}"
read -r -a METHODS <<< "${METHODS:-BF16 RTN_INT4 RTN_INT2 KIVI_INT4 KIVI_INT2 QUAROT_KV_INT4 QUAROT_KV_INT2}"
IFS=',' read -r -a GPUS <<< "${GPUS:-0}"
mkdir -p "${RUN_ROOT}/logs"

stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

gen_args=(
  --seed "${SEED:-0}"
  --num-output-frames "${NUM_OUTPUT_FRAMES:-180}"
  --prompt-path "${PROMPT_FILE}"
  --results-root "${RUN_ROOT}"
)
if [[ -n "${MAX_PROMPTS:-}" ]]; then gen_args+=(--max-prompts "${MAX_PROMPTS}"); fi
if [[ -n "${BLOCK_SIZE:-}" ]]; then gen_args+=(--block-size "${BLOCK_SIZE}"); fi

if stage generate; then
  for ((i = 0; i < ${#METHODS[@]}; i += ${#GPUS[@]})); do
    pids=()
    for ((slot = 0; slot < ${#GPUS[@]} && i + slot < ${#METHODS[@]}; slot++)); do
      method="${METHODS[i + slot]}"
      echo "[generate] ${method} on GPU ${GPUS[slot]} -> ${RUN_ROOT}/logs/generate_${method}.log"
      CUDA_VISIBLE_DEVICES="${GPUS[slot]}" "${PYTHON_BIN}" Self-Forcing/scripts/01_generate.py \
        --method "${method}" "${gen_args[@]}" > "${RUN_ROOT}/logs/generate_${method}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    if (( failed )); then
      echo "Generation failed; see ${RUN_ROOT}/logs/generate_*.log" >&2
      exit 1
    fi
  done
fi

if stage vbench; then
  for method in "${METHODS[@]}"; do
    echo "[vbench] ${method}"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHON_BIN="${EVAL_PYTHON}" bash Self-Forcing/scripts/03_eval_vbench.sh \
      "${method}" "${RUN_ROOT}/videos/${method}" "${PROMPT_FILE}" \
      "${RUN_ROOT}/metrics/vbench_${method}" "${RUN_ROOT}/metrics/vbench_${method}.json"
  done
fi

if stage fidelity; then
  bf16_count="$(find "${BF16_DIR}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
  for method in "${METHODS[@]}"; do
    [[ "${method}" == BF16 ]] && continue
    echo "[fidelity] ${method} vs ${BF16_DIR}"
    count="$(find "${RUN_ROOT}/videos/${method}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
    match_args=()
    if [[ "${count}" != "${bf16_count}" ]]; then
      # A MAX_PROMPTS run against a full BF16 set compares the shared prompts only.
      echo "  ${count} videos vs ${bf16_count} BF16 videos: comparing the intersection"
      match_args=(--match-intersection)
    fi
    "${PYTHON_BIN}" Self-Forcing/scripts/02_eval_fidelity.py \
      --bf16-dir "${BF16_DIR}" \
      --candidate-dir "${RUN_ROOT}/videos/${method}" \
      --output "${RUN_ROOT}/metrics/fidelity_${method}.json" \
      --device cpu \
      "${match_args[@]}"
  done
fi

if stage summary; then
  "${PYTHON_BIN}" Self-Forcing/scripts/05_summarize_results.py --results-root "${RUN_ROOT}" --methods "${METHODS[@]}"
fi
