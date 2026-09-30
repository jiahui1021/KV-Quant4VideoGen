#!/usr/bin/env bash
# Download the weights the experiments use, all under the repository root.
#
#   bash scripts/download_checkpoints.sh                 # wan + self_forcing + causal_forcing
#   bash scripts/download_checkpoints.sh longcat         # LongCat only (large)
#
#   wan             -> wan_models/Wan2.1-T2V-1.3B
#   self_forcing    -> checkpoints/self_forcing_dmd.pt
#   causal_forcing  -> checkpoints/causal_forcing/{framewise,chunkwise}/causal_forcing.pt
#   longcat         -> checkpoints/LongCat-Video/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v hf >/dev/null 2>&1; then
  download() { hf download "$@"; }
elif command -v huggingface-cli >/dev/null 2>&1; then
  download() { huggingface-cli download "$@"; }
else
  echo "Install the Hugging Face CLI first: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

targets=("$@")
if (( ${#targets[@]} == 0 )); then targets=(wan self_forcing causal_forcing); fi

for target in "${targets[@]}"; do
  echo "===== ${target} ====="
  case "${target}" in
    wan)
      download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B ;;
    self_forcing)
      download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir . ;;
    causal_forcing)
      download zhuhz22/Causal-Forcing framewise/causal_forcing.pt chunkwise/causal_forcing.pt \
        --local-dir checkpoints/causal_forcing ;;
    longcat)
      download meituan-longcat/LongCat-Video --local-dir checkpoints/LongCat-Video ;;
    *)
      echo "Unknown target: ${target} (expected wan, self_forcing, causal_forcing or longcat)" >&2
      exit 2 ;;
  esac
done
