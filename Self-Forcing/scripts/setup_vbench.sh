#!/usr/bin/env bash
# Clone VBench into Self-Forcing/third_party/VBench for the 03/04 evaluation scripts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -d third_party/VBench/.git ]]; then
  git clone https://github.com/Vchitect/VBench.git third_party/VBench
fi
echo "VBench is at $(pwd)/third_party/VBench"
