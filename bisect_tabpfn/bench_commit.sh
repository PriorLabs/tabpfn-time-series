#!/usr/bin/env bash
# Usage: bash bench_commit.sh <TabPFN git ref>...
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TABPFN_MODEL_CACHE_DIR="$HOME/tabpfn_cache"
# SkyPilot sets this to 1; CI runs torch with its default thread count.
unset OMP_NUM_THREADS
cd "$(dirname "$0")"
for ref in "$@"; do
  echo "=== $ref"
  venv="$HOME/venvs/$ref"
  if [ ! -d "$venv" ]; then
    uv venv -q --python 3.12 "$venv"
    VIRTUAL_ENV="$venv" uv pip install -q --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.10.0" pandas pyarrow "tabpfn @ git+https://github.com/PriorLabs/TabPFN@$ref"
  fi
  timeout 900 "$venv/bin/python" bench.py 2>&1 | grep -E "^(tabpfn|RESULT)|Error|Traceback" || echo "FAILED $ref"
done
