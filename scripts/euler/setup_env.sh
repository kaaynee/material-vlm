#!/bin/bash
# One-time setup on an Euler login node. Run from the repository root:
#   bash scripts/euler/setup_env.sh
set -euo pipefail

source scripts/euler/env.sh

if ! python -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Python >= 3.10 required, got $(python --version 2>&1). Set EULER_MODULES (see scripts/euler/env.sh)." >&2
    exit 1
fi

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -e ".[dev,qwen35]"
# Optional, compiled against CUDA: the remaining Qwen3.5 kernel. Failure is fine.
pip install causal-conv1d --no-build-isolation \
    || echo "causal-conv1d not installed; Qwen3.5 still works, slightly slower"

mkdir -p "$HF_HOME" logs outputs
echo "Environment ready in $VENV_DIR (HF cache: $HF_HOME)"
