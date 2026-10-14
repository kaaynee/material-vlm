#!/bin/bash
# One-time setup on an Euler login node. Run from the repository root:
#   bash scripts/euler/setup_env.sh
set -euo pipefail

source scripts/euler/env.sh

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -e ".[dev]"

mkdir -p "$HF_HOME" logs outputs
echo "Environment ready in $VENV_DIR (HF cache: $HF_HOME)"
