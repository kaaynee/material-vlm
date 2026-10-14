# Shared environment for Euler. Source this file, do not execute it.
# Check available versions with `module spider python` if a module is missing.

module load stack/2024.06 gcc/12.2.0 python/3.11.6 eth_proxy

# Virtual environment location (several GB once torch is installed).
export VENV_DIR="${VENV_DIR:-$PWD/.venv}"

# Model weights are large: keep the Hugging Face cache on scratch, not in $HOME.
# Note that $SCRATCH is purged regularly, so weights may be re-downloaded.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
