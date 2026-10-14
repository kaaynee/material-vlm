# Shared environment for Euler. Source this file, do not execute it.
#
# Unversioned names load Euler's current defaults. To pin versions, check
# `module avail stack` and `module spider python`, then e.g.:
#   export EULER_MODULES="stack/<version> gcc/<version> python/<version> eth_proxy"
EULER_MODULES="${EULER_MODULES:-stack gcc python eth_proxy}"
module load $EULER_MODULES || return 1

# Virtual environment location (several GB once torch is installed).
export VENV_DIR="${VENV_DIR:-$PWD/.venv}"

# Model weights are large: keep the Hugging Face cache on scratch, not in $HOME.
# Note that $SCRATCH is purged regularly, so weights may be re-downloaded.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
