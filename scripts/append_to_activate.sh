#!/bin/bash
# Run this ONCE to permanently wire the cueq CUDA-13 environment (see
# docs/cueq_setup_alliance_canada.md) into a venv's own activate script, so
# every future `source <venv>/bin/activate` sets it automatically -- no
# need to repeat module load / export lines in every job script or
# interactive shell by hand again.
#
# Usage: scripts/append_to_activate.sh /path/to/venv [cuda_module_version] [cudacore_lib_path]
#
# The two optional args let this be reused on a cluster/CUDA version other
# than the one this was first written against (Narval/Fir, CUDA 13.2) --
# re-derive cudacore_lib_path with the ldd command in
# docs/cueq_setup_alliance_canada.md's step 6 rather than guessing it.
set -euo pipefail

VENV_DIR=${1:?"Usage: $0 /path/to/venv [cuda_module_version] [cudacore_lib_path]"}
CUDA_MODULE_VERSION=${2:-13.2}
CUDACORE_LIB_PATH=${3:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0/lib64}

ACTIVATE="${VENV_DIR}/bin/activate"
if [ ! -f "$ACTIVATE" ]; then
    echo "ERROR: $ACTIVATE not found -- is $VENV_DIR a venv?" >&2
    exit 1
fi

MARKER="# >>> cueq cuda setup >>>"
if grep -qF "$MARKER" "$ACTIVATE"; then
    echo "Already applied to $ACTIVATE -- doing nothing."
    exit 0
fi

cat >> "$ACTIVATE" <<BLOCK

# >>> cueq cuda setup >>>
# Added by scripts/append_to_activate.sh -- see
# docs/cueq_setup_alliance_canada.md. Loading the CUDA module here (rather
# than requiring every job script to do it separately) matches this venv's
# torch build -- a mismatched module load is what caused
# cuequivariance_ops_torch to silently fall back to the naive backend
# originally. Check with: python -c "import torch; print(torch.version.cuda)"
module load cuda/${CUDA_MODULE_VERSION} 2>/dev/null || true
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${CUDACORE_LIB_PATH}:\${LD_LIBRARY_PATH:-}
# <<< cueq cuda setup <<<
BLOCK

echo "Appended cuda setup block to $ACTIVATE."
