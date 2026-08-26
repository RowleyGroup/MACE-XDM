#!/bin/bash
# Installs cuequivariance-ops-torch-cu13 + cuequivariance-ops-cu13 into the
# currently-activated venv, working around two Fir/Narval (Alliance Canada)
# specific issues discovered the hard way:
#
#  1. pip here is pinned to a local wheelhouse (via PIP_CONFIG_FILE) that
#     doesn't carry these packages, and even with --index-url pointed at
#     real PyPI, pip's manylinux platform-tag detection reports ZERO
#     supported manylinux tags on this Gentoo-based Python build even
#     though the actual glibc (2.37) is new enough -- so `pip install`
#     rejects the correct wheel as "not a supported wheel on this platform"
#     even when downloaded directly. Fix: download the wheel with curl and
#     unzip it straight into site-packages, bypassing pip's tag check
#     entirely (a wheel is just a zip file with a known layout).
#
#  2. The extracted packages need a few further plain-Python deps
#     (pynvml, platformdirs, nvidia-cublas) that ARE fine to pip install
#     normally through the wheelhouse.
#
# Run this with the target venv already activated.
#
# Usage: scripts/install_cueq_ops.sh [version]
#   version defaults to the currently-installed cuequivariance-torch's
#   version, so the ops packages stay in lockstep with it.
set -euo pipefail

VERSION=${1:-$(python -c "import importlib.metadata as m; print(m.version('cuequivariance-torch'))")}
PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Installing cuequivariance-ops(-torch)-cu13 ${VERSION} for ${PY_TAG}"

SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

fetch_and_extract() {
    local pkg_index_name=$1   # e.g. cuequivariance-ops-torch-cu13
    local match_pattern=$2    # substring to grep the wheel filename for

    href=$(curl -s "https://pypi.org/simple/${pkg_index_name}/" \
        | grep -oE "href=\"[^\"]*${match_pattern}[^\"]*\"" | head -1)
    if [ -z "$href" ]; then
        echo "ERROR: no wheel found for ${pkg_index_name} matching '${match_pattern}'" >&2
        echo "Check available wheels with:" >&2
        echo "  curl -s https://pypi.org/simple/${pkg_index_name}/ | grep -oE '[a-zA-Z0-9_.+-]+\\.whl'" >&2
        exit 1
    fi
    url=$(echo "$href" | sed -E 's/^href="//; s/"$//; s/#.*$//')
    filename=$(basename "$url")
    echo "Downloading ${filename}..."
    curl -sL -O "$url"
    echo "Extracting into ${SITE_PACKAGES}..."
    unzip -oq "$filename" -d "$SITE_PACKAGES"
}

fetch_and_extract "cuequivariance-ops-torch-cu13" "${VERSION}-${PY_TAG}-${PY_TAG}-manylinux_2_27_x86_64"
fetch_and_extract "cuequivariance-ops-cu13" "${VERSION}-py3-none-manylinux_2_24_x86_64"

cd - > /dev/null
rm -rf "$TMPDIR"

# Plain-Python deps cuequivariance_ops/cuequivariance_ops_torch need at
# import time -- these install fine through the normal wheelhouse.
pip install --quiet pynvml platformdirs nvidia-cublas

python -c "import cuequivariance_ops_torch; print('cuequivariance_ops_torch OK:', cuequivariance_ops_torch.__file__)"

echo
echo "Done. This only installed the packages -- you still need the CUDA-13"
echo "runtime libraries on LD_LIBRARY_PATH at run time (module load cuda/13.2"
echo "+ export LD_LIBRARY_PATH=.../cudacore/13.2.0/lib64), which is baked"
echo "into this venv's activate script if you ran append_to_activate.sh."
