# Getting cuequivariance working on Alliance Canada clusters (Narval, Fir)

This documents everything that turned out to be necessary to get
`--enable_cueq true` actually running the fused `uniform_1d` CUDA kernels
(instead of silently falling back to a slow, pure-Python "naive"
implementation) on Alliance Canada's Narval and Fir clusters, CUDA 13.2,
Python 3.11. None of this is MACE-XDM-specific -- it's entirely about how
`cuequivariance-ops-torch` gets installed and linked on these clusters. If
you're setting up a new venv, or `cuequivariance_ops_torch` stops working
again after some update, this is the troubleshooting trail, in the order we
actually walked it, with the reasoning kept in so future-you can tell which
steps still apply.

## The symptom

Training with `--enable_cueq true` either:

- crashed with `AttributeError: 'SegmentedPolynomialNaive' object has no
  attribute 'buffer_num_segments'` (older `wrapper_ops.py`), or
- crashed with `RuntimeError: cuEquivariance conv fusion needs the
  uniform_1d kernels, but cuequivariance selected method='naive'` (current
  `wrapper_ops.py` -- this is a deliberate loud failure we added; see
  "Why cueq fails loudly instead of silently" below).

Both mean the same underlying thing: `cuequivariance_ops_torch` (the
compiled CUDA kernel package) could not actually be used, so
`cuet.SegmentedPolynomial` silently downgraded to a naive Python
implementation that doesn't have the attributes/behavior MACE's wrapper
expects.

## Why cueq fails loudly instead of silently

`mace/modules/wrapper_ops.py`'s `CueqConvFusionWrapper` explicitly checks
`conv_tp.method != "uniform_1d"` and raises, rather than letting the
naive fallback run unfused. This is intentional: `--enable_cueq true` is
requested explicitly to get GPU acceleration, so silently training unfused
would waste GPU-hours without the acceleration it was asked for, and the
mismatch is easy to miss in a long training log otherwise. If you ever want
the old (bad) silent-fallback behavior back, don't -- fix the install
instead; that's what this doc is for.

## Diagnosis chain (read this before re-running commands blindly)

Each of the following was a genuinely separate problem, discovered one
layer at a time by actually testing after each fix rather than guessing
the whole way through. The order matters: earlier problems mask later ones.

### 1. `torch.version.cuda` vs. loaded module version mismatch

Check what CUDA version your venv's torch was actually built against:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The loaded `module load cuda/X.Y` **must match this exactly**. We had a
script loading `cuda/12.6` while torch was built for CUDA 13.2 -- wrong
module version alone is enough to produce the naive-fallback symptom, even
before any of the packaging issues below.

### 2. `~/.local` shadowing the venv

Python includes `~/.local/lib/pythonX.Y/site-packages` (user-site) on
`sys.path` **even inside an activated venv**, unless `PYTHONNOUSERSITE=1`
is set. If an older/broken `cuequivariance_ops_torch` was ever installed
with `pip install --user` (e.g. from earlier troubleshooting in a
different venv), it can silently shadow the venv's own copy. Symptom: a
traceback showing `/home/<user>/.local/lib/...` paths instead of
`<venv>/lib/.../site-packages/...`.

Fix: `export PYTHONNOUSERSITE=1` before activating the venv. Verify with:

```bash
which python
python -c "import cuequivariance_torch; print(cuequivariance_torch.__file__)"
```

Both should point inside the venv, never `~/.local`.

### 3. Alliance Canada's pip is pinned to a local wheelhouse

```bash
pip config list
```

shows `PIP_CONFIG_FILE=/cvmfs/soft.computecanada.ca/config/python/pip-*.conf`,
which sets `find-links` to `/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/...`.
This wheelhouse does not carry `cuequivariance-ops-torch-cu13` /
`cuequivariance-ops-cu13` (very new, niche CUDA-13-specific packages).
`pip install --index-url https://pypi.org/simple <pkg>` looks like it
should bypass this, but real internet access from a login node was
confirmed working (`curl -sI https://pypi.org/simple/<pkg>/` returns
`HTTP/2 200`) while `pip install` still failed -- see #4, a *different*
problem than network access.

### 4. pip's manylinux tag detection fails on this Python build

Even with the correct wheel downloaded directly (`curl`), plain
`pip install ./that-file.whl` failed with:

```
ERROR: <file>.whl is not a supported wheel on this platform.
```

`ldd --version` showed glibc 2.37 -- comfortably newer than the wheel's
`manylinux_2_27`/`manylinux_2_28` requirement, so the actual binary is
compatible. `python -m pip debug --verbose | grep -i manylinux` returned
**nothing** -- pip's own compatible-tag list contains zero manylinux tags
on this Gentoo-based Python build, a platform-detection quirk (relies on
`platform.libc_ver()`, which doesn't populate correctly here). No CLI flag
fixes this cleanly.

**Fix: skip pip's install path entirely.** A wheel is just a zip file with
a known layout (`<package>/` + `<dist>-<version>.dist-info/`) -- download
it with `curl` and `unzip` it straight into the venv's `site-packages`,
bypassing pip's tag check (and the wheelhouse/index resolution from #3)
completely:

```bash
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

HREF=$(curl -s "https://pypi.org/simple/<pkg-index-name>/" \
  | grep -oE 'href="[^"]*<version>-<tag>[^"]*"' | head -1)
WHEEL_URL=$(echo "$HREF" | sed -E 's/^href="//; s/"$//; s/#.*$//')
curl -L -O "$WHEEL_URL"          # -O (capital) keeps the real filename --
                                  # a hand-typed/truncated filename gets
                                  # rejected by pip as an "invalid wheel
                                  # filename" before it even opens it
unzip -o "$(basename "$WHEEL_URL")" -d "$SITE_PACKAGES"
```

See `scripts/install_cueq_ops.sh` in this repo for the exact, tested
version of this for both packages needed
(step 5).

### 5. Two separate packages, not one

`cuequivariance-ops-torch-cu13` (Python wrapper, `cp3XX`-tagged: tied to a
specific Python minor version) depends on **a separate package**,
`cuequivariance-ops-cu13` (the actual `libcue_ops.so`, `py3-none`-tagged:
Python-version-agnostic). Installing only the first gives:

```
ModuleNotFoundError: No module named 'cuequivariance_ops'
```

Both need the download-and-unzip treatment from step 4. Confirm the
package name is right before chasing wheel URLs -- search PyPI's project
index directly if `pip index versions` gives a suspicious "no matching
distribution" for something you're fairly sure exists:

```bash
curl -s https://pypi.org/simple/ | grep -i "cuequivariance-ops"
```

### 6. `libcue_ops.so` needs CUDA 13 runtime libraries on `LD_LIBRARY_PATH`

Once both packages are actually installed, importing gets further but
`libcue_ops.so` itself needs `libcublas.so.13` / `libcublasLt.so.13` /
`libnvrtc.so.13` at runtime (not link time -- `pip`/`import` succeed even
without them; the failure only shows up as an `ldd`-style missing-symbol
error, or later as a `RuntimeError` when MACE actually tries to build a
model).

Find where they actually live by checking what the **already-working**
torch CUDA extension resolves them from (torch itself works fine, so its
own dependency resolution is a reliable source of truth for the right
path on *this specific cluster*):

```bash
module load cuda/13.2
ldd $(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_cuda.so'))") \
  | grep -E 'nvrtc|cublas'
```

On both Narval and Fir this resolved to the same path (they share a CVMFS
software tree):

```
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0/lib64
```

**Don't assume this path is universal** -- it happened to match on both
clusters we tried, but re-derive it with the `ldd` command above on any
new cluster rather than copying the path blind; a different
software-stack year or CPU microarchitecture level
(`x86-64-v3`/`x86-64-v4`) would change it.

Fix:

```bash
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0/lib64:$LD_LIBRARY_PATH
```

Verify with:

```bash
LIBCUE_OPS=$(find "$SITE_PACKAGES" -iname "libcue_ops*" | head -1)
ldd "$LIBCUE_OPS" | grep "not found"   # must print nothing
```

### 7. A few remaining plain-Python dependencies

Once the compiled pieces resolve, `import cuequivariance_ops_torch` may
still fail on ordinary (non-CUDA-specific) missing packages that pip's own
dependency resolver would normally have installed automatically, had it
been able to resolve the package at all (steps 3-4 prevented that). These
install normally through the wheelhouse, no workaround needed:

```bash
pip install pynvml platformdirs nvidia-cublas
```

(`pip install <compiled-pkg>` itself will actually list exactly which of
these it's missing in its "dependency conflicts" warning, once the package
is present via the unzip workaround -- read that output rather than
guessing.)

## Making it permanent

Two different lifetimes, don't confuse them:

- **Permanent already:** everything `pip install`ed / unzipped into
  `site-packages` -- that's part of the venv, persists across shells and
  SLURM jobs automatically.
- **Not permanent unless you do something:** `module load cuda/13.2` and
  `export LD_LIBRARY_PATH=...` only apply to the current shell. Every new
  login, every new SLURM job starts fresh.

For SLURM jobs, don't rely on `~/.bashrc` (batch-job sourcing behavior is
inconsistent across cluster configs) -- either:

- keep `module load cuda/13.2` + the `export LD_LIBRARY_PATH=...` line
  explicitly in every job script (simple, visible, slightly repetitive), or
- run `scripts/append_to_activate.sh` (this repo) once, which appends both
  lines directly to the venv's own `bin/activate`, so every future
  `source <venv>/bin/activate` -- interactive or inside a job script --
  sets them automatically. This is what we actually did; job scripts can
  now drop the explicit lines if you want, though leaving them in is
  harmless (just a duplicate `export`) and documents the dependency inline.

## Verifying cueq actually works (not just imports)

Import success alone isn't enough -- `cuequivariance_ops_torch` can import
fine while still resolving to the naive backend if the *loaded* CUDA
module doesn't match torch's build (step 1) even after everything else is
fixed. Always verify with an actual fused-kernel construction, the same
code path `--enable_cueq true` uses:

```bash
python - <<'EOF'
from mace.modules.wrapper_ops import TensorProduct, CuEquivarianceConfig
cfg = CuEquivarianceConfig(enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True, conv_fusion=True)
tp = TensorProduct("32x0e + 32x1o", "1x0e + 1x1o", "32x0e + 32x1o", cueq_config=cfg)
print("SUCCESS, weight_numel =", tp.weight_numel)
EOF
```

`SUCCESS` means the fused `uniform_1d` kernel path is genuinely active. A
`RuntimeError: ... method='naive'` means something above regressed --
re-check step 1 (module/torch CUDA version match) first, since that's the
easiest thing to silently drift (e.g. copy-pasting a job script from an
older CUDA-version setup).

## Rebuilding from scratch

If this venv is ever rebuilt, `scripts/install_cueq_ops.sh` (this repo)
automates steps 4 and 7 above (the actual package installation) in one
script -- pass the target venv already activated. Step 6 (`LD_LIBRARY_PATH`)
still needs to be re-derived per-cluster with the `ldd` command in that
section, since it's not something a script run from inside the venv can
discover about the host system without you first loading the matching
`module load cuda/X.Y`.
