"""Verify the XDM dispersion pair-search fixes (commit a6284ff) are actually live in THIS environment.

Run in the same venv you benchmark with, pointing at the checkout you benchmark from:
    python verify_dispersion_fix.py /path/to/MACE-XDM      # explicit (works from any location)
    python bench/verify_dispersion_fix.py                  # or: run from inside that checkout
The script can live anywhere. No model files or GPU needed. Every check prints PASS/FAIL with
what a FAIL means.
"""
import inspect
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def find_repo_root():
    """The checkout whose scripts/gpu_md.py the benchmark would import: explicit argument first, then the
    current directory, then this file's parent. Deliberately NOT derived from where `mace` was imported,
    since comparing the two is the point of check 1."""
    for c in ([os.path.abspath(sys.argv[1])] if len(sys.argv) > 1 else []) + [os.getcwd(), os.path.dirname(HERE)]:
        if os.path.isfile(os.path.join(c, "scripts", "gpu_md.py")):
            return c
    return None

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def git_head(path):
    try:
        d = os.path.dirname(os.path.abspath(path))
        out = subprocess.run(["git", "-C", d, "log", "-1", "--format=%h %s"], capture_output=True, text=True, timeout=10)
        top = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10)
        return f"{out.stdout.strip()}  [repo: {top.stdout.strip()}]"
    except Exception as e:  # noqa: BLE001
        return f"(git unavailable: {e})"


import mace  # noqa: E402
import mace.modules.xdm_dispersion as xd  # noqa: E402

print("mace package        :", mace.__file__)
print("xdm_dispersion.py   :", xd.__file__)
print("   git HEAD         :", git_head(xd.__file__))

repo = find_repo_root()
gpu_md = None
if repo is not None:
    sys.path.insert(0, os.path.join(repo, "scripts"))  # same trick gpu_md_benchmark.py uses
    try:
        import gpu_md  # noqa: E402
        print("gpu_md.py           :", gpu_md.__file__)
        print("   git HEAD         :", git_head(gpu_md.__file__))
    except Exception as e:  # noqa: BLE001
        print("gpu_md.py           : import FAILED:", repr(e))
        gpu_md = None
else:
    print("gpu_md.py           : NOT FOUND (looked at argv[1], the current directory, and this script's parent)")
print()

# 1. mace and gpu_md come from the same checkout (mismatch = new mace + old scripts, or vice versa)
if gpu_md is not None:
    xd_root = os.path.realpath(os.path.join(os.path.dirname(xd.__file__), "..", ".."))
    gm_root = os.path.realpath(os.path.join(os.path.dirname(gpu_md.__file__), ".."))
    check("mace and scripts/gpu_md.py come from the SAME checkout", xd_root == gm_root,
          f"mace root={xd_root}  scripts root={gm_root}" if xd_root != gm_root else "")
else:
    check("scripts/gpu_md.py located and importable", False,
          "run from inside the checkout you benchmark with, or pass its path: python verify_dispersion_fix.py /path/to/MACE-XDM")

# 2. the new constructor arguments / methods exist
cls = xd.XDMDispersionEnergy
sig = inspect.signature(cls.__init__).parameters
check("XDMDispersionEnergy has dense_max_nodes/use_position_cache/cache_skin args",
      all(k in sig for k in ("dense_max_nodes", "use_position_cache", "cache_skin")),
      "old xdm_dispersion.py (pre-a6284ff) is being imported" if "use_position_cache" not in sig else "")
check("XDMDispersionEnergy._DENSE_MAX_NODES == 2048 (was 512)", getattr(cls, "_DENSE_MAX_NODES", None) == 2048,
      f"got {getattr(cls, '_DENSE_MAX_NODES', None)}")
check("cache methods present (_cached_sparse_pairs, _fresh_sparse_pairs)",
      hasattr(cls, "_cached_sparse_pairs") and hasattr(cls, "_fresh_sparse_pairs"))

# 3. gpu_md.load_combined_model actually turns the cache on
if gpu_md is not None:
    src = inspect.getsource(gpu_md.load_combined_model)
    check("gpu_md.load_combined_model passes use_position_cache=True", "use_position_cache=True" in src,
          "scripts/gpu_md.py is an OLD copy -> cache never enabled above 2048 atoms (threshold bump alone still helps 600-2048)"
          if "use_position_cache=True" not in src else "")

# 4. behaviour: above the dense threshold, repeated calls on a slowly-moving structure must NOT re-run the CPU search
if "use_position_cache" in sig:
    from mace.data import default_mlxdm_2x_atomic_number_table, mlxdm_2x_polarizability_reference

    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    torch.set_default_dtype(torch.float64)
    n = 2500  # > 2048, so the sparse (CPU search) path is taken
    rng = np.random.RandomState(0)
    v = rng.randn(n, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pos0 = torch.tensor(v * (rng.rand(n, 1) ** (1 / 3)) * 22.0)  # ball of radius 22 A
    Z = rng.choice(z_table.zs, size=n)
    idx = np.array([z_table.zs.index(z) for z in Z])
    node_attrs = torch.zeros(n, len(z_table.zs), dtype=torch.float64)
    node_attrs[torch.arange(n), torch.tensor(idx)] = 1.0
    batch = torch.zeros(n, dtype=torch.long)
    xdm_atomic = torch.tensor(rng.rand(n, 4) * 5 + 1.0)

    calls = {"n": 0}
    real = xd.get_neighborhood

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    xd.get_neighborhood = counting
    try:
        def run(module, steps=6):
            calls["n"] = 0
            pos = pos0.clone()
            for _ in range(steps):
                pos = pos + torch.tensor(rng.randn(n, 3) * 0.002)  # ~MD-sized displacement per step
                module(positions=pos, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)
            return calls["n"]

        cached = cls(alpha_free=ref["alpha_free"], v_free=ref["v_free"], use_position_cache=True)
        plain = cls(alpha_free=ref["alpha_free"], v_free=ref["v_free"], use_position_cache=False)
        n_cached, n_plain = run(cached), run(plain)
        check(f"cache ON : CPU neighbor search ran {n_cached}x over 6 steps (want 1)", n_cached == 1)
        check(f"cache OFF: CPU neighbor search ran {n_plain}x over 6 steps (want 6)", n_plain == 6,
              "sanity check that the counter itself works")
    finally:
        xd.get_neighborhood = real

print()
print("ALL CHECKS PASSED - the fix is live in this environment." if all(results)
      else f"{results.count(False)} CHECK(S) FAILED - the fix is NOT fully live here; see FAIL lines above.")
sys.exit(0 if all(results) else 1)
