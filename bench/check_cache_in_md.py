"""Does the dispersion pair cache actually engage in a REAL Langevin run, and what does it buy?

Runs the same short Langevin trajectory twice on your real models (settings match gpu_md_benchmark.py:
298.15 K, 1 fs, friction 0.002/fs): once with the dispersion pair cache on, once off. Counts how many
times the CPU neighbor search really ran and times each pass. Takes a minute or two, not 100k steps.

    python check_cache_in_md.py PBE0.model XDM.model water_ball_00800.xyz [--repo /path/to/MACE-XDM]
                                [--steps 200] [--device cuda] [--dense-max-nodes N]

Pick a structure ABOVE the dense threshold (default 2048 atoms, e.g. water_ball_00800.xyz = 2400 atoms);
below it the dispersion never uses the CPU search and the cache is irrelevant (the script says so).
--dense-max-nodes lowers the threshold so a small structure can exercise the sparse path for testing.
"""
import argparse
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("pbe0_model")
ap.add_argument("xdm_model")
ap.add_argument("xyz")
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=5, help="untimed steps first (cueq/autotune warm-up)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dense-max-nodes", type=int, default=None)
args = ap.parse_args()

sys.path.insert(0, os.path.join(args.repo, "scripts"))
try:
    import gpu_md
except ImportError as e:
    sys.exit(f"cannot import gpu_md from {args.repo}/scripts ({e!r}) -- pass --repo /path/to/the/checkout you benchmark with")
import mace.modules.xdm_dispersion as xd

print("mace          :", xd.__file__)
print("gpu_md        :", gpu_md.__file__)

Z, P, cell = gpu_md.read_xyz(args.xyz)
model = gpu_md.load_combined_model(args.pbe0_model, args.xdm_model, device=args.device)
disp = model.dispersion_energy
if args.dense_max_nodes is not None:
    disp.dense_max_nodes = args.dense_max_nodes
print(f"structure     : {args.xyz} ({len(Z)} atoms)")
print(f"dense_max_nodes={disp.dense_max_nodes}  use_position_cache(as loaded)={disp.use_position_cache}  "
      f"cache_skin={disp.cache_skin}")
sparse = len(Z) > disp.dense_max_nodes
if not sparse:
    print(f"\nNOTE: {len(Z)} atoms <= dense_max_nodes, so the dispersion uses the GPU dense path and the cache is "
          "irrelevant here. Use a bigger structure (or --dense-max-nodes) to test the cache.")

calls = {"n": 0}
real = xd.get_neighborhood


def counting(*a, **k):
    calls["n"] += 1
    return real(*a, **k)


xd.get_neighborhood = counting


def sync():
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()


def run(use_cache):
    disp.use_position_cache = use_cache
    disp._cache_ref_positions = disp._cache_idx_i = disp._cache_idx_j = None
    sysm = gpu_md.System(model, Z, P, cell=cell, pbc=(False, False, False), device=args.device)
    gpu_md.init_maxwell_boltzmann(sysm, temperature_K=298.15, seed=0)  # as in gpu_md_benchmark.py
    gpu_md.langevin(sysm, dt_fs=1.0, n_steps=args.warmup, temperature_K=298.15, friction=0.002, seed=0)
    calls["n"] = 0
    sync()
    t = time.perf_counter()
    gpu_md.langevin(sysm, dt_fs=1.0, n_steps=args.steps, temperature_K=298.15, friction=0.002, seed=1)
    sync()
    dt = time.perf_counter() - t
    return calls["n"], dt / args.steps * 1e3


try:
    n_on, ms_on = run(True)
    n_off, ms_off = run(False)
finally:
    xd.get_neighborhood = real

print(f"\ncache ON : {n_on:4d} dispersion CPU searches in {args.steps} steps"
      + (f"  (one per {args.steps / max(n_on, 1):.1f} steps)" if n_on else "") + f"   {ms_on:8.1f} ms/step")
print(f"cache OFF: {n_off:4d} dispersion CPU searches in {args.steps} steps"
      + f"                    {ms_off:8.1f} ms/step")
print(f"speedup from the cache: {ms_off / ms_on:.2f}x")
if sparse:
    if n_on >= args.steps:
        print("\nPROBLEM: the cache is rebuilding every step, so it is engaged but never reusing. "
              "Check cache_skin vs how far atoms move per step.")
    elif n_off < args.steps:
        print("\nNOTE: cache OFF ran fewer searches than steps, so the sparse path wasn't used; result is not meaningful.")
    else:
        print(f"\nOK: cache engaged ({n_on} searches vs {n_off}); timing above is the real per-step gain "
              "(includes the short-range model, so it is smaller than the dispersion-only gain).")
