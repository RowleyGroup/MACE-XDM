"""Summarise the water-ball Langevin benchmark (100,000 steps, 1 fs, H100) as a few fitted numbers
per model plus one figure. Usage (from anywhere):  python bench/analyze_bench.py

Per model we report
  floor_ms      per-step time in the latency-bound regime (median of the 3 smallest sizes)
  slope_us      marginal cost per atom-step in the throughput-bound regime (linear fit t = a + b*N
                over the upper part of the curve)
  N_cross       floor / slope: the size where the model stops being latency-bound
and, for the dispersion variants, the *additive* overhead (with - without) in ms/step.
For MACE+XDM the overhead is also regressed against the number of atom pairs within the 14 A
dispersion cutoff, computed here on ideal 1 g/cm3 water balls (same sizes as the benchmark).
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mace.data.neighborhood import get_neighborhood

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 100_000
FILES = {"ANI-PBE0": "anipbe0.csv", "ANI-PBE0+MLXDM": "mlxdm2x.csv",
         "MACE-PBE0": "macepbe0.csv", "MACE-PBE0+XDM": "macexdm.csv"}
# upper part of each curve used for the linear (throughput-bound) fit
FIT_FROM = {"ANI-PBE0": 6000, "ANI-PBE0+MLXDM": 3000, "MACE-PBE0": 1500}

ms = {k: pd.read_csv(os.path.join(HERE, v)).set_index("n_atoms")["time_seconds"] / STEPS * 1e3
      for k, v in FILES.items()}


def water_ball(n_waters, seed=0):
    rng = np.random.default_rng(seed)
    a = 30.0 ** (1 / 3)
    k = int(np.ceil((n_waters * 1.6) ** (1 / 3)))
    g = np.arange(-k, k + 1)
    pts = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T * a
    pts = pts[np.argsort(np.linalg.norm(pts, axis=1))][:n_waters]
    ref = np.array([[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    P = []
    for c in pts:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P.append(ref @ q.T + c + rng.normal(scale=0.05, size=(3, 3)))
    return np.concatenate(P)


def pair_search(P, cutoff=14.0):
    """The CPU part of XDMDispersionEnergy._sparse_pairs (per MD step, uncached)."""
    edge_index, _, _, _ = get_neighborhood(positions=P, cutoff=cutoff)
    keep = edge_index[0] < edge_index[1]
    torch.as_tensor(edge_index[0][keep], dtype=torch.long)
    torch.as_tensor(edge_index[1][keep], dtype=torch.long)
    return int(keep.sum())


def water_ball_pairs(n_waters):
    return pair_search(water_ball(n_waters))


def cpu_search_ms(n_waters, reps=3):
    P = water_ball(n_waters)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        pair_search(P)
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


rows = []
for name, t in ms.items():
    n = t.index.values.astype(float)
    floor = float(np.median(t.values[:3]))
    row = dict(model=name, max_atoms=int(n.max()), floor_ms=floor)
    if name in FIT_FROM:
        m = n >= FIT_FROM[name]
        b, a = np.polyfit(n[m], t.values[m], 1)
        row.update(slope_us_per_atom_step=b * 1e3, intercept_ms=a, N_cross=floor / b)
    rows.append(row)
summary = pd.DataFrame(rows).set_index("model")

over = pd.DataFrame({
    "MLXDM on ANI": (ms["ANI-PBE0+MLXDM"] - ms["ANI-PBE0"]).dropna(),
    "XDM on MACE": (ms["MACE-PBE0+XDM"] - ms["MACE-PBE0"]).dropna(),
})
nw = (over["XDM on MACE"].dropna().index.values // 3).astype(int)
pairs = pd.Series({int(3 * w): water_ball_pairs(int(w)) for w in nw})
cpu_ms = pd.Series({int(3 * w): cpu_search_ms(int(w)) for w in nw})
xdm = over["XDM on MACE"].dropna()
sparse = xdm.index.values > 512  # dispersion switches from dense-GPU to CPU pair search above 512 atoms
slope_pair, icpt_pair = np.polyfit(pairs[xdm.index].values[sparse], xdm.values[sparse], 1)
pred = icpt_pair + slope_pair * pairs[xdm.index].values[sparse]
r2 = 1 - ((xdm.values[sparse] - pred) ** 2).sum() / ((xdm.values[sparse] - xdm.values[sparse].mean()) ** 2).sum()

pd.set_option("display.width", 200)
print(summary.round(2).to_string())
print("\nadditive dispersion overhead (ms/step):")
print(over.round(1).T.to_string())
print("\nXDM overhead vs pairs within 14 A (sparse/CPU-search path, >512 atoms):")
print(f"  overhead_ms = {icpt_pair:.1f} + {slope_pair * 1e3:.3f} us/pair * pairs   (R^2 = {r2:.3f})")
print("  pairs:", pairs.to_dict())
print("\nCPU pair search alone, measured on THIS machine (ms/call) and as a fraction of the H100 XDM overhead:")
print(pd.DataFrame({"pairs": pairs, "cpu_search_ms": cpu_ms.round(1), "us_per_pair": (cpu_ms / pairs * 1e3).round(3),
                    "H100_overhead_ms": xdm.round(1), "cpu/overhead": (cpu_ms / xdm).round(2)}).to_string())
summary.round(3).to_csv(os.path.join(HERE, "bench_summary_fits.csv"))

# ------------------------------------------------------------------ figure
ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
blue, orange = "#2a78d6", "#eb6834"  # categorical slots 1-2: hue = model family, dash = dispersion added
style = {"ANI-PBE0": (blue, "-"), "ANI-PBE0+MLXDM": (blue, (0, (5, 2))),
         "MACE-PBE0": (orange, "-"), "MACE-PBE0+XDM": (orange, (0, (5, 2)))}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=surface)
for a_ in ax:
    a_.set_facecolor(surface)
    a_.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a_.spines[s].set_color("#c9c8c2")
    a_.tick_params(colors=ink2, labelsize=8)

for name, t in ms.items():
    c, ls = style[name]
    ax[0].plot(t.index, t.values, color=c, ls=ls, lw=1.8, marker="o", ms=3, label=name)
ax[0].set(xscale="log", yscale="log")
ax[0].set_title("A. Cost per MD step vs system size", loc="left", fontsize=10, color=ink)
ax[0].set_xlabel("atoms (water ball)", color=ink2, fontsize=9)
ax[0].set_ylabel("ms / step  (H100, 1 fs)", color=ink2, fontsize=9)
ax[0].legend(fontsize=8, frameon=False, labelcolor=ink2, loc="upper left")

for name, colr in (("MLXDM on ANI", blue), ("XDM on MACE", orange)):
    o = over[name].dropna()
    ax[1].plot(o.index, o.values, color=colr, ls=(0, (5, 2)), lw=1.8, marker="o", ms=3, label=name)
ax[1].axvline(512, color=ink2, lw=0.8, ls=":")
ax[1].set(xscale="log", yscale="log")
ax[1].text(540, 0.27, "512 atoms: XDM pair search\nmoves from GPU-dense to CPU", fontsize=7.5, color=ink2, va="bottom", transform=ax[1].get_xaxis_transform())
ax[1].set_title("B. What dispersion adds (with - without), ms/step", loc="left", fontsize=10, color=ink)
ax[1].set_xlabel("atoms (water ball)", color=ink2, fontsize=9)
ax[1].legend(fontsize=8, frameon=False, labelcolor=ink2, loc="upper left")

xs = pairs[xdm.index].values
ax[2].plot(xs[sparse], xdm.values[sparse], color=orange, ls="none", marker="o", ms=4, label="XDM overhead (>512 atoms)")
ax[2].plot(xs[~sparse], xdm.values[~sparse], color=orange, ls="none", marker="s", ms=4, mfc="none", label="dense path (300 atoms)")
ax[2].plot(xs, cpu_ms[xdm.index].values, color=ink2, marker="^", ms=4, lw=1.0, ls="-", label="CPU pair search alone (measured on this workstation)")
ax[2].set_title("C. XDM overhead vs pairs inside the 14 A cutoff", loc="left", fontsize=10, color=ink)
ax[2].set_xlabel("atom pairs within 14 A (ideal 1 g/cm$^3$ water ball)", color=ink2, fontsize=9)
ax[2].set_ylabel("ms per step (H100 overhead)  /  ms per call (CPU search)", color=ink2, fontsize=8)
ax[2].legend(fontsize=8, frameon=False, labelcolor=ink2, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "benchmark_story.png"), dpi=140, facecolor=surface)
