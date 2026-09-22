"""Label each density time series as converged / vaporized / etc. and plot them.

Usage (from this directory):  python label_density_runs.py

The *.density.csv files are chains of restart segments appended together: the
Step column resets to 0 at every restart, and some restarts jump back to the
initial packing instead of continuing. Classification therefore uses only the
"production chain" = the last run of segments whose density is continuous across
restarts (no >5% jump between the end of one segment and the start of the next).

Labels
  VAPORIZED (unconverged)  chain-end density < 25% of chain-start density
  SHORT (too short to judge)  production chain < 100,000 MD steps
  UNCONVERGED (drifting)   last-half block means trend >2% with p < 0.01
  CONVERGED (marginal drift)  same trend but 0.01 <= p < 0.1
  CONVERGED                otherwise
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_CHAIN_STEPS = 100_000
EXCLUDE = {"thiophene"}  # abandoned; raw files left untouched
JUMP_TOL = 0.05


def load(path):
    step, vol, rho, temp = [], [], [], []
    with open(path) as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            try:
                s, v, d = int(float(p[0])), float(p[1]), float(p[2])
            except (ValueError, IndexError):
                continue
            step.append(s); vol.append(v); rho.append(d)
            temp.append(float(p[3]) if len(p) > 3 and p[3] else np.nan)
    return pd.DataFrame(dict(step=step, vol=vol, rho=rho, T=temp))


def segments(d):
    cuts = np.where(np.diff(d.step.values) <= 0)[0] + 1
    return np.split(np.arange(len(d)), cuts)


def production_chain(d):
    idx, rho = segments(d), d.rho.values
    first = 0
    for i in range(len(idx) - 1):
        r0, r1 = rho[idx[i][-1]], rho[idx[i + 1][0]]
        if abs(r1 - r0) / r0 > JUMP_TOL:
            first = i + 1
    return first, idx


def classify(path):
    d = load(path)
    first, idx = production_chain(d)
    rows = np.concatenate(idx[first:])
    c = d.iloc[rows].reset_index(drop=True)
    rho = c.rho.values
    n = len(rho)
    chain_steps = int(sum(c.step.values[g][-1] - c.step.values[g][0] for g in segments(c)))
    rho_start, rho_end = rho[0], rho[-max(n // 10, 1):].mean()
    half = rho[n // 2:]
    blk = np.array([b.mean() for b in np.array_split(half, 10)])
    lr = stats.linregress(np.arange(10), blk)
    drift = lr.slope * 9 / blk.mean() * 100
    if rho_end < 0.25 * rho_start:
        label = "VAPORIZED (unconverged)"
    elif chain_steps < MIN_CHAIN_STEPS:
        label = "SHORT (too short to judge)"
    elif abs(drift) > 2 and lr.pvalue < 0.01:
        label = "UNCONVERGED (drifting)"
    elif abs(drift) > 2 and lr.pvalue < 0.1:
        label = "CONVERGED (marginal drift)"
    else:
        label = "CONVERGED"
    notes = []
    if first > 0:
        notes.append(f"{first} earlier restart segment(s) not in production chain")
    if d.rho.min() < 0.25 * d.rho.iloc[0] and label.startswith("CONVERGED"):
        notes.append(f"earlier segments expanded to rho_min={d.rho.min():.3f}")
    parts = os.path.relpath(path, HERE).split(os.sep)
    model = "anipbe0" if parts[0] == "anipbe0" else "anipbe0+mlxdm (assumed)"
    base = os.path.basename(path).split(".")
    system = base[0]
    if "run1" in parts:
        system += "_run1"
        notes.append("early short trial (run1/)")
    return d, first, dict(
        model=model, system=system, replica=int(base[1]), label=label,
        segments_in_file=len(idx), chain_from_segment=first + 1,
        chain_steps=chain_steps, rho_initial=d.rho.iloc[0], rho_chain_end=rho_end,
        rho_last_half_mean=half.mean(), last_half_drift_pct=drift, drift_p=lr.pvalue,
        T_mean_K=c["T"].mean(), rho_min_in_file=d.rho.min(), notes="; ".join(notes))


def main():
    files = sorted(f for f in glob.glob(os.path.join(HERE, "**", "*.density.csv"), recursive=True)
                   if os.path.basename(f).split(".")[0] not in EXCLUDE)
    results, data = [], {}
    for f in files:
        d, first, row = classify(f)
        results.append(row)
        data[f] = (d, first, row)
    out = pd.DataFrame(results).sort_values(["model", "system", "replica"])
    out.round(4).to_csv(os.path.join(HERE, "convergence_labels.csv"), index=False)
    print(out.round(3).drop(columns=["segments_in_file", "rho_min_in_file"]).to_string(index=False))

    # ---- figure: one panel per (system, model); left column = +mlxdm, right = anipbe0
    systems = ["ccl4", "cf4", "ch3sch3", "ch3ssch3"]
    names = {"ccl4": "CCl4", "cf4": "CF4", "ch3sch3": "CH3SCH3", "ch3ssch3": "CH3SSCH3"}
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, validated all-pairs
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, axes = plt.subplots(len(systems), 2, figsize=(13, 3.0 * len(systems)), facecolor=surface, sharey="row")
    for r, sysname in enumerate(systems):
        for c, model in enumerate(["anipbe0+mlxdm (assumed)", "anipbe0"]):
            ax = axes[r, c]
            ax.set_facecolor(surface)
            sel = [(f, v) for f, v in data.items()
                   if v[2]["system"] == sysname and v[2]["model"] == model and "run1" not in f]
            if not sel:
                ax.axis("off")
                continue
            for f, (d, first, row) in sorted(sel, key=lambda kv: kv[1][2]["replica"]):
                idx = segments(d)
                offs = np.cumsum([0] + [d.step.values[g][-1] for g in idx[:-1]])
                x = np.concatenate([offs[i] + d.step.values[g] for i, g in enumerate(idx)]) / 1e6
                k = row["replica"]
                ax.plot(x, d.rho.values, lw=0.9, color=colors[k], label=f"rep {k}: {row['label']}")
                if first > 0:
                    ax.axvline(offs[first] / 1e6, color=colors[k], lw=0.8, ls=(0, (3, 3)), alpha=0.7)
            ax.set_title(f"{names[sysname]} - {model}", loc="left", fontsize=10, color=ink)
            ax.grid(True, color="#e4e3df", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c9c8c2")
            ax.tick_params(colors=ink2, labelsize=8)
            ax.legend(fontsize=7.5, frameon=False, labelcolor=ink2, loc="best")
            if c == 0:
                ax.set_ylabel("density (g/cm$^3$)", fontsize=9, color=ink2)
            if r == len(systems) - 1 or sysname == "ch3sch3" and c == 1 or sysname == "ccl4" and False:
                ax.set_xlabel("cumulative MD steps, all restart segments (10$^6$)", fontsize=9, color=ink2)
    fig.suptitle("Density time series, labelled. Dashed line = start of the production chain "
                 "(earlier segments are restarts from the initial packing).", fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(HERE, "density_timeseries_overview.png"), dpi=140, facecolor=surface)


if __name__ == "__main__":
    main()
