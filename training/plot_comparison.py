#!/usr/bin/env python3
"""
plot_comparison.py  --  item_017
Generate publication-quality comparison figures: PhysMDT vs AR baseline.

Outputs
-------
figures/tier_accuracy_comparison.png   Grouped bar chart of symbolic accuracy per tier
figures/r2_distributions.png           Violin plots of R^2 distributions per tier (PhysMDT)
"""

import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PHYSMDT_PATH = REPO_ROOT / "results" / "in_distribution_comparison.json"
BASELINE_PATH = REPO_ROOT / "results" / "baseline_results.json"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(PHYSMDT_PATH, "r") as f:
    physmdt_data = json.load(f)

with open(BASELINE_PATH, "r") as f:
    baseline_data = json.load(f)

# ---------------------------------------------------------------------------
# Global matplotlib styling  (publication quality, serif fonts, 300 DPI)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TIERS = ["1", "2", "3", "4", "5"]

# ===================================================================
# Figure 1 -- Grouped bar chart: Symbolic Accuracy by Tier
# ===================================================================

physmdt_acc = [
    physmdt_data["tier_results"][t]["symbolic_accuracy"] * 100
    for t in TIERS
]
baseline_acc = [
    baseline_data["tier_results"][t]["symbolic_accuracy"] * 100
    for t in TIERS
]

x = np.arange(len(TIERS))
bar_width = 0.35

fig1, ax1 = plt.subplots(figsize=(7, 4.5))

bars_physmdt = ax1.bar(
    x - bar_width / 2, physmdt_acc, bar_width,
    label="PhysMDT", color="#4878CF", edgecolor="white", linewidth=0.6,
)
bars_baseline = ax1.bar(
    x + bar_width / 2, baseline_acc, bar_width,
    label="AR Baseline", color="#EE854A", edgecolor="white", linewidth=0.6,
)

# Value labels on each bar
for bars in (bars_physmdt, bars_baseline):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5,
        )

ax1.set_xlabel("Equation Tier")
ax1.set_ylabel("Symbolic Accuracy (%)")
ax1.set_title("Symbolic Accuracy by Tier: PhysMDT vs AR Baseline")
ax1.set_xticks(x)
ax1.set_xticklabels(TIERS)
ax1.set_ylim(0, 115)  # room for labels
ax1.legend(frameon=False)

fig1.tight_layout()
out1 = FIG_DIR / "tier_accuracy_comparison.png"
fig1.savefig(out1)
plt.close(fig1)
print(f"Saved: {out1}")

# ===================================================================
# Figure 2 -- Violin / box plots: R^2 distribution per tier (PhysMDT)
# ===================================================================

# Collect per-equation mean_r2 values, grouped by tier
tier_r2: dict[str, list[float]] = {t: [] for t in TIERS}
for eq in physmdt_data["per_equation"]:
    t = str(eq["tier"])
    if t in tier_r2:
        tier_r2[t].append(eq["mean_r2"])

# Order the data for plotting
r2_data = [tier_r2[t] for t in TIERS]

fig2, ax2 = plt.subplots(figsize=(7, 4.5))

# Violin plot
parts = ax2.violinplot(
    r2_data,
    positions=np.arange(1, len(TIERS) + 1),
    showmeans=False,
    showmedians=False,
    showextrema=False,
)
for pc in parts["bodies"]:
    pc.set_facecolor("#4878CF")
    pc.set_alpha(0.30)

# Overlay box plot for quartiles / median
bp = ax2.boxplot(
    r2_data,
    positions=np.arange(1, len(TIERS) + 1),
    widths=0.25,
    patch_artist=True,
    showfliers=True,
    flierprops=dict(marker="o", markersize=4, markerfacecolor="#4878CF",
                    markeredgecolor="#4878CF", alpha=0.6),
    medianprops=dict(color="#C44E52", linewidth=1.5),
    boxprops=dict(facecolor="#4878CF", alpha=0.55, edgecolor="#333333",
                  linewidth=0.8),
    whiskerprops=dict(color="#333333", linewidth=0.8),
    capprops=dict(color="#333333", linewidth=0.8),
)

ax2.set_xlabel("Equation Tier")
ax2.set_ylabel(r"$R^2$ Score")
ax2.set_title(r"$R^2$ Score Distribution by Tier (PhysMDT)")
ax2.set_xticks(np.arange(1, len(TIERS) + 1))
ax2.set_xticklabels(TIERS)
ax2.set_ylim(-1.15, 1.15)
ax2.axhline(y=0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

fig2.tight_layout()
out2 = FIG_DIR / "r2_distributions.png"
fig2.savefig(out2)
plt.close(fig2)
print(f"Saved: {out2}")

print("Done -- all figures generated.")
