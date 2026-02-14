#!/usr/bin/env python3
"""
Create the wow_showcase.png multi-panel figure for PhysDiffuser+ results.

Panel A: Exact match rates by difficulty tier (bar chart)
Panel B: SOTA comparison (grouped bar chart)
Panel C: Noise robustness curves (with and without TTA)
Panel D: Ablation component contributions (horizontal bar chart)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Load data ----
with open(os.path.join(ROOT, "results", "feynman_benchmark.json")) as f:
    benchmark = json.load(f)

with open(os.path.join(ROOT, "results", "ablation_study.json")) as f:
    ablation = json.load(f)

with open(os.path.join(ROOT, "results", "noise_robustness.json")) as f:
    noise = json.load(f)

# ---- Style ----
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("PhysDiffuser+: Physics Equation Derivation Results", fontsize=16, fontweight="bold", y=0.98)

# ===========================================================================
# Panel A: Exact match by tier
# ===========================================================================
ax = axes[0, 0]
tiers = ["trivial", "simple", "moderate", "complex", "multi_step"]
tier_labels = ["Trivial\n(n=20)", "Simple\n(n=25)", "Moderate\n(n=30)", "Complex\n(n=25)", "Multi-step\n(n=20)"]
tier_rates = [benchmark["per_tier"][t]["exact_match_rate"] * 100 for t in tiers]

colors_tier = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
bars = ax.bar(tier_labels, tier_rates, color=colors_tier, edgecolor="white", linewidth=1.2, width=0.65)
for bar, val in zip(bars, tier_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Exact Match Rate (%)", fontsize=12)
ax.set_title("A. Exact Match by Difficulty Tier", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.axhline(y=benchmark["overall"]["exact_match_rate"] * 100, color="#3498db", linestyle="--",
           linewidth=1.5, label=f'Overall: {benchmark["overall"]["exact_match_rate"]*100:.1f}%')
ax.legend(loc="upper right", fontsize=10)

# ===========================================================================
# Panel B: SOTA comparison
# ===========================================================================
ax = axes[0, 1]
sota = benchmark["sota_comparison"]
# Sort by exact match rate descending
methods_sorted = sorted(sota.keys(), key=lambda m: sota[m]["exact_match_rate"], reverse=True)
method_labels = []
method_rates = []
bar_colors_sota = []
for m in methods_sorted:
    method_labels.append(f'{m}\n({sota[m]["year"]})')
    method_rates.append(sota[m]["exact_match_rate"] * 100)
    if m == "PhysDiffuser+":
        bar_colors_sota.append("#e74c3c")
    else:
        bar_colors_sota.append("#95a5a6")

bars_sota = ax.barh(method_labels[::-1], method_rates[::-1], color=bar_colors_sota[::-1],
                    edgecolor="white", linewidth=1.2, height=0.55)
for bar, val in zip(bars_sota, method_rates[::-1]):
    ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", ha="left", va="center", fontsize=10, fontweight="bold")

ax.set_xlabel("Exact Match Rate (%)", fontsize=12)
ax.set_title("B. Comparison with Prior Work", fontsize=13, fontweight="bold")
ax.set_xlim(0, 115)

# ===========================================================================
# Panel C: Noise robustness curves
# ===========================================================================
ax = axes[1, 0]
sigma_levels = noise["sigma_levels"]

# without TTA
wo_rates = []
for s in sigma_levels:
    key = str(s) if not str(s).endswith(".0") else str(s)
    # Handle both "0.0" and "0" formats
    for k in [str(s), str(float(s))]:
        if k in noise["without_tta"]:
            wo_rates.append(noise["without_tta"][k]["exact_match_rate"] * 100)
            break

# with TTA
w_rates = []
for s in sigma_levels:
    for k in [str(s), str(float(s))]:
        if k in noise["with_tta"]:
            w_rates.append(noise["with_tta"][k]["exact_match_rate"] * 100)
            break

ax.plot(sigma_levels, wo_rates, "o-", color="#e74c3c", linewidth=2.2, markersize=8, label="Without TTA")
ax.plot(sigma_levels, w_rates, "s-", color="#3498db", linewidth=2.2, markersize=8, label="With TTA")
ax.fill_between(sigma_levels, wo_rates, w_rates, alpha=0.15, color="#3498db")
ax.set_xlabel("Noise Level (sigma)", fontsize=12)
ax.set_ylabel("Exact Match Rate (%)", fontsize=12)
ax.set_title("C. Noise Robustness", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(0, max(max(wo_rates), max(w_rates)) + 10)
ax.set_xticks(sigma_levels)

# ===========================================================================
# Panel D: Ablation -- component contributions
# ===========================================================================
ax = axes[1, 1]

full_rate = ablation["variants"]["full"]["overall"]["exact_match_rate"]["point"] * 100
ablation_items = {
    "No Diffusion": ablation["variants"]["no-diffusion"]["overall"]["exact_match_rate"]["point"] * 100,
    "No Physics Priors": ablation["variants"]["no-physics-priors"]["overall"]["exact_match_rate"]["point"] * 100,
    "No TTA": ablation["variants"]["no-TTA"]["overall"]["exact_match_rate"]["point"] * 100,
    "No Derivation Chains": ablation["variants"]["no-derivation-chains"]["overall"]["exact_match_rate"]["point"] * 100,
    "Baseline AR\n(all removed)": ablation["variants"]["baseline-AR"]["overall"]["exact_match_rate"]["point"] * 100,
}

component_names = list(ablation_items.keys())
drops = [full_rate - v for v in ablation_items.values()]
drop_colors = plt.cm.Reds(np.linspace(0.3, 0.85, len(drops)))

bars_abl = ax.barh(component_names[::-1], [d for d in drops[::-1]], color=drop_colors[::-1],
                   edgecolor="white", linewidth=1.2, height=0.55)
for bar, val in zip(bars_abl, drops[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"-{val:.1f}pp", ha="left", va="center", fontsize=10, fontweight="bold")

ax.set_xlabel("Drop in Exact Match (percentage points)", fontsize=12)
ax.set_title("D. Ablation: Component Contributions", fontsize=13, fontweight="bold")
ax.set_xlim(0, max(drops) + 12)
ax.axvline(x=full_rate, color="#2ecc71", linestyle="--", linewidth=1.3, alpha=0.7)

# ---- Save ----
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(ROOT, "figures", "wow_showcase.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
