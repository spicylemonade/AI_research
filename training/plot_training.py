#!/usr/bin/env python3
"""
Plot PhysMDT curriculum training dynamics.

Reads training step data from CSV and validation/phase data from JSON,
then produces a publication-quality multi-panel figure.

Output: figures/physmdt_training_loss.png and .pdf
"""

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "physmdt_training_steps.csv"
JSON_PATH = REPO_ROOT / "results" / "physmdt_training_log.json"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style configuration  (Nature / Science look)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "mathtext.fontset": "cm",
})

# Phase colour palette
PHASE_COLORS = {
    "Phase 1 (Tier 1-2)": "#1f77b4",  # blue
    "Phase 2 (Tier 1-3)": "#ff7f0e",  # orange
    "Phase 3 (Tier 1-4)": "#2ca02c",  # green
}
PHASE_SHORT = {
    "Phase 1 (Tier 1-2)": "Phase 1",
    "Phase 2 (Tier 1-3)": "Phase 2",
    "Phase 3 (Tier 1-4)": "Phase 3",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# --- CSV (training steps) ---
steps, phases, losses, mask_ratios, lrs = [], [], [], [], []
with open(CSV_PATH, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        steps.append(int(row["global_step"]))
        phases.append(row["phase"])
        losses.append(float(row["loss"]))
        mask_ratios.append(float(row["mask_ratio"]))
        lrs.append(float(row["lr"]))

# --- JSON (validation epochs + phase results) ---
with open(JSON_PATH) as fh:
    log = json.load(fh)

val_epochs = log["val_epochs"]
val_steps = [v["global_step"] for v in val_epochs]
val_losses = [v["val_loss"] for v in val_epochs]
val_train_losses = [v["train_loss"] for v in val_epochs]
val_phases = [v["phase"] for v in val_epochs]
val_epoch_nums = [v["epoch"] for v in val_epochs]

# ---------------------------------------------------------------------------
# Determine phase transition boundaries
# ---------------------------------------------------------------------------
phase_transitions = []  # list of (step, new_phase_label)
prev_phase = phases[0]
for s, p in zip(steps, phases):
    if p != prev_phase:
        phase_transitions.append((s, p))
        prev_phase = p

# ---------------------------------------------------------------------------
# Build figure: 3 rows  (training loss | mask ratio | validation loss)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    3, 1,
    figsize=(5.5, 7.0),
    gridspec_kw={"height_ratios": [3, 1.2, 2], "hspace": 0.30},
)
ax_loss, ax_mask, ax_val = axes

# ===== Panel (a): Training loss (log scale) by phase ======================
for phase_name, color in PHASE_COLORS.items():
    xs = [s for s, p in zip(steps, phases) if p == phase_name]
    ys = [l for l, p in zip(losses, phases) if p == phase_name]
    if xs:
        ax_loss.plot(xs, ys, color=color, alpha=0.85, linewidth=0.9,
                     label=PHASE_SHORT[phase_name])

ax_loss.set_yscale("log")
ax_loss.set_ylabel("Training Loss")
ax_loss.set_xlabel("Global Step")
ax_loss.set_xlim(0, max(steps) * 1.02)

# Phase transition vertical lines
for trans_step, trans_phase in phase_transitions:
    ax_loss.axvline(trans_step, color="grey", linestyle="--", linewidth=0.7,
                    alpha=0.7)
    # Label just above the top of the axes
    ax_loss.text(trans_step, ax_loss.get_ylim()[1] * 0.8,
                 f"  {PHASE_SHORT[trans_phase]}",
                 fontsize=7, color="grey", va="top", ha="left",
                 fontstyle="italic")

ax_loss.legend(loc="upper right", frameon=True, edgecolor="0.8",
               fancybox=False)
ax_loss.grid(True, which="major", linewidth=0.3, alpha=0.5)
ax_loss.grid(True, which="minor", linewidth=0.2, alpha=0.3)
ax_loss.set_title("(a)  Training Loss per Step", loc="left", fontweight="bold",
                  fontsize=9)

# ===== Panel (b): Mask ratio evolution =====================================
for phase_name, color in PHASE_COLORS.items():
    xs = [s for s, p in zip(steps, phases) if p == phase_name]
    ys = [m for m, p in zip(mask_ratios, phases) if p == phase_name]
    if xs:
        ax_mask.plot(xs, ys, color=color, alpha=0.75, linewidth=0.8)

for trans_step, _ in phase_transitions:
    ax_mask.axvline(trans_step, color="grey", linestyle="--", linewidth=0.7,
                    alpha=0.7)

ax_mask.set_ylabel("Mask Ratio")
ax_mask.set_xlabel("Global Step")
ax_mask.set_xlim(0, max(steps) * 1.02)
ax_mask.set_ylim(0, 1.05)
ax_mask.grid(True, which="major", linewidth=0.3, alpha=0.5)
ax_mask.set_title("(b)  Mask Ratio Evolution", loc="left", fontweight="bold",
                  fontsize=9)

# ===== Panel (c): Validation loss per epoch ================================
# We plot per-phase, using sequential epoch index for the x-axis
epoch_index = list(range(1, len(val_epochs) + 1))

for phase_name, color in PHASE_COLORS.items():
    idxs = [i for i, p in enumerate(val_phases) if p == phase_name]
    if idxs:
        xs = [epoch_index[i] for i in idxs]
        ys_val = [val_losses[i] for i in idxs]
        ys_train = [val_train_losses[i] for i in idxs]
        ax_val.plot(xs, ys_val, "o-", color=color, linewidth=1.0,
                    markersize=4, label=f"{PHASE_SHORT[phase_name]} val")
        ax_val.plot(xs, ys_train, "s--", color=color, linewidth=0.8,
                    markersize=3, alpha=0.6,
                    label=f"{PHASE_SHORT[phase_name]} train")

# Phase boundary vertical lines in epoch space
prev_vp = val_phases[0]
for i, vp in enumerate(val_phases):
    if vp != prev_vp:
        ax_val.axvline(epoch_index[i] - 0.5, color="grey", linestyle="--",
                       linewidth=0.7, alpha=0.7)
        prev_vp = vp

ax_val.set_ylabel("Loss")
ax_val.set_xlabel("Epoch (sequential across phases)")
ax_val.set_yscale("log")
ax_val.grid(True, which="major", linewidth=0.3, alpha=0.5)
ax_val.grid(True, which="minor", linewidth=0.2, alpha=0.3)

# Build a compact 2-column legend
ax_val.legend(loc="upper right", frameon=True, edgecolor="0.8",
              fancybox=False, ncol=2, columnspacing=1.0)
ax_val.set_title("(c)  Validation & Train Loss per Epoch", loc="left",
                 fontweight="bold", fontsize=9)

# ---------------------------------------------------------------------------
# Suptitle
# ---------------------------------------------------------------------------
fig.suptitle("PhysMDT Curriculum Training Dynamics",
             fontsize=12, fontweight="bold", y=0.97)

# ---------------------------------------------------------------------------
# Tight layout & save
# ---------------------------------------------------------------------------
fig.subplots_adjust(top=0.93, hspace=0.35, left=0.12, right=0.95, bottom=0.06)

out_png = FIG_DIR / "physmdt_training_loss.png"
out_pdf = FIG_DIR / "physmdt_training_loss.pdf"
fig.savefig(out_png, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
