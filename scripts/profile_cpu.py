#!/usr/bin/env python3
"""
CPU Performance Profile (Item 021) -- Phase 4 Experiment
=========================================================

Measures per-component latency, peak memory, INT8 quantized vs FP32 comparison,
throughput (equations/min), and comparison against NeSymReS published CPU times.

Outputs:
  - results/cpu_performance.json
  - figures/latency_breakdown.png
"""

import signal
import sys
import os
import time
import json
import tracemalloc
import warnings
import traceback

# ---- 5-minute timeout guard ------------------------------------------------
def _timeout_handler(signum, frame):
    print("\n[TIMEOUT] Script exceeded 300 s limit -- aborting.")
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(300)

# ---- path setup -------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch

# Deterministic, single-threaded for fair CPU timing
torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model.phys_diffuser_plus import PhysDiffuserPlus, PhysDiffuserPlusConfig, fit_constants_bfgs
from src.model.encoder import SetTransformerEncoder
from src.model.decoder import AutoregressiveDecoder, VOCAB, VOCAB_SIZE
from src.model.phys_diffuser import PhysDiffuser

# ---- directories ------------------------------------------------------------
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

# ---- helpers ----------------------------------------------------------------
NUM_WARMUP = 3
NUM_RUNS   = 5
INPUT_DIM  = (9 + 1) * 16  # 160


def make_dummy_inputs():
    """Return a consistent dummy observation tensor plus raw numpy arrays."""
    obs = torch.randn(1, 100, INPUT_DIM)
    X_raw = np.random.randn(100, 3)
    y_raw = 2.0 * X_raw[:, 0] + np.sin(X_raw[:, 1])  # non-trivial target
    return obs, X_raw, y_raw


def time_fn(fn, n_warmup=NUM_WARMUP, n_runs=NUM_RUNS):
    """Time *fn* after warmup, return (mean_ms, std_ms, individual_ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        fn()
        times.append((time.time() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), times


# =============================================================================
# 1.  Per-component latency
# =============================================================================
print("=" * 60)
print("CPU Performance Profile -- Item 021")
print("=" * 60)

# Build model with reduced settings for speed
config = PhysDiffuserPlusConfig(
    diffusion_steps=10,
    num_trajectories=2,
    tta_steps=4,
)
model = PhysDiffuserPlus(config)
model.eval()

obs, X_raw, y_raw = make_dummy_inputs()

param_breakdown = model.count_parameters_breakdown()
print(f"\nParameter breakdown:")
for name, count in param_breakdown.items():
    print(f"  {name}: {count:,}")

latency = {}

# 1a. Encoding -----------------------------------------------------------------
print("\n[1] Encoding (SetTransformerEncoder forward) ...")
def _encode():
    with torch.no_grad():
        return model.encoder(obs)

enc_mean, enc_std, _ = time_fn(_encode)
latency["encoding"] = {"mean_ms": round(enc_mean, 3), "std_ms": round(enc_std, 3)}
print(f"    {enc_mean:.1f} +/- {enc_std:.1f} ms")

# Pre-compute z for downstream components
with torch.no_grad():
    z = model.encoder(obs)

# 1b. Diffusion refinement ----------------------------------------------------
print("[2] Diffusion refinement (generate_with_voting) ...")
def _diffusion():
    with torch.no_grad():
        return model.diffuser.generate_with_voting(
            z,
            num_trajectories=config.num_trajectories,
            num_steps=config.diffusion_steps,
            seq_len=32,
            temperature=config.temperature,
        )

diff_mean, diff_std, _ = time_fn(_diffusion)
latency["diffusion_refinement"] = {"mean_ms": round(diff_mean, 3), "std_ms": round(diff_std, 3)}
print(f"    {diff_mean:.1f} +/- {diff_std:.1f} ms")

# Get a prediction for downstream steps
with torch.no_grad():
    candidates = model.diffuser.generate_with_voting(
        z, num_trajectories=config.num_trajectories,
        num_steps=config.diffusion_steps, seq_len=32,
        temperature=config.temperature,
    )
    prediction = candidates[0] if candidates else ["x1"]

# 1c. AR decoding (beam search) -----------------------------------------------
print("[3] AR decoding (beam search, width=5) ...")
def _beam_search():
    with torch.no_grad():
        return model.ar_decoder.generate_beam(z, beam_width=5)

beam_mean, beam_std, _ = time_fn(_beam_search)
latency["ar_beam_search"] = {"mean_ms": round(beam_mean, 3), "std_ms": round(beam_std, 3)}
print(f"    {beam_mean:.1f} +/- {beam_std:.1f} ms")

# 1d. TTA adaptation loop -----------------------------------------------------
print("[4] TTA adaptation loop ...")
def _tta():
    # TTA needs gradients internally, but we still time the full adapt_and_predict
    def gen_fn(enc_out):
        with torch.no_grad():
            return model.diffuser.generate_refinement(enc_out, num_steps=5, seq_len=32)[0]

    adapted = model.tta.adapt_and_predict(
        model.diffuser, z.detach(), prediction, gen_fn,
    )
    return adapted

tta_mean, tta_std, _ = time_fn(_tta, n_warmup=1, n_runs=NUM_RUNS)
latency["tta_adaptation"] = {"mean_ms": round(tta_mean, 3), "std_ms": round(tta_std, 3)}
print(f"    {tta_mean:.1f} +/- {tta_std:.1f} ms")

# 1e. BFGS constant fitting ---------------------------------------------------
print("[5] BFGS constant fitting ...")
tokens_with_C = ["mul", "C", "add", "x1", "sin", "x2"]
def _bfgs():
    return fit_constants_bfgs(tokens_with_C, X_raw, y_raw, max_iter=50)

bfgs_mean, bfgs_std, _ = time_fn(_bfgs)
latency["bfgs_constant_fitting"] = {"mean_ms": round(bfgs_mean, 3), "std_ms": round(bfgs_std, 3)}
print(f"    {bfgs_mean:.1f} +/- {bfgs_std:.1f} ms")

# 1f. End-to-end inference -----------------------------------------------------
print("[6] End-to-end predict() ...")
def _e2e():
    return model.predict(obs, X_raw, y_raw)

e2e_mean, e2e_std, _ = time_fn(_e2e, n_warmup=1, n_runs=NUM_RUNS)
latency["end_to_end"] = {"mean_ms": round(e2e_mean, 3), "std_ms": round(e2e_std, 3)}
print(f"    {e2e_mean:.1f} +/- {e2e_std:.1f} ms")

# =============================================================================
# 2. Peak memory usage via tracemalloc
# =============================================================================
print("\n[7] Peak memory (tracemalloc) ...")
tracemalloc.start()
_ = model.predict(obs, X_raw, y_raw)
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
memory_info = {
    "current_MB": round(current_mem / 1e6, 2),
    "peak_MB": round(peak_mem / 1e6, 2),
}
print(f"    Current: {memory_info['current_MB']:.2f} MB, Peak: {memory_info['peak_MB']:.2f} MB")

# =============================================================================
# 3. INT8 quantized model vs FP32
# =============================================================================
print("\n[8] INT8 dynamic quantization vs FP32 ...")
quantization_results = {}

try:
    import torch.ao.quantization as tq

    # Quantize dynamically (Linear + LSTM layers)
    model_int8 = tq.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8,
    )
    model_int8.eval()

    # FP32 end-to-end timing (reuse e2e_mean from above)
    fp32_mean = e2e_mean

    # INT8 end-to-end timing
    def _e2e_int8():
        return model_int8.predict(obs, X_raw, y_raw)

    int8_mean, int8_std, _ = time_fn(_e2e_int8, n_warmup=1, n_runs=NUM_RUNS)

    speedup = fp32_mean / int8_mean if int8_mean > 0 else float("nan")

    # Accuracy delta: compare predictions (both use random weights, so we
    # compare output token-list length and structure consistency)
    fp32_result = model.predict(obs, X_raw, y_raw)
    int8_result = model_int8.predict(obs, X_raw, y_raw)

    fp32_tokens = fp32_result["prediction"]
    int8_tokens = int8_result["prediction"]
    tokens_match = fp32_tokens == int8_tokens

    quantization_results = {
        "status": "success",
        "fp32_mean_ms": round(fp32_mean, 3),
        "int8_mean_ms": round(int8_mean, 3),
        "int8_std_ms": round(int8_std, 3),
        "speedup_factor": round(speedup, 3),
        "tokens_match": tokens_match,
        "fp32_token_count": len(fp32_tokens),
        "int8_token_count": len(int8_tokens),
        "note": (
            "Both models use random (untrained) weights, so token match is "
            "not a true accuracy measure. Speedup factor is the primary metric."
        ),
    }
    print(f"    FP32: {fp32_mean:.1f} ms, INT8: {int8_mean:.1f} ms")
    print(f"    Speedup: {speedup:.2f}x, Tokens match: {tokens_match}")

except Exception as exc:
    quantization_results = {
        "status": "failed",
        "reason": str(exc),
        "traceback": traceback.format_exc(),
    }
    print(f"    Quantization failed: {exc}")

# =============================================================================
# 4. Throughput: equations solved per minute
# =============================================================================
print("\n[9] Throughput ...")
equations_per_min = 60_000.0 / e2e_mean if e2e_mean > 0 else 0.0
throughput_info = {
    "equations_per_minute": round(equations_per_min, 2),
    "ms_per_equation": round(e2e_mean, 3),
}
print(f"    {equations_per_min:.1f} equations/min  ({e2e_mean:.0f} ms/eq)")

# =============================================================================
# 5. Comparison against NeSymReS published CPU times
# =============================================================================
print("\n[10] Comparison vs NeSymReS ...")
nesymres_low_ms  = 2000.0   # ~2 s per equation (published lower bound)
nesymres_high_ms = 5000.0   # ~5 s per equation (published upper bound)
nesymres_comparison = {
    "nesymres_range_ms": [nesymres_low_ms, nesymres_high_ms],
    "phys_diffuser_plus_ms": round(e2e_mean, 3),
    "faster_than_nesymres_low": e2e_mean < nesymres_low_ms,
    "faster_than_nesymres_high": e2e_mean < nesymres_high_ms,
    "relative_to_nesymres_mid": round(e2e_mean / ((nesymres_low_ms + nesymres_high_ms) / 2), 3),
    "note": (
        "NeSymReS published CPU inference: ~2-5 s per equation (Biggio et al., 2021). "
        "PhysDiffuser+ config: diffusion_steps=10, num_trajectories=2, tta_steps=4."
    ),
}
mid = (nesymres_low_ms + nesymres_high_ms) / 2
print(f"    PhysDiffuser+: {e2e_mean:.0f} ms  vs  NeSymReS: {nesymres_low_ms:.0f}-{nesymres_high_ms:.0f} ms")
print(f"    Ratio to NeSymReS midpoint: {e2e_mean / mid:.2f}x")

# =============================================================================
# 6. Assemble and save JSON
# =============================================================================
results = {
    "experiment": "021_cpu_performance_profile",
    "config": {
        "diffusion_steps": config.diffusion_steps,
        "num_trajectories": config.num_trajectories,
        "tta_steps": config.tta_steps,
        "embed_dim": config.embed_dim,
        "num_warmup": NUM_WARMUP,
        "num_runs": NUM_RUNS,
        "torch_threads": 1,
    },
    "parameter_breakdown": param_breakdown,
    "latency": latency,
    "memory": memory_info,
    "quantization_int8_vs_fp32": quantization_results,
    "throughput": throughput_info,
    "nesymres_comparison": nesymres_comparison,
}

json_path = os.path.join(ROOT, "results", "cpu_performance.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved -> {json_path}")

# =============================================================================
# 7. Figure: stacked bar chart of latency breakdown
# =============================================================================
print("\n[11] Generating latency breakdown figure ...")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    warnings.warn("Style 'seaborn-v0_8-whitegrid' not found; using default.")

component_names = [
    "Encoding",
    "Diffusion\nRefinement",
    "AR Beam\nSearch",
    "TTA\nAdaptation",
    "BFGS\nFitting",
]
component_keys = [
    "encoding",
    "diffusion_refinement",
    "ar_beam_search",
    "tta_adaptation",
    "bfgs_constant_fitting",
]
means = [latency[k]["mean_ms"] for k in component_keys]
stds  = [latency[k]["std_ms"]  for k in component_keys]

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

fig, ax = plt.subplots(figsize=(10, 6))

# Stacked horizontal bar
left = 0.0
bars = []
for i, (name, mean, std, color) in enumerate(zip(component_names, means, stds, colors)):
    bar = ax.barh(
        "PhysDiffuser+\n(CPU, 1 thread)",
        mean,
        left=left,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        label=f"{name}  ({mean:.0f} ms)",
    )
    # Annotate inside bar if wide enough
    if mean > 0.05 * sum(means):
        ax.text(
            left + mean / 2,
            0,
            f"{mean:.0f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
        )
    left += mean
    bars.append(bar)

# End-to-end line
ax.axvline(x=e2e_mean, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(
    e2e_mean + 10, 0.25,
    f"End-to-end: {e2e_mean:.0f} ms",
    fontsize=9, va="bottom",
)

# NeSymReS reference band
ax.axvspan(nesymres_low_ms, nesymres_high_ms, alpha=0.12, color="gray", label="NeSymReS range (2-5 s)")

ax.set_xlabel("Latency (ms)", fontsize=11)
ax.set_title("PhysDiffuser+ CPU Latency Breakdown (per equation)", fontsize=13, fontweight="bold")
ax.legend(
    loc="upper right", fontsize=8, framealpha=0.9,
    title="Component (mean latency)", title_fontsize=9,
)
ax.set_xlim(0, max(e2e_mean * 1.15, nesymres_high_ms * 1.05))

plt.tight_layout()
fig_path = os.path.join(ROOT, "figures", "latency_breakdown.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved -> {fig_path}")

# =============================================================================
# Done
# =============================================================================
signal.alarm(0)  # cancel timeout
print("\n" + "=" * 60)
print("CPU Performance Profile complete.")
print("=" * 60)
