"""
Efficiency analysis: compare PARR and baseline models on parameter count,
inference latency, GPU memory, and throughput at various refinement steps.
Saves results to results/efficiency_results.json.
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.parr_transformer import create_parr_model
from src.models.baseline_transformer import create_baseline_model
from src.data.equation_templates import MAX_OBS_POINTS, MAX_INPUT_VARS


SEED = 42
BATCH_SIZE = 32
N_WARMUP = 10
N_BATCHES = 100


def count_parameters(model):
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dummy_batch(batch_size, device):
    """Create a random observation tensor matching the dataset schema."""
    # shape: (batch_size, MAX_OBS_POINTS, MAX_INPUT_VARS + 1)
    return torch.randn(batch_size, MAX_OBS_POINTS, MAX_INPUT_VARS + 1, device=device)


def measure_inference(model, generate_fn, device, batch_size=BATCH_SIZE,
                      n_warmup=N_WARMUP, n_batches=N_BATCHES):
    """Measure latency, throughput, and peak GPU memory for a generate call.

    Args:
        model: the model (already on device, in eval mode).
        generate_fn: callable(observations) -> tokens.
        device: torch device string.
        batch_size: equations per batch.
        n_warmup: warm-up iterations (not timed).
        n_batches: timed iterations.

    Returns:
        dict with ms_per_equation, equations_per_second, peak_gpu_memory_mb.
    """
    model.eval()
    obs = make_dummy_batch(batch_size, device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = generate_fn(obs)

    # Reset memory stats if on GPU
    use_gpu = device != "cpu" and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Timed loop
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = generate_fn(obs)
            if use_gpu:
                torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_equations = batch_size * n_batches
    ms_per_eq = (elapsed / total_equations) * 1000.0
    eq_per_sec = total_equations / elapsed

    peak_mem_mb = 0.0
    if use_gpu:
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "ms_per_equation": round(ms_per_eq, 3),
        "equations_per_second": round(eq_per_sec, 2),
        "peak_gpu_memory_mb": round(peak_mem_mb, 2),
        "total_time_s": round(elapsed, 3),
        "n_batches": n_batches,
        "batch_size": batch_size,
    }


def run_efficiency(
    parr_checkpoint="checkpoints/parr_best.pt",
    baseline_checkpoint="checkpoints/baseline_best.pt",
    results_dir="results",
    d_model=512,
    K_trained=8,
    device="cuda",
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(results_dir, exist_ok=True)

    results = {}

    # ------------------------------------------------------------------
    # Baseline model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("BASELINE MODEL")
    print("=" * 70)
    if not os.path.exists(baseline_checkpoint):
        print(f"Checkpoint not found at {baseline_checkpoint}. "
              "Creating model with random weights for demonstration.")
        baseline = create_baseline_model(device=device)
    else:
        baseline = create_baseline_model(device=device)
        baseline.load_state_dict(
            torch.load(baseline_checkpoint, map_location=device)
        )
    baseline.eval()

    baseline_params = count_parameters(baseline)
    print(f"  Parameters: {baseline_params:,} ({baseline_params/1e6:.1f}M)")

    print("  Measuring inference (AR generation)...")
    baseline_stats = measure_inference(
        baseline,
        generate_fn=lambda obs: baseline.generate(obs),
        device=device,
    )
    baseline_stats["parameters"] = baseline_params
    baseline_stats["parameters_M"] = round(baseline_params / 1e6, 1)
    results["baseline"] = baseline_stats

    print(f"  Latency   : {baseline_stats['ms_per_equation']:.3f} ms/eq")
    print(f"  Throughput: {baseline_stats['equations_per_second']:.1f} eq/s")
    print(f"  Peak GPU  : {baseline_stats['peak_gpu_memory_mb']:.1f} MB")

    # Free baseline to save memory
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PARR model at different K values
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PARR MODEL")
    print("=" * 70)
    if not os.path.exists(parr_checkpoint):
        print(f"Checkpoint not found at {parr_checkpoint}. "
              "Creating model with random weights for demonstration.")
        parr = create_parr_model(d_model=d_model, K=K_trained, device=device)
    else:
        parr = create_parr_model(d_model=d_model, K=K_trained, device=device)
        parr.load_state_dict(
            torch.load(parr_checkpoint, map_location=device)
        )
    parr.eval()

    parr_params = count_parameters(parr)
    print(f"  Parameters: {parr_params:,} ({parr_params/1e6:.1f}M)")

    k_values = [0, 2, 4, 8]
    results["parr"] = {"parameters": parr_params, "parameters_M": round(parr_params / 1e6, 1)}
    results["parr"]["by_K"] = {}

    for k in k_values:
        print(f"\n  --- PARR K={k} ---")
        stats = measure_inference(
            parr,
            generate_fn=lambda obs, _k=k: parr.generate(obs, K=_k),
            device=device,
        )
        results["parr"]["by_K"][str(k)] = stats

        print(f"    Latency   : {stats['ms_per_equation']:.3f} ms/eq")
        print(f"    Throughput: {stats['equations_per_second']:.1f} eq/s")
        print(f"    Peak GPU  : {stats['peak_gpu_memory_mb']:.1f} MB")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EFFICIENCY SUMMARY")
    print("=" * 70)
    header = f"{'Model':<20} {'Params (M)':>12} {'ms/eq':>10} {'eq/s':>10} {'GPU MB':>10}"
    print(header)
    print("-" * len(header))

    bl = results["baseline"]
    print(f"{'Baseline':<20} {bl['parameters_M']:>12.1f} {bl['ms_per_equation']:>10.3f} "
          f"{bl['equations_per_second']:>10.1f} {bl['peak_gpu_memory_mb']:>10.1f}")

    for k in k_values:
        s = results["parr"]["by_K"][str(k)]
        label = f"PARR (K={k})"
        pm = results["parr"]["parameters_M"]
        print(f"{label:<20} {pm:>12.1f} {s['ms_per_equation']:>10.3f} "
              f"{s['equations_per_second']:>10.1f} {s['peak_gpu_memory_mb']:>10.1f}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(results_dir, "efficiency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_efficiency(device=device)
