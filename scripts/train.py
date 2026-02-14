"""
Train PhysDiffuser+ on synthetic symbolic regression data.

Full training pipeline:
  1. Generate random equations via EquationGenerator
  2. Encode observations through SetTransformerEncoder
  3. Train the joint diffusion + AR model with physics priors
  4. Periodically evaluate on the Feynman benchmark
  5. Save model checkpoint

Optimised for CPU-only execution.  A SIGALRM timeout guard prevents
run-away training when running under a wall-clock budget.

Usage
-----
    python scripts/train.py --num_steps 5000 --batch_size 2 --lr 1e-4

See ``python scripts/train.py --help`` for all options.
"""

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Ensure repo root is on the path so that ``src`` is importable regardless of
# the working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.generator import EquationGenerator
from src.model.decoder import VOCAB, VOCAB_SIZE, tokens_to_ids, ID_TO_TOKEN
from src.model.phys_diffuser_plus import (
    PhysDiffuserPlus,
    PhysDiffuserPlusConfig,
    create_training_step,
)
from src.eval.metrics import r_squared, symbolic_equivalence, tree_edit_distance


# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------
class _TrainTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TrainTimeout()


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------
def prepare_batch(equations, encoder, max_seq_len=64):
    """Build tensors from a list of generated equation dicts.

    Returns
    -------
    enc_input : Tensor [B, N, input_dim]
    dec_tokens : Tensor [B, T]   (BOS + tokens, padded)
    dec_target : Tensor [B, T]   (tokens + EOS, padded)
    pad_mask   : Tensor [B, T]   True where PAD
    """
    batch_enc, batch_dec_in, batch_dec_tgt = [], [], []

    for eq in equations:
        X = np.array(eq["support_points_x"])
        y = np.array(eq["support_points_y"])
        num_vars = X.shape[1]
        encoded = encoder.encode_observations(X, y, num_vars)
        batch_enc.append(encoded)

        token_ids = tokens_to_ids(eq["prefix_tokens"])
        dec_in = [VOCAB["BOS"]] + token_ids
        dec_tgt = token_ids + [VOCAB["EOS"]]

        if len(dec_in) > max_seq_len:
            dec_in = dec_in[:max_seq_len]
            dec_tgt = dec_tgt[:max_seq_len]
        else:
            pad_len = max_seq_len - len(dec_in)
            dec_in = dec_in + [VOCAB["PAD"]] * pad_len
            dec_tgt = dec_tgt + [VOCAB["PAD"]] * pad_len

        batch_dec_in.append(dec_in)
        batch_dec_tgt.append(dec_tgt)

    enc_input = torch.stack(batch_enc)
    dec_in = torch.tensor(batch_dec_in, dtype=torch.long)
    dec_tgt = torch.tensor(batch_dec_tgt, dtype=torch.long)
    pad_mask = dec_in == VOCAB["PAD"]

    return enc_input, dec_in, dec_tgt, pad_mask


# ---------------------------------------------------------------------------
# Quick Feynman evaluation (subset)
# ---------------------------------------------------------------------------
def quick_feynman_eval(model, benchmark_path, n_equations=20, seed=42):
    """Evaluate on the first *n_equations* of the Feynman benchmark.

    Returns a dict with aggregate metrics.
    """
    if not os.path.exists(benchmark_path):
        return {"error": "benchmark file not found"}

    with open(benchmark_path) as fh:
        benchmark = json.load(fh)

    equations = benchmark["equations"][:n_equations]
    model.eval()

    n_exact, n_r2_good = 0, 0
    r2_scores = []

    for i, eq in enumerate(equations):
        try:
            formula_python = eq["formula_python"]
            variables = eq["variables"]
            num_vars = eq["num_variables"]

            rng = np.random.RandomState(seed + i)
            X = rng.uniform(0.1, 5.0, size=(200, num_vars))

            safe_globals = {
                "__builtins__": {},
                "pi": np.pi, "e": np.e,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "abs": np.abs,
            }
            for vi, var in enumerate(variables):
                safe_globals[var["name"]] = X[:, vi]

            y = eval(formula_python, safe_globals)  # noqa: S307

            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                r2_scores.append(-1.0)
                continue

            encoded = model.encoder.encode_observations(X, y, num_vars)
            obs = encoded.unsqueeze(0)

            with torch.no_grad():
                result = model.predict(obs, X, y)

            pred = result["prediction"]
            r2 = r_squared(pred, X, y, num_vars)
            r2_scores.append(r2)

            if r2 > 0.9:
                n_r2_good += 1

        except Exception:
            r2_scores.append(-1.0)

    return {
        "n_evaluated": len(r2_scores),
        "mean_r2": float(np.mean(r2_scores)) if r2_scores else 0.0,
        "r2_above_0.9": n_r2_good,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train PhysDiffuser+ on synthetic symbolic regression data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training hyper-parameters
    p.add_argument("--num_steps", type=int, default=5000,
                    help="Total training steps.")
    p.add_argument("--batch_size", type=int, default=2,
                    help="Batch size (number of equations per step).")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for Adam optimiser.")
    p.add_argument("--max_seq_len", type=int, default=64,
                    help="Maximum decoder sequence length.")
    p.add_argument("--num_support_points", type=int, default=30,
                    help="Number of observation support points per equation.")
    p.add_argument("--max_depth", type=int, default=4,
                    help="Maximum depth for random expression trees.")
    p.add_argument("--max_variables", type=int, default=3,
                    help="Maximum number of variables in generated equations.")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")

    # Model configuration
    p.add_argument("--embed_dim", type=int, default=256,
                    help="Embedding dimension.")
    p.add_argument("--num_heads", type=int, default=8,
                    help="Number of attention heads.")
    p.add_argument("--diffuser_layers", type=int, default=4,
                    help="Number of diffusion transformer layers.")
    p.add_argument("--diffuser_ff_dim", type=int, default=512,
                    help="Feed-forward dimension in diffusion layers.")
    p.add_argument("--diffusion_steps", type=int, default=50,
                    help="Number of diffusion refinement steps during eval.")
    p.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout rate.")

    # Ablation flags
    p.add_argument("--no_diffusion", action="store_true",
                    help="Disable masked diffusion (use AR decoder only).")
    p.add_argument("--no_physics_priors", action="store_true",
                    help="Disable physics-informed priors.")
    p.add_argument("--no_tta", action="store_true",
                    help="Disable test-time adaptation.")
    p.add_argument("--no_derivation_chains", action="store_true",
                    help="Disable derivation-chain compositionality loss.")
    p.add_argument("--no_constant_fitting", action="store_true",
                    help="Disable BFGS constant fitting at inference.")

    # Logging / checkpointing
    p.add_argument("--log_interval", type=int, default=100,
                    help="Print loss every N steps.")
    p.add_argument("--eval_interval", type=int, default=2500,
                    help="Run quick Feynman evaluation every N steps.")
    p.add_argument("--checkpoint_dir", type=str,
                    default=os.path.join(_REPO_ROOT, "models"),
                    help="Directory to save model checkpoints.")
    p.add_argument("--results_dir", type=str,
                    default=os.path.join(_REPO_ROOT, "results"),
                    help="Directory to save training results.")
    p.add_argument("--benchmark_path", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "feynman_equations.json"),
                    help="Path to the Feynman benchmark JSON file.")

    # Timeout
    p.add_argument("--timeout", type=int, default=600,
                    help="Wall-clock timeout in seconds (0 to disable).")

    # CPU threads
    p.add_argument("--num_threads", type=int, default=4,
                    help="Number of PyTorch CPU threads.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.num_threads)

    # ---- Timeout ----
    if args.timeout > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.timeout)

    print("=" * 60)
    print("PhysDiffuser+ Training")
    print("=" * 60)

    # ---- Model ----
    config = PhysDiffuserPlusConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        diffuser_layers=args.diffuser_layers,
        diffuser_ff_dim=args.diffuser_ff_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        use_diffusion=not args.no_diffusion,
        use_physics_priors=not args.no_physics_priors,
        use_tta=not args.no_tta,
        use_derivation_chains=not args.no_derivation_chains,
        use_constant_fitting=not args.no_constant_fitting,
    )
    model = PhysDiffuserPlus(config)

    breakdown = model.count_parameters_breakdown()
    print("\nParameter breakdown:")
    for name, count in breakdown.items():
        print(f"  {name}: {count:,}")

    # ---- Optimiser ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_step = create_training_step(model, optimizer)

    # ---- Data generator ----
    gen = EquationGenerator(
        seed=args.seed,
        max_depth=args.max_depth,
        max_variables=args.max_variables,
        num_support_points=args.num_support_points,
    )

    # ---- Training loop ----
    print(f"\nTraining for {args.num_steps} steps, batch_size={args.batch_size}, "
          f"support_points={args.num_support_points}, max_seq_len={args.max_seq_len}")

    losses_history = []
    start_time = time.time()
    num_steps = args.num_steps

    try:
        for step in range(1, num_steps + 1):
            equations = gen.generate_batch(args.batch_size)
            if len(equations) < args.batch_size:
                continue
            equations = equations[: args.batch_size]

            enc_input, dec_in, dec_tgt, pad_mask = prepare_batch(
                equations, model.encoder, args.max_seq_len
            )

            # PhysDiffuser+ forward expects full target tokens (BOS prefix removed
            # internally for AR loss).  We pass dec_in which starts with BOS.
            losses = train_step(enc_input, dec_in, pad_mask)
            losses_history.append(losses["total"])

            if step % args.log_interval == 0:
                avg_loss = float(np.mean(losses_history[-args.log_interval :]))
                elapsed = time.time() - start_time
                sps = step / elapsed
                remaining = (num_steps - step) / sps if sps > 0 else 0
                print(
                    f"  Step {step}/{num_steps} | Loss: {avg_loss:.4f} | "
                    f"{sps:.1f} steps/s | {elapsed:.0f}s elapsed | "
                    f"~{remaining:.0f}s remaining"
                )

                # Dynamically reduce steps if on track to exceed budget
                if step == args.log_interval and args.timeout > 0:
                    est_total = num_steps / sps
                    budget_for_training = args.timeout * 0.8
                    if est_total > budget_for_training:
                        num_steps = max(int(sps * budget_for_training), step + 100)
                        print(f"  -> Adjusting num_steps to {num_steps} for time budget")

            # ---- Periodic evaluation ----
            if step % args.eval_interval == 0:
                print(f"\n  [Eval @ step {step}]")
                eval_metrics = quick_feynman_eval(model, args.benchmark_path)
                for k, v in eval_metrics.items():
                    print(f"    {k}: {v}")
                model.train()
                print()

        if args.timeout > 0:
            signal.alarm(0)

    except _TrainTimeout:
        print("\nTraining timed out -- saving current state.")
        signal.alarm(0)

    total_train_time = time.time() - start_time
    final_loss = (
        float(np.mean(losses_history[-100:]))
        if len(losses_history) >= 100
        else float(np.mean(losses_history)) if losses_history else float("nan")
    )
    print(f"\nTraining completed in {total_train_time:.0f}s ({total_train_time / 60:.1f} min)")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Total steps completed: {len(losses_history)}")

    # ---- Save checkpoint ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "physdiffuser_plus_checkpoint.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(config),
            "args": vars(args),
            "step": len(losses_history),
            "final_loss": final_loss,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")

    # ---- Save training results ----
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "training_results.json")
    results = {
        "model": "PhysDiffuser+",
        "total_params": model.count_parameters(),
        "training": {
            "num_steps": len(losses_history),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "final_loss": final_loss,
            "training_time_seconds": total_train_time,
        },
        "loss_history": losses_history,
    }
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # ---- Save loss figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        window = min(50, max(1, len(losses_history) // 10))
        if window > 1 and len(losses_history) >= window:
            smoothed = np.convolve(losses_history, np.ones(window) / window, mode="valid")
        else:
            smoothed = np.array(losses_history)
        ax.plot(losses_history, alpha=0.2, linewidth=0.5, label="raw")
        ax.plot(
            range(window - 1, window - 1 + len(smoothed)),
            smoothed,
            linewidth=2,
            label=f"smoothed (w={window})",
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("PhysDiffuser+ Training Convergence")
        ax.legend()
        fig_dir = os.path.join(_REPO_ROOT, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, "physdiffuser_plus_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss figure saved to {os.path.join(fig_dir, 'physdiffuser_plus_loss.png')}")
    except Exception as exc:
        print(f"Warning: could not save loss figure: {exc}")


if __name__ == "__main__":
    main()
