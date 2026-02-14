"""
Interactive demo for PhysDiffuser+ equation derivation.

Given observation data (X, y), run the full inference pipeline and output
the derived equation in both prefix notation and human-readable infix form,
together with an R-squared goodness-of-fit score.

Usage
-----
    # Inline data (comma-separated x values and y values)
    python scripts/demo.py \\
        --x "0.1,0.5,1.0,1.5,2.0,2.5,3.0" \\
        --y "0.0998,0.4794,0.8415,0.9975,0.9093,0.5985,0.1411"

    # From a CSV file (first columns are x variables, last column is y)
    python scripts/demo.py --file observations.csv

    # Use baseline checkpoint instead
    python scripts/demo.py --checkpoint models/baseline_checkpoint.pt --baseline \\
        --x "1,2,3,4" --y "1,4,9,16"

See ``python scripts/demo.py --help`` for all options.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.generator import prefix_to_infix, BINARY_OPS, UNARY_OPS
from src.model.phys_diffuser_plus import PhysDiffuserPlus, PhysDiffuserPlusConfig
from src.model.encoder import SetTransformerEncoder
from src.model.decoder import (
    AutoregressiveDecoder,
    VOCAB,
    VOCAB_SIZE,
    ID_TO_TOKEN,
)
from src.eval.metrics import r_squared, prefix_to_sympy


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def parse_inline_data(x_str, y_str):
    """Parse comma-separated strings into numpy arrays.

    ``x_str`` may encode a 1-D variable or multiple variables separated by
    semicolons (one variable per group).  For example:

        --x "1,2,3;4,5,6" --y "5,7,9"

    yields X of shape (3, 2).
    """
    groups = x_str.split(";")
    cols = []
    for g in groups:
        vals = [float(v.strip()) for v in g.split(",") if v.strip()]
        cols.append(vals)
    X = np.array(cols, dtype=np.float64).T  # [N, num_vars]
    y = np.array([float(v.strip()) for v in y_str.split(",") if v.strip()], dtype=np.float64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of x points ({X.shape[0]}) does not match "
            f"number of y points ({y.shape[0]})"
        )
    return X, y


def load_csv_data(path):
    """Load observations from a CSV file.

    Assumes the last column is y and all preceding columns are x variables.
    """
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            # Skip header rows that are not numeric
            try:
                float(row[0])
            except (ValueError, IndexError):
                continue
            rows.append([float(v) for v in row])
    data = np.array(rows, dtype=np.float64)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def prefix_to_human(tokens):
    """Convert prefix tokens to a more human-readable infix string."""
    return prefix_to_infix(tokens)


def prefix_to_sympy_str(tokens):
    """Try to produce a simplified SymPy string; fall back to infix."""
    try:
        expr = prefix_to_sympy(tokens)
        if expr is not None:
            import sympy
            return str(sympy.simplify(expr))
    except Exception:
        pass
    return prefix_to_human(tokens)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model, X, y):
    """Run the PhysDiffuser+ inference pipeline on raw data.

    Returns (prediction_tokens, timings_dict).
    """
    num_vars = X.shape[1]
    encoded = model.encoder.encode_observations(X, y, num_vars)
    obs = encoded.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        result = model.predict(obs, X, y)

    return result["prediction"], result.get("timings", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "PhysDiffuser+ interactive demo. Derives a symbolic equation "
            "from numerical observations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = p.add_argument_group("Observation data")
    data_group.add_argument(
        "--x",
        type=str,
        default=None,
        help=(
            "Comma-separated x values. For multiple variables, separate "
            "groups with semicolons, e.g. '1,2,3;4,5,6'."
        ),
    )
    data_group.add_argument(
        "--y",
        type=str,
        default=None,
        help="Comma-separated y values.",
    )
    data_group.add_argument(
        "--file",
        type=str,
        default=None,
        help="CSV file with observations (last column = y).",
    )

    model_group = p.add_argument_group("Model")
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_REPO_ROOT, "models", "physdiffuser_plus_checkpoint.pt"),
        help="Path to model checkpoint.",
    )
    model_group.add_argument(
        "--baseline",
        action="store_true",
        help="Load baseline (encoder+decoder) checkpoint instead of PhysDiffuser+.",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--num_threads", type=int, default=4, help="PyTorch CPU threads.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.num_threads)

    # ---- Load data ----
    if args.file is not None:
        X, y = load_csv_data(args.file)
    elif args.x is not None and args.y is not None:
        X, y = parse_inline_data(args.x, args.y)
    else:
        print("ERROR: provide observation data via --x/--y or --file.")
        print("Run with --help for usage details.")
        sys.exit(1)

    num_vars = X.shape[1]
    n_points = X.shape[0]
    print("=" * 60)
    print("PhysDiffuser+ Demo")
    print("=" * 60)
    print(f"Observations: {n_points} points, {num_vars} variable(s)")
    print(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")

    # ---- Load model ----
    if not os.path.isfile(args.checkpoint):
        print(f"\nERROR: checkpoint not found at {args.checkpoint}")
        print("Train a model first:  python scripts/train.py")
        sys.exit(1)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.baseline:
        # Baseline checkpoint has encoder_state / decoder_state
        encoder = SetTransformerEncoder(embed_dim=256, num_heads=8, num_layers=2, num_inducing=16)
        decoder = AutoregressiveDecoder(embed_dim=256, num_heads=8, num_layers=4, ff_dim=512)
        if "encoder_state" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state"], strict=False)
        if "decoder_state" in ckpt:
            decoder.load_state_dict(ckpt["decoder_state"], strict=False)
        encoder.eval()
        decoder.eval()
        print("Loaded baseline encoder + decoder")

        # Encode and generate
        encoded = encoder.encode_observations(X, y, num_vars)
        obs = encoded.unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            z = encoder(obs)
            pred_tokens = decoder.generate_beam(z, beam_width=5, max_length=64)[0]
        elapsed_ms = (time.time() - t0) * 1000
    else:
        # PhysDiffuser+ checkpoint
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            config = PhysDiffuserPlusConfig(**{
                k: v for k, v in ckpt["config"].items()
                if k in PhysDiffuserPlusConfig.__dataclass_fields__
            })
        else:
            config = PhysDiffuserPlusConfig()
        model = PhysDiffuserPlus(config)
        state_key = "model_state" if "model_state" in ckpt else "state_dict"
        if state_key in ckpt:
            model.load_state_dict(ckpt[state_key], strict=False)
        print(f"Loaded PhysDiffuser+ ({model.count_parameters():,} params)")

        t0 = time.time()
        pred_tokens, timings = run_inference(model, X, y)
        elapsed_ms = (time.time() - t0) * 1000

    # ---- Compute R-squared ----
    r2 = r_squared(pred_tokens, X, y, num_vars)

    # ---- Display results ----
    print(f"\n{'=' * 60}")
    print("Derived Equation")
    print(f"{'=' * 60}")
    print(f"  Prefix notation : {' '.join(pred_tokens)}")
    print(f"  Infix (readable): {prefix_to_human(pred_tokens)}")
    sympy_str = prefix_to_sympy_str(pred_tokens)
    if sympy_str != prefix_to_human(pred_tokens):
        print(f"  Simplified      : {sympy_str}")
    print(f"\n  R-squared       : {r2:.6f}")
    print(f"  Inference time  : {elapsed_ms:.0f} ms")
    print(f"  Token count     : {len(pred_tokens)}")

    if r2 > 0.99:
        print("\n  Fit quality: EXCELLENT (R^2 > 0.99)")
    elif r2 > 0.9:
        print("\n  Fit quality: GOOD (R^2 > 0.9)")
    elif r2 > 0.5:
        print("\n  Fit quality: MODERATE (R^2 > 0.5)")
    else:
        print("\n  Fit quality: POOR (R^2 <= 0.5)")

    print()


if __name__ == "__main__":
    main()
