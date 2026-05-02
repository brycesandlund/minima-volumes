"""
volume_example.py
-----------------
Sample script showing how to orchestrate volume estimation using
perturbations.py and volume.py.

Assumes you already have:
  - a list of trained models
  - a list of datasets (each a (x, y) tensor pair)
  - a loss function
  - knowledge of which (model, dataset) pairs you want volumes for

The three stages are kept separate so the expensive stage 1 is run
once per pair and saved; stages 2 and 3 can be re-run interactively
with different thresholds at no cost.
"""

import numpy as np
import torch
from pathlib import Path

from volume import VolumeConfig, compute_loss_curves, save_curves, load_curves
from volume import loss_curves_to_radii, log_volume_from_radii


# ---------------------------------------------------------------------------
# 0. Setup — define your models, datasets, and pairs
# ---------------------------------------------------------------------------

# Each entry is just a label and the thing it labels.
# Labels are used to build the save path for each pair's curves file.

models = {
    "full":       ...,   # your trained nn.Module
    "half_1":     ...,
    "half_2":     ...,
    "avg_halves": ...,
}

datasets = {
    "full":   (..., ...),   # (x_tensor, y_tensor)
    "half_1": (..., ...),
    "half_2": (..., ...),
}

loss_fn = torch.nn.CrossEntropyLoss()   # or whatever you trained with

# Which (model, dataset) pairs do you want volumes for?
pairs = [
    ("full",       "full"),
    ("full",       "half_1"),
    ("full",       "half_2"),
    ("half_1",     "half_1"),
    ("half_2",     "half_2"),
    ("avg_halves", "full"),
    ("avg_halves", "half_1"),
    ("avg_halves", "half_2"),
]

# Where to save curve files
curves_dir = Path("curves")

# ---------------------------------------------------------------------------
# 1. Config — shared across all pairs in this experiment
# ---------------------------------------------------------------------------

config = VolumeConfig(
    num_directions=100,          # increase for more accurate estimates
    perturbation_seed=1,
    coefficients=np.linspace(0, 1, 100) ** 2,   # quadratic: finer near origin
)

# ---------------------------------------------------------------------------
# Stage 1 — compute and save loss curves
#
# Run this once. If a curves file already exists for a pair, skip it so
# you can safely re-run the script after interruption.
# ---------------------------------------------------------------------------

print("=== Stage 1: computing loss curves ===")
for model_id, dataset_id in pairs:
    save_path = curves_dir / f"{model_id}_on_{dataset_id}.npz"

    if save_path.exists():
        print(f"  {model_id} on {dataset_id}: already exists, skipping")
        continue

    print(f"  {model_id} on {dataset_id} ...", end=" ", flush=True)
    model = models[model_id]
    x, y  = datasets[dataset_id]

    curves = compute_loss_curves(model, x, y, config, loss_fn)
    save_curves(curves, config, save_path)
    print(f"saved ({len(curves)} directions)")

# ---------------------------------------------------------------------------
# Stage 2 + 3 — load curves, threshold, compute volumes
#
# Run this as many times as you like with different thresholds.
# No model or dataset needed beyond this point.
# ---------------------------------------------------------------------------

loss_threshold = 0.1   # adjust freely; re-run stages 2+3 at no cost

print(f"\n=== Stages 2+3: computing volumes (threshold={loss_threshold}) ===")

volumes = {}
for model_id, dataset_id in pairs:
    save_path = curves_dir / f"{model_id}_on_{dataset_id}.npz"

    curves, loaded_config = load_curves(save_path)

    # num_params comes from the model, but since directions are
    # deterministic from the seed you can also just store it once
    num_params = sum(p.numel() for p in models[model_id].parameters())

    radii   = loss_curves_to_radii(curves, loss_threshold, loaded_config)
    log_vol = log_volume_from_radii(radii, num_params)

    volumes[(model_id, dataset_id)] = log_vol
    crossed = len(radii)
    total   = loaded_config.num_directions
    print(f"  {model_id:12s} on {dataset_id:7s}: "
          f"log_vol = {log_vol:.3f}  ({crossed}/{total} directions crossed)")

# ---------------------------------------------------------------------------
# Inspect — rank models by volume on each dataset
# ---------------------------------------------------------------------------

print("\n=== Volume rankings (higher = flatter basin) ===")

all_datasets = sorted({ds for _, ds in pairs})
for dataset_id in all_datasets:
    relevant = [(m, volumes[(m, d)]) for m, d in pairs if d == dataset_id]
    relevant.sort(key=lambda x: x[1], reverse=True)
    ranked = "  >  ".join(f"{m} ({v:.2f})" for m, v in relevant)
    print(f"  {dataset_id:7s}: {ranked}")
