# Volume Estimation Library

Estimates the **loss-basin volume** of a trained neural network at a given minimum,
measured with respect to a specific dataset's loss landscape.

The core idea: perturb the model's weights in random directions and measure how far
you can move before the loss degrades past a threshold. The distribution of these
threshold-crossing distances defines the basin's volume. Larger volume = flatter basin.

This is used to test the **volume hypothesis**: minima found by training on more data
are sharper (smaller volume) than those found on less data.

---

## Files

| File | Purpose |
|------|---------|
| `perturbations.py` | Low-level primitives: perturbing model weights, generating random directions, filter-norm scaling |
| `volume.py` | The three-stage volume estimation pipeline, config/result types, save/load |
| `volume_example.py` | Worked example showing how to orchestrate the full pipeline |

---

## The three-stage pipeline

Volume estimation is deliberately split into three stages with a save between stages 1 and 2.
This is because stage 1 is expensive and you often don't know what loss threshold to use
until you've seen the curves.

```
Stage 1   compute_loss_curves(model, x, y, config, loss_fn)
          ↓ saves to disk
          CurveResult list  (loss curve + perturbation norm per direction)

Stage 2   loss_curves_to_radii(curves, loss_threshold, config)
          ↓ in memory only — cheap, re-run freely with different thresholds
          radii list  (one float per direction that crossed the threshold)

Stage 3   log_volume_from_radii(radii, num_params)
          ↓
          float  (the log-volume proxy)
```

### Stage 1 — compute_loss_curves

For each of `config.num_directions` random directions:
1. Generate a Gaussian random direction in weight space (seeded, so reproducible).
2. Scale it by per-filter norms (`filter_norm_scaling`) so perturbations are
   isotropic across layers — each filter is perturbed relative to its own magnitude.
   This scaling is computed **once per model** and reused across all directions.
3. Walk the model incrementally along the direction through `config.coefficients`
   steps, recording loss at each step using the provided `loss_fn`.
4. Reset the model weights exactly to their original state.

Returns a `List[CurveResult]`. Each `CurveResult` holds:
- `seed` — which direction this was (so it can be regenerated if needed)
- `perturbation_norm` — the L2 norm of the raw direction before filter scaling
- `loss` — 1-D array of shape `(len(coefficients),)`

**Save this output.** It is the expensive result. Use `save_curves(curves, config, path)`.
The config is saved alongside the curves so stages 2 and 3 are self-contained.

### Stage 2 — loss_curves_to_radii

For each curve, finds the first coefficient where loss exceeds `loss_threshold`.
The basin radius in that direction is:

```
r = coefficient × perturbation_norm
```

This is the Euclidean distance in weight space from the minimum to the basin boundary
in that direction. Directions whose loss never crosses the threshold are excluded
(they indicate directions where the basin extends beyond the sweep range).

**This is cheap. Re-run it freely with different thresholds without touching the model.**

### Stage 3 — log_volume_from_radii

Computes `log(E[r^n])` where `n = num_params`, using log-sum-exp for numerical
stability. This is the volume proxy. The unit-ball constant is omitted since it is
identical for all models being compared and cancels out in any comparison.

Higher = flatter basin = larger volume.

---

## Key design decisions

**Directions are not saved.** They are fully deterministic from a seed and the model
architecture. Saving them would be enormous (one tensor per parameter per direction).
To regenerate direction `i`, call `generate_random_perturbation(model, seed=perturbation_seed+i)`.

**Filter-norm scaling is computed once per model, not once per direction.** It depends
only on the model weights, not the dataset or the direction. `compute_loss_curves`
handles this internally.

**`loss_fn` is passed explicitly.** The pipeline does not hardcode any loss function.
Pass the same loss function you trained with.

**`VolumeConfig` bundles the coefficients with the other hyperparameters.** This ensures
the coefficient array used in stage 1 is always the same one used in stages 2 and 3 —
it is saved with the curves and loaded back automatically.

---

## Minimal usage

```python
import numpy as np
import torch
from volume import VolumeConfig, compute_loss_curves, save_curves
from volume import load_curves, loss_curves_to_radii, log_volume_from_radii

# Define config once, share across all pairs in an experiment
config = VolumeConfig(
    num_directions=100,
    perturbation_seed=1,
    coefficients=np.linspace(0, 1, 100) ** 2,
)

loss_fn = torch.nn.CrossEntropyLoss()

# Stage 1 — run once, save
curves = compute_loss_curves(model, x, y, config, loss_fn)
save_curves(curves, config, "curves/my_model_on_my_dataset.npz")

# Stages 2+3 — run as many times as needed with different thresholds
curves, config = load_curves("curves/my_model_on_my_dataset.npz")
num_params = sum(p.numel() for p in model.parameters())

radii   = loss_curves_to_radii(curves, loss_threshold=0.1, config=config)
log_vol = log_volume_from_radii(radii, num_params)
print(f"log-volume: {log_vol:.3f}  ({len(radii)}/{config.num_directions} directions crossed)")
```

## Orchestrating multiple (model, dataset) pairs

See `volume_example.py` for a complete worked example. The pattern is:

```python
pairs = [("full", "full"), ("full", "half_1"), ("avg_halves", "half_1"), ...]

# Stage 1: one curves file per pair
for model_id, dataset_id in pairs:
    curves = compute_loss_curves(models[model_id], *datasets[dataset_id], config, loss_fn)
    save_curves(curves, config, f"curves/{model_id}_on_{dataset_id}.npz")

# Stages 2+3: load and threshold
for model_id, dataset_id in pairs:
    curves, config = load_curves(f"curves/{model_id}_on_{dataset_id}.npz")
    radii   = loss_curves_to_radii(curves, loss_threshold, config)
    log_vol = log_volume_from_radii(radii, num_params)
```

---

## Hyperparameter guidance

**`num_directions`** — more directions gives a better estimate of E[r^n].
100 is a reasonable starting point for exploration; 1000+ for publication-quality results.

**`coefficients`** — controls how far along each direction you sweep.
Quadratic spacing (`np.linspace(0,1,N)**2`) concentrates resolution near the origin
where the basin boundary typically sits. The maximum coefficient determines the furthest
you look — if many directions never cross the threshold, increase it.

**`loss_threshold`** — where you define the basin boundary.
Too low: almost no directions cross, radii list is empty.
Too high: every direction crosses immediately, radii are all tiny.
A good value sits above the training loss and below the loss of an untrained model.
Look at the raw loss curves first before committing to a threshold.
