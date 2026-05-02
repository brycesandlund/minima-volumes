"""
volume.py
---------
Loss-basin volume estimation via random weight-space perturbations.

Three-stage pipeline
--------------------
1. compute_loss_curves(model, x, y, config, loss_fn, batch_size)
       -> List[CurveResult]

   The expensive stage. Walks the model along `config.num_directions`
   random filter-normalised directions, recording loss at each step.
   Results carry enough metadata (seed, perturbation_norm, coefficients)
   to be self-describing when saved.

2. loss_curves_to_radii(curves, loss_threshold)
       -> List[float]

   Cheap. For each curve, finds the first coefficient where loss
   exceeds `loss_threshold` and returns r = coeff × perturbation_norm.
   Directions that never cross are excluded.
   Run this repeatedly with different thresholds — no recomputation needed.

3. log_volume_from_radii(radii, num_params)
       -> float

   Computes log(E[r^n]) where n = num_params via log-sum-exp.
   This is the volume proxy; the unit-ball constant is omitted since
   it is identical for all models being compared.

Save / load
-----------
save_curves(curves, config, path)   – save stage-1 output to .npz
load_curves(path)                   – reload, returns (curves, config)

Public API
----------
VolumeConfig        dataclass: num_directions, perturbation_seed, coefficients
CurveResult         dataclass: one direction's loss curve + metadata
compute_loss_curves
loss_curves_to_radii
log_volume_from_radii
save_curves
load_curves
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from perturbations import (
    ModelPerturber,
    generate_random_perturbation,
    perturbation_norm,
    filter_norm_scaling,
    scale_perturbation,
)


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class VolumeConfig:
    """
    Hyperparameters that define one volume estimation run.

    Saving this alongside the loss curves ensures that stage 2 and 3
    always use the same coefficients that were used in stage 1.

    Attributes
    ----------
    num_directions    : how many random directions to sample
    perturbation_seed : base seed; direction i uses seed+i, so runs are
                        fully reproducible
    coefficients      : 1-D array of step sizes along each direction,
                        e.g. np.linspace(0, 1, 100) ** 2
                        (quadratic spacing gives finer resolution near
                        the origin where the basin boundary typically is)
    """
    num_directions:    int
    perturbation_seed: int
    coefficients:      np.ndarray

    def to_dict(self) -> dict:
        return {
            "num_directions":    self.num_directions,
            "perturbation_seed": self.perturbation_seed,
            "coefficients":      self.coefficients,
        }

    @staticmethod
    def from_dict(d: dict) -> "VolumeConfig":
        return VolumeConfig(
            num_directions=int(d["num_directions"]),
            perturbation_seed=int(d["perturbation_seed"]),
            coefficients=np.asarray(d["coefficients"]),
        )


@dataclass
class CurveResult:
    """
    Loss curve for one perturbation direction.

    Attributes
    ----------
    seed              : the RNG seed used to generate this direction
                        (perturbation_seed + direction_index)
    perturbation_norm : L2 norm of the raw (pre-filter-scaling) direction
                        stored so r = coeff × norm can be computed in
                        stage 2 without the model
    loss              : 1-D array, shape (len(coefficients),)
                        loss at each coefficient step
    """
    seed:              int
    perturbation_norm: float
    loss:              np.ndarray


# ---------------------------------------------------------------------------
# Stage 1 – compute loss curves
# ---------------------------------------------------------------------------

def compute_loss_curves(
    model:      nn.Module,
    x:          torch.Tensor,
    y:          torch.Tensor,
    config:     VolumeConfig,
    loss_fn:    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: Optional[int] = None,
) -> List[CurveResult]:
    """
    Walk `model` along `config.num_directions` random filter-normalised
    directions in weight space, recording loss at each coefficient step.

    Filter-norm scaling is computed once from the model's current weights
    and reused for all directions — it does not depend on the dataset.

    Each direction is generated deterministically from its seed, so results
    are reproducible and individual directions can be regenerated if needed.

    Args
    ----
    model      : trained nn.Module — weights are perturbed then restored,
                 the model is not modified permanently
    x, y       : dataset to evaluate loss on (moved to model's device
                 automatically)
    config     : VolumeConfig with num_directions, perturbation_seed,
                 coefficients
    loss_fn    : callable(logits, labels) -> scalar tensor
    batch_size : if set, evaluates in mini-batches; if None, full dataset
                 in one forward pass

    Returns
    -------
    List of CurveResult, one per direction, in seed order.
    """
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    coefficients = config.coefficients

    # Filter-norm scaling depends only on model weights — compute once
    scaling = filter_norm_scaling(model)
    perturber = ModelPerturber(model)

    results = []
    for i in range(config.num_directions):
        seed = config.perturbation_seed + i

        raw_direction = generate_random_perturbation(model, seed=seed)
        p_norm = perturbation_norm(raw_direction).item()
        scaled_direction = scale_perturbation(raw_direction, scaling)

        losses = _walk_direction(
            model, perturber, x, y,
            scaled_direction, coefficients, loss_fn, batch_size,
        )
        results.append(CurveResult(seed=seed, perturbation_norm=p_norm, loss=losses))

    return results


def _walk_direction(
    model:            nn.Module,
    perturber:        ModelPerturber,
    x:                torch.Tensor,
    y:                torch.Tensor,
    scaled_direction: dict,
    coefficients:     np.ndarray,
    loss_fn:          Callable,
    batch_size:       Optional[int],
) -> np.ndarray:
    """
    Evaluate loss at each coefficient step along one direction.

    Uses incremental perturbation (apply delta at each step rather than
    resetting and re-applying from zero) to avoid floating-point drift
    accumulating across many small steps.

    Internal helper — not part of the public API.
    """
    loss_values = torch.zeros(len(coefficients), device=x.device)
    previous_coeff = 0.0

    with torch.no_grad():
        for i, coeff in enumerate(coefficients):
            delta = float(coeff) - previous_coeff
            perturber.apply_perturbation(
                {k: v * delta for k, v in scaled_direction.items()}
            )
            previous_coeff = float(coeff)

            if batch_size is None:
                loss_values[i] = loss_fn(model(x), y)
            else:
                total, n = 0.0, x.shape[0]
                for s in range(0, n, batch_size):
                    xb, yb = x[s:s + batch_size], y[s:s + batch_size]
                    total += loss_fn(model(xb), yb).item() * len(xb)
                loss_values[i] = total / n

        perturber.reset()

    return loss_values.cpu().numpy()


# ---------------------------------------------------------------------------
# Stage 2 – loss curves → radii
# ---------------------------------------------------------------------------

def loss_curves_to_radii(
    curves:         List[CurveResult],
    loss_threshold: float,
    config:         VolumeConfig,
) -> List[float]:
    """
    For each curve, find the first coefficient where loss exceeds
    `loss_threshold` and compute the basin radius in that direction:

        r = coefficient × perturbation_norm

    This is the Euclidean distance in weight space from the minimum to
    the point where the loss landscape "escapes" the basin.

    Directions whose loss never exceeds the threshold are excluded —
    they indicate directions where the basin extends beyond the range
    of the coefficient sweep.

    Args
    ----
    curves         : output of compute_loss_curves
    loss_threshold : scalar; typical values 0.05 – 0.5 depending on scale
    config         : the VolumeConfig used to generate the curves
                     (coefficients are read from here)

    Returns
    -------
    List of positive floats, one per direction that crossed the threshold.
    May be shorter than len(curves) if some directions never crossed.
    """
    radii = []
    coefficients = config.coefficients
    for curve in curves:
        crossings = np.where(np.asarray(curve.loss) > loss_threshold)[0]
        if len(crossings) > 0:
            coeff = float(coefficients[crossings[0]])
            radii.append(coeff * curve.perturbation_norm)
    return radii


# ---------------------------------------------------------------------------
# Stage 3 – radii → log-volume
# ---------------------------------------------------------------------------

def log_volume_from_radii(
    radii:      List[float],
    num_params: int,
) -> float:
    """
    Compute the log-volume proxy  log( E[r^n] )  where n = num_params.

    In an n-dimensional ball the volume scales as r^n. Taking the
    expectation over random directions and working in log-space (via
    log-sum-exp) keeps the computation numerically stable even for very
    large n.

    The unit-ball constant  log(π^(n/2) / Γ(n/2+1))  is omitted — it
    is identical for all models being compared and cancels out.

    Args
    ----
    radii      : output of loss_curves_to_radii
    num_params : number of model parameters (the exponent n);
                 use  sum(p.numel() for p in model.parameters())

    Returns
    -------
    float — log-volume proxy, higher means a wider/flatter basin.
    Returns -inf if radii is empty or all zeros.
    """
    r = np.array([v for v in radii if v > 0], dtype=np.float64)
    if len(r) == 0:
        return float("-inf")

    logs = num_params * np.log(r)
    # log-sum-exp for numerical stability, then convert to log-mean
    # denominator is len(radii) (total directions), not len(r),
    # so directions that never cross contribute zero to E[r^n]
    log_vol = (
        logs.max()
        + np.log(np.sum(np.exp(logs - logs.max())))
        - np.log(len(radii))
    )
    return float(log_vol)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_curves(
    curves: List[CurveResult],
    config: VolumeConfig,
    path:   str | Path,
) -> None:
    """
    Save stage-1 loss curves and the config used to generate them.

    Stores everything needed for stages 2 and 3 without the model.

    File schema (npz)
    -----------------
    seeds              : int array, shape (num_directions,)
    perturbation_norms : float array, shape (num_directions,)
    loss_curves        : float array, shape (num_directions, len(coefficients))
    config             : object array wrapping VolumeConfig.to_dict()

    Args
    ----
    curves : output of compute_loss_curves
    config : the VolumeConfig used to generate the curves
    path   : file path (should end in .npz, added automatically if absent)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seeds  = np.array([c.seed              for c in curves], dtype=np.int64)
    norms  = np.array([c.perturbation_norm for c in curves], dtype=np.float64)
    losses = np.stack([c.loss              for c in curves])   # (N, len(coeff))

    np.savez_compressed(
        path,
        seeds=seeds,
        perturbation_norms=norms,
        loss_curves=losses,
        config=np.array(config.to_dict(), dtype=object),
    )


def load_curves(path: str | Path) -> tuple[List[CurveResult], VolumeConfig]:
    """
    Load stage-1 results saved by save_curves.

    Returns
    -------
    (curves, config)
        curves : List[CurveResult] — reconstructed from the npz arrays
        config : VolumeConfig      — as saved, including coefficients
    """
    path = Path(path)
    npz = np.load(path, allow_pickle=True)

    config = VolumeConfig.from_dict(npz["config"].item())

    curves = [
        CurveResult(
            seed=int(npz["seeds"][i]),
            perturbation_norm=float(npz["perturbation_norms"][i]),
            loss=npz["loss_curves"][i],
        )
        for i in range(len(npz["seeds"]))
    ]

    return curves, config
