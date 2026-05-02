"""
perturbations.py
----------------
Low-level primitives for random weight-space perturbations.

No experiment logic, no saving, no opinions about how perturbations
are used. Import this wherever you need to walk a model along a
direction in parameter space.

Public API
----------
ModelPerturber                  – apply / revert in-place weight changes
generate_random_perturbation    – one Gaussian direction dict
perturbation_norm               – L2 norm of a direction dict
filter_norm_scaling             – per-filter scale tensors for a model
scale_perturbation              – apply filter-norm scaling to a direction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ModelPerturber
# ---------------------------------------------------------------------------

class ModelPerturber:
    """
    Apply additive perturbations to a model's parameters in-place,
    then revert to the original state.

    Caches the parameter list on construction so repeated apply/reset
    cycles don't re-traverse the module tree.

    Usage
    -----
        perturber = ModelPerturber(model)
        perturber.apply_perturbation(direction)   # model weights shifted
        loss = evaluate(model, x, y)
        perturber.reset()                         # model weights restored
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.param_list = list(model.named_parameters())
        self.original_state = [p.detach().clone() for _, p in self.param_list]

    def apply_perturbation(self, perturbation: Dict[str, torch.Tensor]):
        """Add `perturbation` to current model weights in-place."""
        with torch.no_grad():
            for name, param in self.param_list:
                if name in perturbation:
                    param.add_(perturbation[name])

    def reset(self):
        """Restore all parameters to their state at construction time."""
        with torch.no_grad():
            for (_, param), original in zip(self.param_list, self.original_state):
                param.copy_(original)


# ---------------------------------------------------------------------------
# Direction generation
# ---------------------------------------------------------------------------

def generate_random_perturbation(
    model: nn.Module,
    perturb_list: List[str] = ["weight", "bias"],
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate one random Gaussian perturbation direction whose tensors
    match the shape of the model's named parameters.

    Only parameters whose name contains a token from `perturb_list`
    (split on '.') are included — e.g. 'model.0.weight' is included
    when 'weight' is in perturb_list.

    Args
    ----
    model        : the model whose parameter shapes are used
    perturb_list : name tokens to include (default: weights and biases)
    seed         : if given, sets torch's RNG before sampling so the
                   direction is fully reproducible

    Returns
    -------
    dict mapping parameter names → random tensors of identical shape,
    on the same device as the model parameters
    """
    if seed is not None:
        torch.manual_seed(seed)

    return {
        name: torch.randn_like(param)
        for name, param in model.named_parameters()
        if any(token in name.split(".") for token in perturb_list)
    }


def perturbation_norm(perturbation: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Global L2 norm of a perturbation dict (treats all tensors as one
    flat vector).

    Returns a scalar tensor.
    """
    return torch.norm(
        torch.cat([p.flatten() for p in perturbation.values()]), p=2
    )


# ---------------------------------------------------------------------------
# Filter-norm scaling
# ---------------------------------------------------------------------------

def filter_norm_scaling(
    model: nn.Module,
    perturb_list: List[str] = ["weight", "bias"],
) -> Dict[str, torch.Tensor]:
    """
    Compute per-filter normalization scale tensors for the model's
    current weights.

    Scaling a random direction by these tensors makes perturbations
    more isotropic across layers — each filter is perturbed in
    proportion to its own magnitude rather than in raw Gaussian units.
    See Li et al. (2018), "Visualizing the Loss Landscape of Neural Nets".

    Rules by parameter shape
    ------------------------
    2-D (Linear weight)   : scale = ‖row_i‖ for each output unit i
    4-D (Conv2D weight)   : scale = ‖filter_i‖ for each output filter i
    1-D (bias, BN params) : scale = 1 (no normalisation)
    other                 : scale = 1 (fallback)

    Args
    ----
    model        : model whose current weights define the scaling
    perturb_list : same token filter as generate_random_perturbation

    Returns
    -------
    dict mapping parameter names → scale tensors of matching shape,
    on the same device as the model
    """
    norm_dict = {}
    for name, param in model.named_parameters():
        if not any(token in name.split(".") for token in perturb_list):
            continue

        p = param.detach()
        shape = p.shape

        if len(shape) in (2, 4):
            # One scale value per output filter, broadcast to full shape
            stacked = torch.stack(
                [torch.ones_like(p[i]) * torch.norm(p[i]) for i in range(shape[0])]
            )
        elif len(shape) == 1:
            stacked = torch.ones_like(p)
        else:
            stacked = torch.ones_like(p)

        norm_dict[name] = stacked

    return norm_dict


def scale_perturbation(
    perturbation: Dict[str, torch.Tensor],
    scaling: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Element-wise multiply a perturbation dict by a scaling dict.

    Typically used to apply filter-norm scaling from `filter_norm_scaling`
    to a raw Gaussian direction from `generate_random_perturbation`.

    Both dicts must share the same keys.
    """
    return {name: perturbation[name] * scaling[name] for name in perturbation}
