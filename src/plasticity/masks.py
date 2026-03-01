"""Derive and manage binary pruning masks from importance tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


class MaskSet:
    """A collection of binary masks keyed by ``module_name``.

    Each mask is a bool tensor with the same shape as the corresponding
    weight matrix.  ``True`` means *keep*, ``False`` means *prune*.
    """

    def __init__(self, masks: Dict[str, torch.Tensor]) -> None:
        self.masks = masks

    @property
    def module_names(self) -> List[str]:
        return list(self.masks.keys())

    def sparsity(self) -> float:
        """Return the overall fraction of weights that are pruned (masked out)."""
        total = sum(m.numel() for m in self.masks.values())
        pruned = sum((~m).sum().item() for m in self.masks.values())
        return pruned / total if total > 0 else 0.0

    def per_module_sparsity(self) -> Dict[str, float]:
        """Return sparsity fraction for each module."""
        return {
            name: float((~mask).sum().item()) / mask.numel() if mask.numel() > 0 else 0.0
            for name, mask in self.masks.items()
        }

    def save(self, path: str | Path) -> None:
        """Save masks to a directory (one .pt file per module + metadata)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {"modules": list(self.masks.keys()), "sparsity": self.sparsity()}
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        for name, mask in self.masks.items():
            safe_name = name.replace(".", "_")
            torch.save(mask, out / f"{safe_name}.pt")

    @classmethod
    def load(cls, path: str | Path) -> "MaskSet":
        """Load masks from a directory previously saved with :meth:`save`."""
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        masks: Dict[str, torch.Tensor] = {}
        for name in meta["modules"]:
            safe_name = name.replace(".", "_")
            masks[name] = torch.load(p / f"{safe_name}.pt", weights_only=True, map_location="cpu")
        return cls(masks)


def derive_masks(
    importance: Dict[str, torch.Tensor],
    sparsity: float,
) -> MaskSet:
    """Derive a binary mask at the target sparsity using per-layer unstructured pruning.

    For each layer independently, the bottom ``sparsity`` fraction of weights
    (by importance score) are masked out.
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

    masks: Dict[str, torch.Tensor] = {}
    for name, imp in importance.items():
        n_weights = imp.numel()
        n_prune = int(n_weights * sparsity)
        if n_prune == 0:
            masks[name] = torch.ones_like(imp, dtype=torch.bool)
            continue
        if n_prune >= n_weights:
            masks[name] = torch.zeros_like(imp, dtype=torch.bool)
            continue
        flat = imp.flatten()
        threshold = torch.kthvalue(flat, n_prune).values
        masks[name] = imp > threshold
    return MaskSet(masks)


def derive_random_masks(
    reference_importance: Dict[str, torch.Tensor],
    sparsity: float,
    seed: int = 999,
) -> MaskSet:
    """Derive a random binary mask at the target sparsity (baseline).

    Uses the shapes from *reference_importance* but ignores the values.
    """
    gen = torch.Generator().manual_seed(seed)
    masks: Dict[str, torch.Tensor] = {}
    for name, imp in reference_importance.items():
        rand_scores = torch.rand(imp.shape, generator=gen)
        n_weights = imp.numel()
        n_prune = int(n_weights * sparsity)
        if n_prune == 0:
            masks[name] = torch.ones(imp.shape, dtype=torch.bool)
        elif n_prune >= n_weights:
            masks[name] = torch.zeros(imp.shape, dtype=torch.bool)
        else:
            threshold = torch.kthvalue(rand_scores.flatten(), n_prune).values
            masks[name] = rand_scores > threshold
    return MaskSet(masks)


def derive_global_masks(
    per_family_importance: Dict[str, Dict[str, torch.Tensor]],
    sparsity: float,
) -> MaskSet:
    """Derive a single global mask by averaging importance across families."""
    module_names = list(next(iter(per_family_importance.values())).keys())
    averaged: Dict[str, torch.Tensor] = {}
    n_families = len(per_family_importance)
    for mod_name in module_names:
        summed: Optional[torch.Tensor] = None
        for family_imp in per_family_importance.values():
            imp = family_imp[mod_name]
            summed = imp.clone() if summed is None else summed + imp
        averaged[mod_name] = summed / n_families  # type: ignore[operator]
    return derive_masks(averaged, sparsity)


def save_importance(
    importance: Dict[str, torch.Tensor],
    path: str | Path,
    family: str,
) -> None:
    """Save per-module importance tensors for one family."""
    out = Path(path) / family
    out.mkdir(parents=True, exist_ok=True)
    meta = {"family": family, "modules": list(importance.keys())}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    for name, tensor in importance.items():
        safe_name = name.replace(".", "_")
        torch.save(tensor, out / f"{safe_name}.pt")


def load_importance(path: str | Path, family: str) -> Dict[str, torch.Tensor]:
    """Load per-module importance tensors for one family."""
    p = Path(path) / family
    with open(p / "meta.json") as f:
        meta = json.load(f)
    result: Dict[str, torch.Tensor] = {}
    for name in meta["modules"]:
        safe_name = name.replace(".", "_")
        result[name] = torch.load(p / f"{safe_name}.pt", weights_only=True, map_location="cpu")
    return result
