"""Utilities for merging base models and LoRA adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class MergeComponent:
    name: str
    weight: torch.Tensor
    alpha: float = 1.0


class ModelMerger:
    """Compose base weights and LoRA deltas to grow/shrink organisms."""

    def __init__(self, components: Iterable[MergeComponent]) -> None:
        self.components: list[MergeComponent] = list(components)

    def merge(self) -> torch.Tensor:
        if not self.components:
            raise ValueError("No components to merge")
        base = self.components[0].weight.clone()
        for component in self.components[1:]:
            delta = component.weight
            if delta.shape != base.shape:
                expanded = torch.zeros_like(base)
                rows = min(base.shape[0], delta.shape[0])
                cols = min(base.shape[1], delta.shape[1])
                expanded[:rows, :cols] = delta[:rows, :cols]
                delta = expanded
            base += component.alpha * delta
        return base

    @staticmethod
    def chain(weights: Iterable[torch.Tensor]) -> torch.Tensor:
        result = None
        for weight in weights:
            result = weight if result is None else result @ weight
        if result is None:
            raise ValueError("No weights to chain")
        return result


__all__ = ["MergeComponent", "ModelMerger"]
