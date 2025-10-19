"""Utility helpers wrapping torch for optional availability."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def resolve_device(device: str | int | torch.device | None = None) -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


@contextmanager
def no_grad() -> Iterator[None]:
    with torch.no_grad():
        yield


def clamp_norm(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(tensor)
    if norm > max_norm:
        tensor = tensor * (max_norm / (norm + 1e-9))
    return tensor


def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


__all__ = ["clamp_norm", "ensure_dtype", "no_grad", "resolve_device"]
