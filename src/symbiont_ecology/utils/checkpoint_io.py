"""Secure checkpoint load/save helpers.

Default behavior only accepts tensor-safe checkpoints that can be loaded with
``torch.load(..., weights_only=True)``. Legacy pickle checkpoints remain
available behind an explicit opt-in for backwards compatibility.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torch

UNSAFE_PICKLE_ENV = "EVOLORA_ALLOW_UNSAFE_PICKLE"


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded safely."""


def _env_allows_unsafe_pickle() -> bool:
    raw = os.getenv(UNSAFE_PICKLE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_checkpoint(path: Path | str, *, allow_unsafe_pickle: bool = False) -> dict[str, Any]:
    """Load checkpoint state with safe defaults.

    Args:
        path: Checkpoint path.
        allow_unsafe_pickle: When true, allows fallback to legacy ``pickle``.
            This is unsafe and should only be used with trusted artifacts.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if allow_unsafe_pickle or _env_allows_unsafe_pickle():
            data = pickle.loads(
                checkpoint_path.read_bytes()
            )  # nosec B301 - explicit trusted-only opt-in
            if not isinstance(data, dict):
                raise CheckpointLoadError(
                    f"Legacy checkpoint at {checkpoint_path} did not deserialize to dict"
                ) from exc
            return data
        raise CheckpointLoadError(
            f"Checkpoint {checkpoint_path} is not in safe tensor format. "
            f"To load trusted legacy pickle artifacts, set {UNSAFE_PICKLE_ENV}=1 "
            "or pass --allow-unsafe-pickle."
        ) from exc

    if not isinstance(state, dict):
        raise CheckpointLoadError(f"Checkpoint {checkpoint_path} is not a dict payload")
    return state


def save_checkpoint(path: Path | str, state: dict[str, Any]) -> None:
    """Atomically save checkpoint state in torch tensor format."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)


__all__ = ["CheckpointLoadError", "UNSAFE_PICKLE_ENV", "load_checkpoint", "save_checkpoint"]
