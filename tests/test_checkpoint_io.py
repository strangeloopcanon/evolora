from __future__ import annotations

import pickle
from pathlib import Path

import torch

from symbiont_ecology.utils.checkpoint_io import (
    CheckpointLoadError,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_io_round_trip_safe_format(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    state = {
        "generation": 7,
        "adapter_states": {"org1": {"layer.lora_A": torch.ones(2, 3)}},
        "compute_budget": {"train_tokens": 123},
    }
    save_checkpoint(path, state)

    loaded = load_checkpoint(path)
    assert int(loaded.get("generation", 0)) == 7
    adapter_states = loaded.get("adapter_states")
    assert isinstance(adapter_states, dict)
    assert "org1" in adapter_states


def test_checkpoint_io_rejects_legacy_pickle_by_default(tmp_path: Path) -> None:
    path = tmp_path / "legacy_checkpoint.pt"
    path.write_bytes(pickle.dumps({"generation": 3}))

    try:
        load_checkpoint(path)
        raise AssertionError("Expected CheckpointLoadError for legacy pickle checkpoint")
    except CheckpointLoadError:
        pass


def test_checkpoint_io_allows_legacy_pickle_with_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "legacy_checkpoint.pt"
    path.write_bytes(pickle.dumps({"generation": 9}))

    loaded = load_checkpoint(path, allow_unsafe_pickle=True)
    assert int(loaded.get("generation", 0)) == 9
