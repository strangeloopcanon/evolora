from pathlib import Path

import pytest
from omegaconf.errors import OmegaConfBaseException
from yaml import YAMLError

from symbiont_ecology.config import load_ecology_config


def test_load_ecology_config_fallback_minimal(tmp_path: Path) -> None:
    # Minimal YAML that exercises the stdlib fallback parser (no omegaconf)
    yaml = (
        "host:\n  backbone_model: google/gemma-3-270m-it\n"
        "grid:\n  families: [word.count, math]\n  depths: [short]\n"
        "environment:\n  auto_batch: true\n"
        "energy:\n  Emax: 5.0\n  m: 1\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml)

    cfg = load_ecology_config(path)

    # Top-level sections parsed
    assert cfg.host.backbone_model == "google/gemma-3-270m-it"
    # Lists parsed as strings
    assert [str(x) for x in cfg.grid.families] == ["word.count", "math"]
    assert [str(x) for x in cfg.grid.depths] == ["short"]
    # Booleans and numerics parsed
    assert cfg.environment.auto_batch is True
    assert abs(cfg.energy.Emax - 5.0) < 1e-6
    assert cfg.energy.m == 1


def test_load_ecology_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    yaml = "host:\n" "  backbone_model: Qwen/Qwen3-0.6B\n" "  tokenizer Qwen/Qwen3-0.6B\n"
    path = tmp_path / "cfg2.yaml"
    path.write_text(yaml)
    with pytest.raises((ValueError, OmegaConfBaseException, YAMLError)):
        load_ecology_config(path)
