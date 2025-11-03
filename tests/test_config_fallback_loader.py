import builtins
from pathlib import Path
from types import SimpleNamespace

from symbiont_ecology.config import load_ecology_config


def test_load_ecology_config_fallback_parser(tmp_path: Path, monkeypatch):
    # Force ImportError for omegaconf to trigger fallback parser
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "omegaconf":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cfg_text = (
        "grid:\n"
        "  families: [word.count, math]\n"
        "energy:\n"
        "  m: 1\n"
        "  alpha: 0.5\n"
    )
    path = tmp_path / "mini.yaml"
    path.write_text(cfg_text)
    cfg = load_ecology_config(path)
    assert cfg.grid.families == ["word.count", "math"]
    assert cfg.energy.m == 1
    assert abs(cfg.energy.alpha - 0.5) < 1e-9

