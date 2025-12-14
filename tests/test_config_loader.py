from pathlib import Path

from symbiont_ecology.config import load_ecology_config


def test_load_ecology_config_roundtrip(tmp_path: Path):
    yaml = (
        "host:\n  backbone_model: google/gemma-3-270m-it\n"
        "grid:\n  families: [word.count]\n  depths: [short]\n"
        "energy:\n  Emax: 5.0\n  m: 1.0\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml)
    cfg = load_ecology_config(path)
    assert cfg.host.backbone_model.endswith("gemma-3-270m-it")
    assert "word.count" in cfg.grid.families
    assert cfg.energy.Emax == 5.0
