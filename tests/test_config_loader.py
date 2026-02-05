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


def test_load_ecology_config_normalizes_legacy_top_level_keys(tmp_path: Path) -> None:
    yaml = (
        "model:\n"
        "  name: Qwen/Qwen3-0.6B\n"
        "  device: cpu\n"
        "env:\n"
        "  grid:\n"
        "    families: [math, word.count]\n"
        "    depths: [short]\n"
        "teacher:\n"
        "  tau: 0.7\n"
        "  beta: 0.2\n"
        "  eta: 0.3\n"
        "population:\n"
        "  mu: 3\n"
        "  lambda: 9\n"
        "  max_population: 12\n"
    )
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml)
    cfg = load_ecology_config(path)
    assert cfg.host.backbone_model == "Qwen/Qwen3-0.6B"
    assert cfg.host.tokenizer == "Qwen/Qwen3-0.6B"
    assert cfg.host.device == "cpu"
    assert cfg.grid.families == ["math", "word.count"]
    assert cfg.grid.depths == ["short"]
    assert abs(cfg.controller.tau - 0.7) < 1e-9
    assert cfg.population_strategy.mu == 3
    assert cfg.population_strategy.lambda_ == 9
    assert cfg.population_strategy.max_population == 12
