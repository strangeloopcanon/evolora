from pathlib import Path
import json
import importlib.util
from types import ModuleType


def _load_analyzer() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "analyze_ecology_run.py"
    spec = importlib.util.spec_from_file_location("analyze_ecology_run", mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_summarise_assimilation_dr_strata(tmp_path: Path):
    path = tmp_path / "assimilation.jsonl"
    events = [
        {
            "type": "assimilation",
            "decision": False,
            "sample_size": 6,
            "ci_low": -0.01,
            "ci_high": 0.03,
            "power": 0.2,
            "method": "z_test+bootstrap",
            "dr_used": True,
            "strata": {"family": {"paired": 3}, "depth": {"paired": 2}},
        },
        {
            "type": "assimilation",
            "decision": True,
            "sample_size": 10,
            "ci_low": 0.02,
            "ci_high": 0.06,
            "power": 0.7,
            "method": "z_test+bootstrap",
            "dr_used": True,
            "strata": {"family": {"paired": 4}, "depth": {"paired": 1}},
        },
    ]
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    analyzer = _load_analyzer()
    out = analyzer.summarise_assimilation(path)
    assert out.get("events") == 2
    assert out.get("dr_used") == 2
    top = dict(out.get("dr_strata_top") or [])
    assert top.get("family") == 7 and top.get("depth") == 3
