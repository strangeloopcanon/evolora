import json
import importlib.util
from pathlib import Path

_AN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ecology_run.py"
spec = importlib.util.spec_from_file_location("_an", _AN_PATH)
assert spec and spec.loader
_an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_an)


def test_summarise_assimilation_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "assim.jsonl"
    records = [
        {
            "type": "assimilation",
            "decision": True,
            "sample_size": 8,
            "ci_low": -0.1,
            "ci_high": 0.2,
            "power": 0.6,
            "method": "z_test+bootstrap",
            "dr_used": False,
        },
        {
            "type": "assimilation",
            "decision": False,
            "sample_size": 6,
            "ci_low": -0.2,
            "ci_high": 0.1,
            "power": 0.5,
            "method": "dr+bootstrap",
            "dr_used": True,
        },
    ]
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    out = _an.summarise_assimilation(path)
    assert out["events"] == 2
    assert out["passes"] == 1
    assert out["failures"] == 1
    assert "power_mean" in out
    assert "methods" in out
