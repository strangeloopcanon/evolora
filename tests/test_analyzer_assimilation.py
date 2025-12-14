import importlib.util
import json
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
            "soup": [
                {"organelle_id": "a", "weight": 0.6, "importance": 1.0},
                {"organelle_id": "b", "weight": 0.4, "importance": 0.4},
            ],
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
    assert "fisher_importance_mean" in out
    assert out["fisher_importance_max"] >= out["fisher_importance_mean"]
    assert "merge_weight_mean" in out


def test_summarise_generations_collects_assimilation_history() -> None:
    records = [
        {
            "generation": 1,
            "avg_roi": 0.1,
            "avg_total": 0.2,
            "avg_energy_cost": 0.3,
            "active": 1,
            "bankrupt": 0,
            "merges": 0,
            "culled_bankrupt": 0,
            "mean_energy_balance": 0.4,
            "lp_mix_base": 0.0,
            "lp_mix_active": 0.0,
            "assimilation_gating": {},
            "assimilation_history": {
                "org:math:short": [
                    {"generation": 1, "uplift": 0.15, "p_value": 0.05},
                ]
            },
        }
    ]
    summary = _an.summarise_generations(records)
    series = summary.get("assimilation_history_series") or {}
    assert "org:math:short" in series
    assert series["org:math:short"][0]["generation"] == 1
