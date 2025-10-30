import importlib.util
from pathlib import Path

_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ecology_run.py"
spec = importlib.util.spec_from_file_location("_analyzer", _ANALYZER_PATH)
assert spec and spec.loader
_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_analyzer)
summarise_generations = _analyzer.summarise_generations


def test_colonies_timeline_and_sizes_in_summary() -> None:
    records = [
        {"generation": 1, "avg_roi": 1.0, "colonies": 0},
        {
            "generation": 2,
            "avg_roi": 1.1,
            "colonies": 2,
            "colonies_meta": {
                "col_a": {"members": ["o1", "o2"]},
                "col_b": {"members": ["o3", "o4", "o5"]},
            },
        },
        {
            "generation": 3,
            "avg_roi": 1.2,
            "colonies": 1,
            "colonies_meta": {
                "col_c": {"members": ["o6", "o7"]},
            },
        },
    ]
    summary = summarise_generations(records)
    assert summary["colonies_count_series"] == [0, 2, 1]
    # avg sizes across gens: [0.0, 2.5, 2.0] -> mean 1.5
    assert abs(summary["colonies_avg_size_mean"] - 1.5) < 1e-6
