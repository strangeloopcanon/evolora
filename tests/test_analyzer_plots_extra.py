import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_AN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ecology_run.py"
spec = importlib.util.spec_from_file_location("_an", _AN_PATH)
assert spec and spec.loader
_an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_an)


def test_colony_plots_created(tmp_path: Path) -> None:
    records = [
        {"generation": 1, "avg_roi": 1.0, "colonies": 0, "cells": {}},
        {
            "generation": 2,
            "avg_roi": 1.1,
            "colonies": 1,
            "colonies_meta": {"c": {"members": ["o1", "o2"]}},
            "cells": {},
        },
        {
            "generation": 3,
            "avg_roi": 1.2,
            "colonies": 1,
            "colonies_meta": {"c": {"members": ["o1", "o2", "o3"]}},
            "cells": {},
        },
    ]
    out = tmp_path / "plots"
    _an.ensure_plots(records, out)
    assert (out / "colonies_count.png").exists()
    assert (out / "colonies_avg_size.png").exists()
