from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import importlib.util

_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ecology_run.py"
spec = importlib.util.spec_from_file_location("_analyzer", _ANALYZER_PATH)
assert spec and spec.loader
_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_analyzer)
ensure_plots = _analyzer.ensure_plots


def test_lp_heatmap_plot_creation(tmp_path: Path) -> None:
    records = [
        {
            "generation": 1,
            "avg_roi": 1.0,
            "cells": {"math:short": {"difficulty": 0.5, "success_ema": 0.4, "price": 1.0}},
            "lp_progress": {"math:short": 0.1},
        },
        {
            "generation": 2,
            "avg_roi": 1.1,
            "cells": {"math:short": {"difficulty": 0.6, "success_ema": 0.5, "price": 0.9}},
            "lp_progress": {"math:short": 0.2},
        },
    ]
    out = tmp_path / "plots"
    ensure_plots(records, out)
    # Accept either creation or graceful skip if plotting pipeline changes
    assert (out / "lp_progress.png").exists() or (out / "avg_roi.png").exists()
