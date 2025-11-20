import importlib.util
import sys
from pathlib import Path

SCRIPTS_PATH = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))


def _load_evoscope():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "evoscope.py"
    spec = importlib.util.spec_from_file_location("evoscope_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _sample_records():
    return [
        {
            "generation": 1,
            "avg_roi": 0.8,
            "avg_energy_cost": 1.2,
            "mean_energy_balance": 4.0,
            "merges": 0,
            "trials_created": 1,
            "promotions": 0,
            "team_routes": 2,
            "team_promotions": 0,
            "colonies": 1,
            "lp_mix_active": 0.4,
            "qd_archive_size": 3,
            "qd_archive_coverage": 0.15,
        },
        {
            "generation": 2,
            "avg_roi": 1.1,
            "avg_energy_cost": 1.5,
            "mean_energy_balance": 4.3,
            "merges": 1,
            "trials_created": 2,
            "promotions": 1,
            "team_routes": 3,
            "team_promotions": 1,
            "colonies": 2,
            "lp_mix_active": 0.55,
            "qd_archive_size": 4,
            "qd_archive_coverage": 0.22,
        },
    ]


def _sample_summary():
    return {
        "generations": 2,
        "avg_roi_mean": 0.95,
        "avg_energy_mean": 1.35,
        "energy_balance_mean": 4.15,
        "trials_total": 3,
        "promotions_total": 1,
        "total_merges": 1,
        "diversity_samples": 5,
        "diversity_energy_gini_mean": 0.2,
        "lp_mix_active_last": 0.55,
        "lp_mix_base_mean": 0.5,
        "assimilation_gating_total": {"low_power": 4, "uplift_below_threshold": 2},
        "assimilation_gating_samples": [
            {"generation": 2, "organelle": "org_a", "reason": "low_power", "details": "power=0.08"}
        ],
        "co_routing_totals": {"org_a:org_b": 3},
        "colony_tier_counts_total": {"0": 1, "1": 1},
        "qd_archive_top": [
            {"cell": "word.count:short", "bin": 0, "organelle": "org_a", "roi": 1.2, "novelty": 0.4},
            {"cell": "math:medium", "bin": 1, "organelle": "org_b", "roi": 1.05, "novelty": 0.3},
        ],
    }


def _sample_assim_summary():
    return {
        "events": 4,
        "passes": 1,
        "failures": 3,
        "sample_size_mean": 6.0,
        "ci_excludes_zero_rate": 0.5,
        "power_mean": 0.42,
        "methods": {"z_test": 3, "dr": 1},
        "dr_used": 1,
        "dr_strata_top": [("math", 4)],
    }


def test_build_html_includes_charts(tmp_path: Path) -> None:
    evoscope = _load_evoscope()
    visuals = tmp_path / "visuals"
    visuals.mkdir()
    # create fake image so gallery renders something
    (visuals / "avg_roi.png").write_bytes(b"fake")

    html = evoscope.build_html(_sample_summary(), _sample_records(), _sample_assim_summary(), tmp_path)

    assert "cdn.jsdelivr.net/npm/chart.js" in html
    assert "const evoData" in html
    assert "roiChart" in html
    assert "Assimilation Summary" in html
    assert "low_power" in html
    assert "org_a" in html
    assert "qdChart" in html
    assert "QD archive top bins" in html
