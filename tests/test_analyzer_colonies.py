import importlib.util
import pytest
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
    # avg sizes across gens: [0.0, 2.5, 2.0] -> mean 1.5 (allow tiny tolerance)
    assert summary["colonies_avg_size_mean"] == pytest.approx(1.5, abs=1e-6)


def test_colony_selection_aggregates_series() -> None:
    records = [
        {
            "generation": 1,
            "avg_roi": 1.0,
            "colonies": 2,
            "colony_selection": {"dissolved": 1, "replicated": 0, "pool_members": 2, "pool_pot": 1.0},
            "colony_selection_events": [{"type": "selection", "dissolved": "col_x", "best": "col_y", "created": 0}],
            "colony_selection_pool": [{"type": "pool_add", "count": 2, "pot": 1.0}],
            "colony_metrics": {
                "col_x": {
                    "bandwidth_budget": 0.5,
                    "size": 2,
                    "last_delta": 0.1,
                    "variance_ratio": 0.8,
                    "hazard_members": 0,
                    "tier": 1,
                }
            },
        },
        {
            "generation": 2,
            "avg_roi": 1.1,
            "colonies": 2,
            "colony_selection": {"dissolved": 0, "replicated": 1, "pool_members": 1, "pool_pot": 0.5},
            "colony_selection_events": [{"type": "replicate", "from": "col_y", "child": "col_y_c2"}],
            "colony_metrics": {
                "col_y": {
                    "bandwidth_budget": 0.75,
                    "size": 3,
                    "last_delta": 0.12,
                    "variance_ratio": 0.7,
                    "hazard_members": 1,
                    "tier": 2,
                }
            },
        },
    ]
    summary = summarise_generations(records)
    assert summary["colony_selection_dissolved_series"] == [1, 0]
    assert summary["colony_selection_replicated_series"] == [0, 1]
    assert summary["colony_selection_pool_members_series"] == [2, 1]
    assert summary["colony_selection_pool_pot_series"] == [1.0, 0.5]
    assert summary["colony_selection_dissolved_total"] == 1
    assert summary["colony_selection_replicated_total"] == 1
    assert summary["colony_selection_events"]
    assert summary["colony_tier_mean_series"][0] == pytest.approx(1.0)
    assert summary["colony_tier_mean_series"][1] == pytest.approx(2.0)
    assert summary["colony_tier_counts_total"]["1"] == 1
    assert summary["colony_tier_counts_total"]["2"] == 1


def test_merge_audit_summary_fields() -> None:
    records = [
        {
            "generation": 1,
            "avg_roi": 1.0,
            "merge_audits": [
                {"organelle_id": "org_a", "cell": {"family": "math", "depth": "short"}, "pre_roi": 0.2, "post_roi": 0.35, "delta": 0.15, "tasks": 3}
            ],
        },
        {
            "generation": 2,
            "avg_roi": 1.05,
            "merge_audits": [],
        },
        {
            "generation": 3,
            "avg_roi": 1.1,
            "merge_audits": [
                {"organelle_id": "org_b", "cell": {"family": "word", "depth": "short"}, "pre_roi": 0.6, "post_roi": 0.5, "delta": -0.1, "tasks": 4}
            ],
            "assimilation_attempts": [
                {
                    "generation": 3,
                    "organelle_id": "org_a",
                    "cell": {"family": "math", "depth": "short"},
                    "uplift": 0.2,
                    "p_value": 0.01,
                    "passes_stat_test": True,
                    "holdout_passed": True,
                    "global_probe_passed": True,
                    "holdout": {"candidate_roi": 0.4},
                    "audit": {"delta": 0.05, "tasks": 3},
                },
                {
                    "generation": 3,
                    "organelle_id": "org_b",
                    "cell": {"family": "word", "depth": "short"},
                    "uplift": -0.05,
                    "p_value": 0.2,
                    "passes_stat_test": False,
                    "holdout_passed": False,
                    "global_probe_passed": False,
                    "holdout": {"candidate_roi": 0.3},
                    "audit": {"delta": -0.1, "tasks": 4},
                }
            ],
        },
    ]
    summary = summarise_generations(records)
    assert summary["merge_audit_count_series"] == [1, 0, 1]
    assert summary["merge_audit_total"] == 2
    assert summary["merge_audit_delta_series"][0] == pytest.approx(0.15)
    assert summary["merge_audit_delta_series"][1] is None
    assert summary["merge_audit_delta_series"][2] == pytest.approx(-0.1)
    assert summary["merge_audit_delta_mean"] == pytest.approx((0.15 - 0.1) / 2, rel=1e-6)
    assert len(summary["merge_audit_records"]) == 2
    fam_stats = summary["assimilation_family_summary"]["math"]
    assert fam_stats["attempts"] == 1
    assert fam_stats["pass_rate"] == pytest.approx(1.0)


def test_evaluation_family_stats_summary() -> None:
    records = [
        {
            "generation": 1,
            "avg_roi": 1.0,
            "evaluation": {
                "accuracy": 1.0,
                "correct": 1,
                "total": 1,
                "avg_cost": 0.2,
                "avg_roi": 1.4,
                "avg_delta": 0.3,
                "evaluated_routes": 1,
                "family_breakdown": {
                    "math.multi_step": {
                        "accuracy": 1.0,
                        "correct": 1,
                        "total": 1,
                        "avg_cost": 0.2,
                        "avg_roi": 1.4,
                        "avg_delta": 0.3,
                        "count": 1,
                    }
                },
            },
        },
        {
            "generation": 2,
            "avg_roi": 1.05,
        },
    ]
    summary = summarise_generations(records)
    stats = summary["evaluation_family_stats"]["math.multi_step"]
    assert stats["accuracy"] == pytest.approx(1.0)
    assert stats["avg_delta"] == pytest.approx(0.3)
