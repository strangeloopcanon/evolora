from pathlib import Path

from symbiont_ecology.evaluation.regex_generalization import (
    CapabilityAxis,
    EvalReport,
    HoldOutType,
    MutationType,
    TaskResult,
)
from symbiont_ecology.evaluation.regex_reporting import (
    generate_comparison_latex_table,
    generate_comparison_markdown,
    plot_capability_comparison,
    plot_holdout_comparison,
    plot_radar_comparison,
    save_full_report,
)


def _sample_report() -> EvalReport:
    task_results = [
        TaskResult(
            task_id="t_short",
            capability=CapabilityAxis.SYNTHESIS,
            holdout_type=HoldOutType.OPERATOR,
            mutation_type=MutationType.WIDEN_RANGE,
            success=True,
            response="^a+$",
            details={"note": "short response"},
        ),
        TaskResult(
            task_id="t_long",
            capability=CapabilityAxis.DEBUGGING,
            holdout_type=None,
            mutation_type=None,
            success=False,
            response="x" * 200,
            details={"reason": "long response triggers truncation path"},
        ),
    ]
    return EvalReport(
        total_tasks=10,
        total_correct=7,
        overall_accuracy=0.7,
        capability_breakdown={
            "synthesis": {"correct": 3, "total": 5, "accuracy": 0.6},
            "debugging": {"correct": 4, "total": 5, "accuracy": 0.8},
        },
        holdout_breakdown={
            "operator": {"correct": 2, "total": 3, "accuracy": 2 / 3},
        },
        mutation_breakdown={
            "widen_range": {"correct": 1, "total": 2, "accuracy": 0.5},
        },
        simplicity_stats={"mean_complexity": 12.34},
        task_results=task_results,
    )


def test_regex_reporting_save_full_report_smoke(tmp_path: Path) -> None:
    report = _sample_report()
    out = save_full_report(report, tmp_path, model_name="TestModel", include_latex=True)

    assert out["json"].exists()
    assert out["markdown"].exists()
    assert out["latex"].exists()
    assert out["plot"].exists()


def test_regex_reporting_comparison_plots_smoke(tmp_path: Path) -> None:
    comparison = {
        "overall": {
            "SFT": {"accuracy": 0.5, "total": 10},
            "Evolved": {"accuracy": 0.6, "total": 10},
            "delta": 0.1,
        },
        "capability_comparison": {
            "synthesis": {"SFT": 0.4, "Evolved": 0.7, "delta": 0.3},
        },
        "holdout_comparison": {
            "operator": {"SFT": 0.5, "Evolved": 0.5, "delta": 0.0},
        },
    }

    md = generate_comparison_markdown(comparison, label_a="SFT", label_b="Evolved")
    tex = generate_comparison_latex_table(comparison, label_a="SFT", label_b="Evolved")
    assert "Overall Comparison" in md
    assert r"\begin{table}" in tex

    cap_path = tmp_path / "cap.png"
    holdout_path = tmp_path / "holdout.png"
    radar_path = tmp_path / "radar.png"

    fig1 = plot_capability_comparison(comparison, output_path=cap_path)
    fig2 = plot_holdout_comparison(comparison, output_path=holdout_path)
    fig3 = plot_radar_comparison(comparison, output_path=radar_path)

    assert cap_path.exists()
    assert holdout_path.exists()
    assert radar_path.exists()

    fig1.clf()
    fig2.clf()
    fig3.clf()
