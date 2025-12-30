"""Reporting and visualization utilities for regex generalization evaluations.

This module provides utilities for generating paper-quality reports and visualizations
from regex generalization evaluation results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .regex_generalization import EvalReport


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Regex Generalization Evaluation Report"
    include_task_details: bool = False
    max_detail_tasks: int = 20
    show_failed_only: bool = False


def generate_markdown_report(
    report: EvalReport,
    config: ReportConfig | None = None,
    model_name: str = "Unknown Model",
) -> str:
    """Generate a markdown report from evaluation results.

    Args:
        report: The evaluation report to format
        config: Report configuration options
        model_name: Name of the evaluated model

    Returns:
        Markdown-formatted report string
    """
    if config is None:
        config = ReportConfig()

    lines = []

    # Header
    lines.append(f"# {config.title}")
    lines.append("")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tasks | {report.total_tasks} |")
    lines.append(f"| Correct | {report.total_correct} |")
    lines.append(f"| Accuracy | {report.overall_accuracy:.1%} |")
    lines.append("")

    # Capability Breakdown
    if report.capability_breakdown:
        lines.append("## Capability Breakdown")
        lines.append("")
        lines.append("Measures different types of regex competence:")
        lines.append("")
        lines.append("| Capability | Correct | Total | Accuracy |")
        lines.append("|------------|---------|-------|----------|")
        for cap, stats in sorted(report.capability_breakdown.items()):
            acc = stats["accuracy"]
            lines.append(f"| {cap} | {int(stats['correct'])} | {int(stats['total'])} | {acc:.1%} |")
        lines.append("")

    # Hold-Out Breakdown
    if report.holdout_breakdown:
        lines.append("## Hold-Out Structure Breakdown")
        lines.append("")
        lines.append("Measures generalization to novel regex structures:")
        lines.append("")
        lines.append("| Hold-Out Type | Correct | Total | Accuracy |")
        lines.append("|---------------|---------|-------|----------|")
        for ho, stats in sorted(report.holdout_breakdown.items()):
            acc = stats["accuracy"]
            lines.append(f"| {ho} | {int(stats['correct'])} | {int(stats['total'])} | {acc:.1%} |")
        lines.append("")

    # Mutation Breakdown
    if report.mutation_breakdown:
        lines.append("## Mutation Test Breakdown")
        lines.append("")
        lines.append("Measures local semantic reasoning about regex changes:")
        lines.append("")
        lines.append("| Mutation Type | Correct | Total | Accuracy |")
        lines.append("|---------------|---------|-------|----------|")
        for mt, stats in sorted(report.mutation_breakdown.items()):
            acc = stats["accuracy"]
            lines.append(f"| {mt} | {int(stats['correct'])} | {int(stats['total'])} | {acc:.1%} |")
        lines.append("")

    # Simplicity Metrics
    if report.simplicity_stats:
        lines.append("## Simplicity Metrics")
        lines.append("")
        lines.append("Measures solution quality for successful synthesis tasks:")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in sorted(report.simplicity_stats.items()):
            lines.append(f"| {key.replace('_', ' ').title()} | {value:.2f} |")
        lines.append("")

    # Task Details (optional)
    if config.include_task_details and report.task_results:
        lines.append("## Task Details")
        lines.append("")

        results_to_show = report.task_results
        if config.show_failed_only:
            results_to_show = [r for r in results_to_show if not r.success]

        if len(results_to_show) > config.max_detail_tasks:
            lines.append(f"*Showing {config.max_detail_tasks} of {len(results_to_show)} tasks*")
            lines.append("")
            results_to_show = results_to_show[: config.max_detail_tasks]

        for result in results_to_show:
            status = "PASS" if result.success else "FAIL"
            lines.append(f"### {result.task_id} [{status}]")
            lines.append("")
            lines.append(f"- **Capability:** {result.capability.value}")
            if result.holdout_type:
                lines.append(f"- **Hold-Out:** {result.holdout_type.value}")
            if result.mutation_type:
                lines.append(f"- **Mutation:** {result.mutation_type.value}")
            lines.append(
                f"- **Response:** `{result.response[:100]}...`"
                if len(result.response) > 100
                else f"- **Response:** `{result.response}`"
            )
            if result.details:
                lines.append(f"- **Details:** {json.dumps(result.details, indent=2)[:200]}")
            lines.append("")

    return "\n".join(lines)


def generate_comparison_markdown(
    comparison: dict[str, Any],
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> str:
    """Generate a markdown comparison report.

    Args:
        comparison: Comparison dict from compare_reports()
        label_a: Label for first model
        label_b: Label for second model

    Returns:
        Markdown-formatted comparison report
    """
    lines = []

    lines.append("# Regex Generalization Comparison Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall comparison
    lines.append("## Overall Comparison")
    lines.append("")
    overall = comparison["overall"]
    delta = overall["delta"]
    delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
    winner = label_b if delta > 0 else label_a if delta < 0 else "Tie"

    lines.append("| Model | Accuracy | Tasks |")
    lines.append("|-------|----------|-------|")
    lines.append(
        f"| {label_a} | {overall[label_a]['accuracy']:.1%} | {overall[label_a]['total']} |"
    )
    lines.append(
        f"| {label_b} | {overall[label_b]['accuracy']:.1%} | {overall[label_b]['total']} |"
    )
    lines.append(f"| **Delta** | **{delta_str}** | - |")
    lines.append("")
    lines.append(f"**Winner:** {winner}")
    lines.append("")

    # Capability comparison
    if comparison.get("capability_comparison"):
        lines.append("## Capability Comparison")
        lines.append("")
        lines.append(f"| Capability | {label_a} | {label_b} | Delta | Better |")
        lines.append("|------------|----------|----------|-------|--------|")
        for cap, stats in sorted(comparison["capability_comparison"].items()):
            delta = stats["delta"]
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            better = label_b if delta > 0 else label_a if delta < 0 else "="
            lines.append(
                f"| {cap} | {stats[label_a]:.1%} | {stats[label_b]:.1%} | {delta_str} | {better} |"
            )
        lines.append("")

    # Hold-out comparison
    if comparison.get("holdout_comparison"):
        lines.append("## Hold-Out Structure Comparison")
        lines.append("")
        lines.append(f"| Hold-Out Type | {label_a} | {label_b} | Delta | Better |")
        lines.append("|---------------|----------|----------|-------|--------|")
        for ho, stats in sorted(comparison["holdout_comparison"].items()):
            delta = stats["delta"]
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            better = label_b if delta > 0 else label_a if delta < 0 else "="
            lines.append(
                f"| {ho} | {stats[label_a]:.1%} | {stats[label_b]:.1%} | {delta_str} | {better} |"
            )
        lines.append("")

    # Mutation comparison
    if comparison.get("mutation_comparison"):
        lines.append("## Mutation Test Comparison")
        lines.append("")
        lines.append(f"| Mutation Type | {label_a} | {label_b} | Delta | Better |")
        lines.append("|---------------|----------|----------|-------|--------|")
        for mt, stats in sorted(comparison["mutation_comparison"].items()):
            delta = stats["delta"]
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            better = label_b if delta > 0 else label_a if delta < 0 else "="
            lines.append(
                f"| {mt} | {stats[label_a]:.1%} | {stats[label_b]:.1%} | {delta_str} | {better} |"
            )
        lines.append("")

    # Simplicity comparison
    if comparison.get("simplicity_comparison"):
        lines.append("## Simplicity Comparison")
        lines.append("")
        sc = comparison["simplicity_comparison"]
        delta = sc.get("delta", 0)
        # Lower is better for complexity
        better = label_a if delta > 0 else label_b if delta < 0 else "="
        lines.append("*Note: Lower complexity score indicates simpler, more elegant solutions.*")
        lines.append("")
        lines.append("| Model | Mean Complexity |")
        lines.append("|-------|-----------------|")
        lines.append(f"| {label_a} | {sc.get(label_a, 0):.2f} |")
        lines.append(f"| {label_b} | {sc.get(label_b, 0):.2f} |")
        lines.append("")
        lines.append(f"**Better (simpler solutions):** {better}")
        lines.append("")

    return "\n".join(lines)


def generate_latex_table(
    report: EvalReport,
    table_type: str = "capability",
) -> str:
    """Generate a LaTeX table for paper inclusion.

    Args:
        report: The evaluation report
        table_type: One of 'capability', 'holdout', 'mutation', 'summary'

    Returns:
        LaTeX-formatted table string
    """
    lines = []

    if table_type == "summary":
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lr}")
        lines.append(r"\toprule")
        lines.append(r"Metric & Value \\")
        lines.append(r"\midrule")
        lines.append(f"Total Tasks & {report.total_tasks} \\\\")
        lines.append(f"Correct & {report.total_correct} \\\\")
        lines.append(f"Accuracy & {report.overall_accuracy:.1%} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Summary of regex generalization evaluation results.}")
        lines.append(r"\label{tab:regex-summary}")
        lines.append(r"\end{table}")

    elif table_type == "capability" and report.capability_breakdown:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"Capability & Correct & Total & Accuracy \\")
        lines.append(r"\midrule")
        for cap, stats in sorted(report.capability_breakdown.items()):
            acc = stats["accuracy"]
            cap_formatted = cap.replace("_", " ").title()
            lines.append(
                f"{cap_formatted} & {int(stats['correct'])} & {int(stats['total'])} & {acc:.1%} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Accuracy breakdown by regex capability axis.}")
        lines.append(r"\label{tab:regex-capability}")
        lines.append(r"\end{table}")

    elif table_type == "holdout" and report.holdout_breakdown:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"Hold-Out Type & Correct & Total & Accuracy \\")
        lines.append(r"\midrule")
        for ho, stats in sorted(report.holdout_breakdown.items()):
            acc = stats["accuracy"]
            ho_formatted = ho.replace("_", " ").title()
            lines.append(
                f"{ho_formatted} & {int(stats['correct'])} & {int(stats['total'])} & {acc:.1%} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Accuracy breakdown by hold-out structure type.}")
        lines.append(r"\label{tab:regex-holdout}")
        lines.append(r"\end{table}")

    elif table_type == "mutation" and report.mutation_breakdown:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"Mutation Type & Correct & Total & Accuracy \\")
        lines.append(r"\midrule")
        for mt, stats in sorted(report.mutation_breakdown.items()):
            acc = stats["accuracy"]
            mt_formatted = mt.replace("_", " ").title()
            lines.append(
                f"{mt_formatted} & {int(stats['correct'])} & {int(stats['total'])} & {acc:.1%} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Accuracy breakdown by mutation/counterfactual test type.}")
        lines.append(r"\label{tab:regex-mutation}")
        lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_comparison_latex_table(
    comparison: dict[str, Any],
    label_a: str = "SFT",
    label_b: str = "Evolved",
) -> str:
    """Generate a LaTeX comparison table for paper inclusion.

    Args:
        comparison: Comparison dict from compare_reports()
        label_a: Label for first model
        label_b: Label for second model

    Returns:
        LaTeX-formatted comparison table
    """
    lines = []

    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(f"Capability & {label_a} & {label_b} & $\\Delta$ \\\\")
    lines.append(r"\midrule")

    if comparison.get("capability_comparison"):
        for cap, stats in sorted(comparison["capability_comparison"].items()):
            delta = stats["delta"]
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            cap_formatted = cap.replace("_", " ").title()
            lines.append(
                f"{cap_formatted} & {stats[label_a]:.1%} & {stats[label_b]:.1%} & {delta_str} \\\\"
            )

    lines.append(r"\midrule")
    overall = comparison["overall"]
    delta = overall["delta"]
    delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
    lines.append(
        f"\\textbf{{Overall}} & \\textbf{{{overall[label_a]['accuracy']:.1%}}} & \\textbf{{{overall[label_b]['accuracy']:.1%}}} & \\textbf{{{delta_str}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        f"\\caption{{Comparison of regex generalization between {label_a} and {label_b} training methods.}}"
    )
    lines.append(r"\label{tab:regex-comparison}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# Import matplotlib if available (used for plotting)
try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def save_full_report(
    report: EvalReport,
    output_dir: Path,
    model_name: str = "Unknown Model",
    include_latex: bool = True,
) -> dict[str, Path]:
    """Save a complete set of report files.

    Args:
        report: The evaluation report
        output_dir: Directory to save files to
        model_name: Name of the evaluated model
        include_latex: Whether to include LaTeX tables

    Returns:
        Dict mapping file type to path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # Save JSON
    json_path = output_dir / "report.json"
    with json_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
    saved_files["json"] = json_path

    # Save markdown
    config = ReportConfig(include_task_details=True, max_detail_tasks=50)
    md_content = generate_markdown_report(report, config, model_name)
    md_path = output_dir / "report.md"
    with md_path.open("w") as f:
        f.write(md_content)
    saved_files["markdown"] = md_path

    # Save LaTeX tables
    if include_latex:
        latex_lines = []
        latex_lines.append("% Regex Generalization Evaluation Tables")
        latex_lines.append(f"% Model: {model_name}")
        latex_lines.append(f"% Generated: {datetime.now().isoformat()}")
        latex_lines.append("")

        latex_lines.append(generate_latex_table(report, "summary"))
        latex_lines.append("")

        if report.capability_breakdown:
            latex_lines.append(generate_latex_table(report, "capability"))
            latex_lines.append("")

        if report.holdout_breakdown:
            latex_lines.append(generate_latex_table(report, "holdout"))
            latex_lines.append("")

        if report.mutation_breakdown:
            latex_lines.append(generate_latex_table(report, "mutation"))
            latex_lines.append("")

        latex_path = output_dir / "tables.tex"
        with latex_path.open("w") as f:
            f.write("\n".join(latex_lines))
        saved_files["latex"] = latex_path

    # Generate plot if matplotlib is available
    if HAS_MATPLOTLIB:
        plot_path = output_dir / "results.png"
        plot_single_report(report, model_name, plot_path)
        saved_files["plot"] = plot_path

    return saved_files


def plot_single_report(
    report: EvalReport,
    model_name: str = "Model",
    output_path: Path | None = None,
) -> Any:
    """Plot bar charts for a single model's evaluation results.

    Args:
        report: The evaluation report
        model_name: Name of the model
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object (if matplotlib available)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Count how many plots we need
    n_plots = 1  # capability always
    if report.holdout_breakdown:
        n_plots += 1
    if report.mutation_breakdown:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Capability breakdown
    if report.capability_breakdown:
        ax = axes[plot_idx]
        caps = sorted(report.capability_breakdown.keys())
        scores = [report.capability_breakdown[c]["accuracy"] * 100 for c in caps]
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(caps)))

        bars = ax.bar(range(len(caps)), scores, color=colors)
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([c.replace("_", "\n") for c in caps], fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("By Capability")
        ax.set_ylim(0, 100)
        ax.axhline(
            y=report.overall_accuracy * 100,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Overall: {report.overall_accuracy:.0%}",
        )
        ax.legend(fontsize=8)

        for bar, score in zip(bars, scores):
            ax.annotate(
                f"{score:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, score),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        plot_idx += 1

    # Holdout breakdown
    if report.holdout_breakdown:
        ax = axes[plot_idx]
        holdouts = sorted(report.holdout_breakdown.keys())
        scores = [report.holdout_breakdown[h]["accuracy"] * 100 for h in holdouts]
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(holdouts)))

        bars = ax.bar(range(len(holdouts)), scores, color=colors)
        ax.set_xticks(range(len(holdouts)))
        ax.set_xticklabels([h.replace("_", "\n") for h in holdouts], fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("By Hold-Out Type")
        ax.set_ylim(0, 100)

        for bar, score in zip(bars, scores):
            ax.annotate(
                f"{score:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, score),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        plot_idx += 1

    # Mutation breakdown
    if report.mutation_breakdown:
        ax = axes[plot_idx]
        mutations = sorted(report.mutation_breakdown.keys())
        scores = [report.mutation_breakdown[m]["accuracy"] * 100 for m in mutations]
        colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(mutations)))

        bars = ax.bar(range(len(mutations)), scores, color=colors)
        ax.set_xticks(range(len(mutations)))
        ax.set_xticklabels([m.replace("_", "\n") for m in mutations], fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("By Mutation Type")
        ax.set_ylim(0, 100)

        for bar, score in zip(bars, scores):
            ax.annotate(
                f"{score:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, score),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle(f"Regex Generalization: {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_capability_comparison(
    comparison: dict[str, Any],
    label_a: str = "SFT",
    label_b: str = "Evolved",
    output_path: Path | None = None,
) -> Any:
    """Plot a bar chart comparing capabilities between two models.

    Args:
        comparison: Comparison dict from compare_reports()
        label_a: Label for first model
        label_b: Label for second model
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object (if matplotlib available)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    cap_data = comparison.get("capability_comparison", {})
    if not cap_data:
        raise ValueError("No capability comparison data available")

    capabilities = sorted(cap_data.keys())
    a_scores = [cap_data[c][label_a] * 100 for c in capabilities]
    b_scores = [cap_data[c][label_b] * 100 for c in capabilities]

    x = np.arange(len(capabilities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - width / 2, a_scores, width, label=label_a, color="#4C72B0")
    bars_b = ax.bar(x + width / 2, b_scores, width, label=label_b, color="#55A868")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Regex Generalization: Capability Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in capabilities], rotation=0)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars_a, bars_b]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_holdout_comparison(
    comparison: dict[str, Any],
    label_a: str = "SFT",
    label_b: str = "Evolved",
    output_path: Path | None = None,
) -> Any:
    """Plot a bar chart comparing hold-out structure performance.

    Args:
        comparison: Comparison dict from compare_reports()
        label_a: Label for first model
        label_b: Label for second model
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object (if matplotlib available)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    holdout_data = comparison.get("holdout_comparison", {})
    if not holdout_data:
        raise ValueError("No hold-out comparison data available")

    holdouts = sorted(holdout_data.keys())
    a_scores = [holdout_data[h][label_a] * 100 for h in holdouts]
    b_scores = [holdout_data[h][label_b] * 100 for h in holdouts]

    x = np.arange(len(holdouts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - width / 2, a_scores, width, label=label_a, color="#4C72B0")
    bars_b = ax.bar(x + width / 2, b_scores, width, label=label_b, color="#55A868")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Regex Generalization: Hold-Out Structure Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([h.replace("_", "\n") for h in holdouts], rotation=0)
    ax.legend()
    ax.set_ylim(0, 100)

    for bars in [bars_a, bars_b]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_radar_comparison(
    comparison: dict[str, Any],
    label_a: str = "SFT",
    label_b: str = "Evolved",
    output_path: Path | None = None,
) -> Any:
    """Plot a radar chart comparing all dimensions.

    Args:
        comparison: Comparison dict from compare_reports()
        label_a: Label for first model
        label_b: Label for second model
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object (if matplotlib available)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Collect all metrics
    metrics = []
    a_values = []
    b_values = []

    for cap, stats in sorted(comparison.get("capability_comparison", {}).items()):
        metrics.append(cap)
        a_values.append(stats[label_a] * 100)
        b_values.append(stats[label_b] * 100)

    for ho, stats in sorted(comparison.get("holdout_comparison", {}).items()):
        metrics.append(f"HO:{ho[:4]}")
        a_values.append(stats[label_a] * 100)
        b_values.append(stats[label_b] * 100)

    if not metrics:
        raise ValueError("No data available for radar chart")

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    a_values += a_values[:1]
    b_values += b_values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, a_values, "o-", linewidth=2, label=label_a, color="#4C72B0")
    ax.fill(angles, a_values, alpha=0.25, color="#4C72B0")
    ax.plot(angles, b_values, "o-", linewidth=2, label=label_b, color="#55A868")
    ax.fill(angles, b_values, alpha=0.25, color="#55A868")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title("Regex Generalization: Multi-Dimension Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
