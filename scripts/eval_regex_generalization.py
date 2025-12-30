#!/usr/bin/env python3
"""CLI utility for running regex generalization evaluations.

This script evaluates the generalizability of regex skills in language models,
comparing different training methods (e.g., SFT vs evolutionary LoRA).

Usage:
    # Generate evaluation tasks
    python scripts/eval_regex_generalization.py generate --output config/evaluation/regex_gen.jsonl

    # Evaluate a model interactively (for testing)
    python scripts/eval_regex_generalization.py eval --tasks config/evaluation/regex_gen.jsonl --interactive

    # Evaluate with a HuggingFace model
    python scripts/eval_regex_generalization.py eval --tasks config/evaluation/regex_gen.jsonl --model Qwen/Qwen3-0.6B

    # Compare two models
    python scripts/eval_regex_generalization.py compare \
        --report-a results/sft_report.json \
        --report-b results/evolved_report.json \
        --output comparison.json

    # Run quick smoke test
    python scripts/eval_regex_generalization.py smoke

See docs/regex_generalization.md for framework documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symbiont_ecology.evaluation.regex_generalization import (
    EvalReport,
    RegexGeneralizationEvaluator,
    compare_reports,
    generate_full_eval_suite,
    save_tasks_to_jsonl,
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_section(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'-' * 40}")
    print(f"  {text}")
    print(f"{'-' * 40}")


def format_accuracy(accuracy: float) -> str:
    """Format accuracy as percentage with color indicator."""
    pct = accuracy * 100
    if pct >= 80:
        return f"{pct:.1f}%"
    elif pct >= 50:
        return f"{pct:.1f}%"
    else:
        return f"{pct:.1f}%"


def print_report(report: EvalReport) -> None:
    """Print a formatted evaluation report."""
    print_header("REGEX GENERALIZATION EVALUATION REPORT")

    print(
        f"\nOverall: {report.total_correct}/{report.total_tasks} correct ({format_accuracy(report.overall_accuracy)})"
    )

    if report.capability_breakdown:
        print_section("Capability Breakdown")
        for cap, stats in sorted(report.capability_breakdown.items()):
            acc = format_accuracy(stats["accuracy"])
            print(f"  {cap:20s}: {int(stats['correct']):2d}/{int(stats['total']):2d} ({acc})")

    if report.holdout_breakdown:
        print_section("Hold-Out Structure Breakdown")
        for ho, stats in sorted(report.holdout_breakdown.items()):
            acc = format_accuracy(stats["accuracy"])
            print(f"  {ho:20s}: {int(stats['correct']):2d}/{int(stats['total']):2d} ({acc})")

    if report.mutation_breakdown:
        print_section("Mutation Test Breakdown")
        for mt, stats in sorted(report.mutation_breakdown.items()):
            acc = format_accuracy(stats["accuracy"])
            print(f"  {mt:20s}: {int(stats['correct']):2d}/{int(stats['total']):2d} ({acc})")

    if report.simplicity_stats:
        print_section("Simplicity Metrics (for successful synthesis tasks)")
        for key, value in sorted(report.simplicity_stats.items()):
            print(f"  {key:20s}: {value:.2f}")


def print_comparison(comparison: dict[str, Any], label_a: str, label_b: str) -> None:
    """Print a formatted comparison report."""
    print_header(f"COMPARISON: {label_a} vs {label_b}")

    overall = comparison["overall"]
    print(
        f"\n{label_a}: {format_accuracy(overall[label_a]['accuracy'])} ({overall[label_a]['total']} tasks)"
    )
    print(
        f"{label_b}: {format_accuracy(overall[label_b]['accuracy'])} ({overall[label_b]['total']} tasks)"
    )
    delta = overall["delta"]
    direction = "+" if delta >= 0 else ""
    print(f"Delta: {direction}{delta * 100:.1f}%")

    if comparison.get("capability_comparison"):
        print_section("Capability Comparison")
        for cap, stats in sorted(comparison["capability_comparison"].items()):
            delta = stats["delta"]
            direction = "+" if delta >= 0 else ""
            print(
                f"  {cap:20s}: {label_a}={format_accuracy(stats[label_a]):6s} {label_b}={format_accuracy(stats[label_b]):6s} ({direction}{delta * 100:.1f}%)"
            )

    if comparison.get("holdout_comparison"):
        print_section("Hold-Out Comparison")
        for ho, stats in sorted(comparison["holdout_comparison"].items()):
            delta = stats["delta"]
            direction = "+" if delta >= 0 else ""
            print(
                f"  {ho:20s}: {label_a}={format_accuracy(stats[label_a]):6s} {label_b}={format_accuracy(stats[label_b]):6s} ({direction}{delta * 100:.1f}%)"
            )

    if comparison.get("simplicity_comparison"):
        print_section("Simplicity Comparison (lower is better)")
        sc = comparison["simplicity_comparison"]
        delta = sc.get("delta", 0)
        direction = "+" if delta >= 0 else ""
        better = label_a if delta > 0 else label_b
        print(f"  {label_a}: {sc.get(label_a, 0):.2f}")
        print(f"  {label_b}: {sc.get(label_b, 0):.2f}")
        print(f"  Delta: {direction}{delta:.2f} ({better} produces simpler regexes)")


# ---------------------------------------------------------------------------
# Model Runners
# ---------------------------------------------------------------------------


def interactive_runner(prompt: str) -> str:
    """Interactive runner for manual testing."""
    print("\n" + "=" * 50)
    print("PROMPT:")
    print(prompt)
    print("=" * 50)
    response = input("Your response: ")
    return response


def create_hf_runner(model_name: str, device: str = "auto"):
    """Create a HuggingFace model runner."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
    )
    print("Model loaded.")

    def runner(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    return runner


def create_openai_runner(model: str = "gpt-4o-mini", api_key: str | None = None):
    """Create an OpenAI API model runner."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    def runner(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content or ""

    return runner


def create_anthropic_runner(model: str = "claude-3-haiku-20240307", api_key: str | None = None):
    """Create an Anthropic API model runner."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    def runner(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return runner


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate evaluation tasks."""
    tasks = generate_full_eval_suite()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_tasks_to_jsonl(tasks, output_path)

    print(f"Generated {len(tasks)} evaluation tasks")
    print(f"Saved to: {output_path}")

    # Print summary
    caps = {}
    holdouts = {}
    mutations = {}
    for t in tasks:
        caps[t.capability.value] = caps.get(t.capability.value, 0) + 1
        if t.holdout_type:
            holdouts[t.holdout_type.value] = holdouts.get(t.holdout_type.value, 0) + 1
        if t.mutation_type:
            mutations[t.mutation_type.value] = mutations.get(t.mutation_type.value, 0) + 1

    print("\nBy capability:")
    for cap, count in sorted(caps.items()):
        print(f"  {cap}: {count}")

    print("\nBy hold-out type:")
    for ho, count in sorted(holdouts.items()):
        print(f"  {ho}: {count}")

    print("\nBy mutation type:")
    for mt, count in sorted(mutations.items()):
        print(f"  {mt}: {count}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a model on regex generalization tasks."""
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Error: Tasks file not found: {tasks_path}")
        sys.exit(1)

    evaluator = RegexGeneralizationEvaluator.from_jsonl(tasks_path)
    print(f"Loaded {len(evaluator.tasks)} tasks from {tasks_path}")

    # Select model runner
    if args.interactive:
        runner = interactive_runner
    elif args.model:
        if args.model.startswith("gpt-"):
            runner = create_openai_runner(args.model)
        elif args.model.startswith("claude-"):
            runner = create_anthropic_runner(args.model)
        else:
            runner = create_hf_runner(args.model, device=args.device)
    else:
        print("Error: Must specify --interactive or --model")
        sys.exit(1)

    # Run evaluation
    print("\nRunning evaluation...")
    report = evaluator.evaluate_all(runner, verbose=args.verbose)

    # Print report
    print_report(report)

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two evaluation reports."""
    path_a = Path(args.report_a)
    path_b = Path(args.report_b)

    if not path_a.exists():
        print(f"Error: Report A not found: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: Report B not found: {path_b}")
        sys.exit(1)

    with path_a.open() as f:
        data_a = json.load(f)
    with path_b.open() as f:
        data_b = json.load(f)

    # Reconstruct EvalReport objects (simplified)
    report_a = EvalReport(
        total_tasks=data_a["summary"]["total_tasks"],
        total_correct=data_a["summary"]["total_correct"],
        overall_accuracy=data_a["summary"]["overall_accuracy"],
        capability_breakdown=data_a.get("capability_breakdown", {}),
        holdout_breakdown=data_a.get("holdout_breakdown", {}),
        mutation_breakdown=data_a.get("mutation_breakdown", {}),
        simplicity_stats=data_a.get("simplicity_stats", {}),
        task_results=[],
    )
    report_b = EvalReport(
        total_tasks=data_b["summary"]["total_tasks"],
        total_correct=data_b["summary"]["total_correct"],
        overall_accuracy=data_b["summary"]["overall_accuracy"],
        capability_breakdown=data_b.get("capability_breakdown", {}),
        holdout_breakdown=data_b.get("holdout_breakdown", {}),
        mutation_breakdown=data_b.get("mutation_breakdown", {}),
        simplicity_stats=data_b.get("simplicity_stats", {}),
        task_results=[],
    )

    label_a = args.label_a or "Model A"
    label_b = args.label_b or "Model B"

    comparison = compare_reports(report_a, report_b, label_a, label_b)
    print_comparison(comparison, label_a, label_b)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_path}")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run a quick smoke test."""
    print_header("REGEX GENERALIZATION SMOKE TEST")

    # Generate tasks
    tasks = generate_full_eval_suite()
    print(f"Generated {len(tasks)} tasks")

    # Create a simple "model" that gives reasonable responses
    def dummy_runner(prompt: str) -> str:
        # Return plausible responses based on task type
        if "yes or no" in prompt.lower():
            return "No, because the hour 24 is invalid."
        elif "write a regex" in prompt.lower():
            return r"\d+"
        elif "explain" in prompt.lower():
            return "This regex matches timestamps with year, month, day, hour, minute, second."
        elif "fix" in prompt.lower() or "correct" in prompt.lower():
            return r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"
        elif "simplify" in prompt.lower():
            return (
                r"^20\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"
            )
        else:
            return "42"

    evaluator = RegexGeneralizationEvaluator(tasks)
    report = evaluator.evaluate_all(dummy_runner, verbose=False)

    print("\nSmoke test complete!")
    print(f"Tasks evaluated: {report.total_tasks}")
    print(f"Accuracy: {report.overall_accuracy:.1%}")
    print("\nNote: Dummy model used - results are not meaningful.")

    # Verify core functionality works
    assert report.total_tasks > 0
    assert 0 <= report.overall_accuracy <= 1
    print("\nAll smoke tests passed!")


def cmd_list(args: argparse.Namespace) -> None:
    """List all tasks in an evaluation file."""
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Error: Tasks file not found: {tasks_path}")
        sys.exit(1)

    evaluator = RegexGeneralizationEvaluator.from_jsonl(tasks_path)

    print_header(f"TASKS IN {tasks_path.name}")
    print(f"Total: {len(evaluator.tasks)}\n")

    for i, task in enumerate(evaluator.tasks, 1):
        holdout = f" [{task.holdout_type.value}]" if task.holdout_type else ""
        mutation = f" [mut:{task.mutation_type.value}]" if task.mutation_type else ""
        print(f"{i:3d}. [{task.capability.value:12s}]{holdout}{mutation}")
        print(f"     ID: {task.task_id}")
        if args.verbose:
            print(f"     Prompt: {task.prompt[:80]}...")
        print()


def cmd_report(args: argparse.Namespace) -> None:
    """Generate full report from evaluation results."""
    from symbiont_ecology.evaluation.regex_reporting import (
        save_full_report,
    )

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)

    with report_path.open() as f:
        data = json.load(f)

    # Reconstruct EvalReport
    report = EvalReport(
        total_tasks=data["summary"]["total_tasks"],
        total_correct=data["summary"]["total_correct"],
        overall_accuracy=data["summary"]["overall_accuracy"],
        capability_breakdown=data.get("capability_breakdown", {}),
        holdout_breakdown=data.get("holdout_breakdown", {}),
        mutation_breakdown=data.get("mutation_breakdown", {}),
        simplicity_stats=data.get("simplicity_stats", {}),
        task_results=[],
    )

    output_dir = Path(args.output)
    model_name = args.model_name or "Unknown Model"

    saved = save_full_report(
        report,
        output_dir,
        model_name=model_name,
        include_latex=args.latex,
    )

    print(f"Generated reports in: {output_dir}")
    for file_type, path in saved.items():
        print(f"  - {file_type}: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regex Generalization Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate evaluation tasks")
    gen_parser.add_argument(
        "--output",
        "-o",
        default="config/evaluation/regex_generalization.jsonl",
        help="Output JSONL file path",
    )

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument(
        "--tasks",
        "-t",
        required=True,
        help="Path to tasks JSONL file",
    )
    eval_parser.add_argument(
        "--model",
        "-m",
        help="Model name (HuggingFace, OpenAI, or Anthropic)",
    )
    eval_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode (manual responses)",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for report",
    )
    eval_parser.add_argument(
        "--device",
        default="auto",
        help="Device for HuggingFace models (default: auto)",
    )
    eval_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare two evaluation reports")
    cmp_parser.add_argument(
        "--report-a",
        "-a",
        required=True,
        help="Path to first report JSON",
    )
    cmp_parser.add_argument(
        "--report-b",
        "-b",
        required=True,
        help="Path to second report JSON",
    )
    cmp_parser.add_argument(
        "--label-a",
        help="Label for first model (default: Model A)",
    )
    cmp_parser.add_argument(
        "--label-b",
        help="Label for second model (default: Model B)",
    )
    cmp_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for comparison",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List tasks in an evaluation file")
    list_parser.add_argument(
        "--tasks",
        "-t",
        required=True,
        help="Path to tasks JSONL file",
    )
    list_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show prompts",
    )

    # report command
    report_parser = subparsers.add_parser(
        "report", help="Generate full report from evaluation results"
    )
    report_parser.add_argument(
        "--report",
        "-r",
        required=True,
        help="Path to evaluation report JSON",
    )
    report_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for generated reports",
    )
    report_parser.add_argument(
        "--model-name",
        "-m",
        help="Name of the evaluated model",
    )
    report_parser.add_argument(
        "--latex",
        action="store_true",
        default=True,
        help="Include LaTeX tables (default: true)",
    )
    report_parser.add_argument(
        "--no-latex",
        action="store_false",
        dest="latex",
        help="Exclude LaTeX tables",
    )

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
