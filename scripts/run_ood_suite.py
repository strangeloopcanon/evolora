#!/usr/bin/env python3
"""Run a small OOD evaluation suite for an evo-vs-SFT run directory.

This script is meant to answer: "does evolution shine on tasks outside the training mix?"

It evaluates base vs SFT vs evo (optionally routed) on a few holdout sets and writes:
  - one JSON per holdout (raw eval outputs from scripts/evaluate_holdout.py)
  - a combined summary.json + summary.md

Notes:
  - For evo routed evaluation, selection tasks must be different from the holdout to avoid leakage.
  - For OOD, it's useful to report both:
      * "frozen" routing (reuse the run's final_holdout.json routing map, if available)
      * "reselect" routing (pick specialists using a small OOD selection set)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalCase:
    name: str
    holdout_path: Path
    routing: str  # single|family|cell
    selection_tasks_path: Path | None = None
    routing_json_path: Path | None = None
    max_samples: int | None = None
    holdout_sampling: str | None = None
    holdout_seed: int | None = None


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _extract_model_accuracy(eval_payload: dict[str, Any]) -> dict[str, float]:
    rows = eval_payload.get("results") or []
    acc: dict[str, float] = {}
    for row in rows:
        model = str(row.get("model") or row.get("model_name") or "")
        if not model:
            continue
        accuracy = row.get("accuracy")
        if accuracy is None:
            continue
        acc[model] = float(accuracy)
    return acc


def _pick_evo_accuracy(acc: dict[str, float]) -> float | None:
    for key in sorted(acc.keys()):
        if key.startswith("evolution"):
            return acc[key]
    return None


def _make_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# OOD Suite Summary")
    lines.append("")
    lines.append(f"- created_at: `{summary.get('created_at')}`")
    lines.append(f"- run_dir: `{summary.get('run_dir')}`")
    lines.append(f"- model: `{summary.get('model')}`")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| case | base | sft | evo | notes |")
    lines.append("| --- | ---: | ---: | ---: | --- |")

    for case in summary.get("cases", []):
        name = str(case.get("name", ""))
        acc = case.get("accuracy", {}) or {}
        base = _format_pct(acc.get("base"))
        sft = _format_pct(acc.get("sft"))
        evo = _format_pct(acc.get("evolution"))
        notes = str(case.get("notes", "") or "")
        lines.append(f"| {name} | {base} | {sft} | {evo} | {notes} |")

    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _ensure_paper_selection_set(
    *,
    out_dir: Path,
    seed: int,
    selection_size: int,
) -> Path:
    datasets_dir = out_dir / "paper_selection_datasets"
    selection_path = datasets_dir / "selection_tasks.jsonl"
    if selection_path.exists():
        return selection_path

    datasets_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("AGENT_MODE", "baseline")
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_grid_datasets.py"),
            "--config",
            str(ROOT / "config" / "experiments" / "paper_qwen3_ecology.yaml"),
            "--seed",
            str(seed),
            "--train-size",
            "1000",
            "--selection-size",
            str(selection_size),
            "--holdout-size",
            "128",
            "--out-dir",
            str(datasets_dir),
        ],
        env=env,
    )
    return selection_path


def _evaluate_case(
    *,
    case: EvalCase,
    model: str,
    sft_adapter: Path | None,
    evo_checkpoint: Path,
    out_dir: Path,
    device: str,
    attn_implementation: str,
    force: bool,
    evo_selection_max_samples: int | None,
    evo_selection_max_new_tokens: int | None,
) -> Path:
    out_path = out_dir / f"{case.name}.json"
    if out_path.exists() and not force:
        return out_path
    env = os.environ.copy()
    env.setdefault("AGENT_MODE", "baseline")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_holdout.py"),
        "--holdout",
        str(case.holdout_path),
        "--model",
        model,
        "--device",
        device,
        "--attn-implementation",
        attn_implementation,
        "--evo-checkpoint",
        str(evo_checkpoint),
        "--evo-eval-routing",
        case.routing,
        "--output",
        str(out_path),
    ]
    if sft_adapter is not None:
        cmd.extend(["--sft-adapter", str(sft_adapter)])
    if case.selection_tasks_path is not None:
        cmd.extend(["--evo-selection-tasks", str(case.selection_tasks_path)])
        if evo_selection_max_samples is not None and int(evo_selection_max_samples) > 0:
            cmd.extend(["--evo-selection-max-samples", str(int(evo_selection_max_samples))])
        if evo_selection_max_new_tokens is not None and int(evo_selection_max_new_tokens) > 0:
            cmd.extend(["--evo-selection-max-new-tokens", str(int(evo_selection_max_new_tokens))])
    if case.routing_json_path is not None:
        cmd.extend(["--evo-routing-json", str(case.routing_json_path)])
    if case.max_samples is not None:
        cmd.extend(["--max-samples", str(int(case.max_samples))])
    if case.holdout_sampling is not None:
        cmd.extend(["--holdout-sampling", str(case.holdout_sampling)])
    if case.holdout_seed is not None:
        cmd.extend(["--holdout-seed", str(int(case.holdout_seed))])

    _run(cmd, env=env)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OOD evaluation suite for a run directory.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing checkpoint.pt (and optionally sft/peft_adapter).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write suite outputs (default: <run-dir>/ood_suite).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation (auto/cpu/mps/cuda).",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Attention implementation to pass through to evaluate_holdout.py.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per holdout (use with --holdout-sampling for smokes).",
    )
    parser.add_argument(
        "--holdout-sampling",
        type=str,
        choices=["head", "random", "stratified_family", "stratified_cell"],
        default="stratified_family",
        help="Sampling mode when --max-samples is set (default: stratified_family).",
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=9403,
        help="Seed for holdout sampling (default: 9403).",
    )
    parser.add_argument(
        "--paper-selection-seed",
        type=int,
        default=4242,
        help="Seed used to generate the paper-style selection set (default: 4242).",
    )
    parser.add_argument(
        "--paper-selection-size",
        type=int,
        default=192,
        help="Selection set size for paper-family routing (default: 192).",
    )
    parser.add_argument(
        "--evo-selection-max-samples",
        type=int,
        default=64,
        help="Cap selection-task evaluation for evo organelle picking (default: 64).",
    )
    parser.add_argument(
        "--evo-selection-max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens during selection generations (default: 64).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even if output JSON files already exist.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    sft_adapter = run_dir / "sft" / "peft_adapter"
    if not sft_adapter.exists():
        sft_adapter = None

    out_dir = args.output_dir if args.output_dir is not None else run_dir / "ood_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    id_holdout = run_dir / "datasets" / "holdout_tasks.jsonl"
    id_selection = run_dir / "datasets" / "selection_tasks.jsonl"
    final_holdout_json = run_dir / "final_holdout.json"
    routing_json = final_holdout_json if final_holdout_json.exists() else None

    paper_holdout = ROOT / "config" / "evaluation" / "paper_qwen3_holdout_v1.jsonl"
    paper_selection = _ensure_paper_selection_set(
        out_dir=out_dir,
        seed=int(args.paper_selection_seed),
        selection_size=int(args.paper_selection_size),
    )

    cases: list[EvalCase] = []
    if id_holdout.exists():
        cases.append(
            EvalCase(
                name="id_routed",
                holdout_path=id_holdout,
                routing="cell",
                selection_tasks_path=id_selection if id_selection.exists() else None,
                routing_json_path=routing_json,
                max_samples=args.max_samples,
                holdout_sampling=args.holdout_sampling if args.max_samples else None,
                holdout_seed=args.holdout_seed if args.max_samples else None,
            )
        )

    # OOD: reuse routing map (if any) vs allow re-selection on OOD tasks.
    cases.append(
        EvalCase(
            name="ood_paper_best_single",
            holdout_path=paper_holdout,
            routing="single",
            selection_tasks_path=id_selection if id_selection.exists() else None,
            max_samples=args.max_samples,
            holdout_sampling=args.holdout_sampling if args.max_samples else None,
            holdout_seed=args.holdout_seed if args.max_samples else None,
        )
    )
    cases.append(
        EvalCase(
            name="ood_paper_routed_reselect",
            holdout_path=paper_holdout,
            routing="family",
            selection_tasks_path=paper_selection,
            max_samples=args.max_samples,
            holdout_sampling=args.holdout_sampling if args.max_samples else None,
            holdout_seed=args.holdout_seed if args.max_samples else None,
        )
    )

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    suite_cases: list[dict[str, Any]] = []
    for case in cases:
        out_path = _evaluate_case(
            case=case,
            model=str(args.model),
            sft_adapter=sft_adapter,
            evo_checkpoint=checkpoint,
            out_dir=out_dir,
            device=str(args.device),
            attn_implementation=str(args.attn_implementation),
            force=bool(args.force),
            evo_selection_max_samples=int(args.evo_selection_max_samples),
            evo_selection_max_new_tokens=int(args.evo_selection_max_new_tokens),
        )
        payload = _load_json(out_path)
        acc = _extract_model_accuracy(payload)
        evo_acc = _pick_evo_accuracy(acc)
        suite_cases.append(
            {
                "name": case.name,
                "holdout": str(case.holdout_path),
                "output": str(out_path),
                "accuracy": {
                    "base": acc.get("base"),
                    "sft": acc.get("sft"),
                    "evolution": evo_acc,
                },
                "notes": (
                    "routed (final_holdout.json)"
                    if case.routing_json_path is not None and case.routing != "single"
                    else (
                        "routed (selection tasks)"
                        if case.selection_tasks_path is not None and case.routing != "single"
                        else ("single organelle (ID-selected)" if case.routing == "single" else "")
                    )
                ),
            }
        )

    summary = {
        "created_at": created_at,
        "run_dir": str(run_dir),
        "model": str(args.model),
        "cases": suite_cases,
    }
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_make_markdown(summary), encoding="utf-8")
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
