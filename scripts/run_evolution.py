#!/usr/bin/env python3
"""Long-form evolution runner with per-generation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers.utils import logging as hf_logging

from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    HostKernel,
    HumanBandit,
    PopulationManager,
    TelemetrySink,
    load_ecology_config,
)
from symbiont_ecology.economics.api import compute_route_cost
from symbiont_ecology.environment.grid import GridCellState, GridEnvironment, GridKey, GridTask
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.ledger import ATPAccount
from symbiont_ecology.evolution.population import Genome
from symbiont_ecology.metrics.telemetry import ComputeBudget
from symbiont_ecology.utils.checkpoint_io import load_checkpoint, save_checkpoint

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore", message="You are trying to modify a model with PEFT for a second time"
)
warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute")
warnings.filterwarnings(
    "ignore",
    message=r"Adapter .* was active which is now deleted. Setting active adapter to default.",
)

_LAST_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_LAST_INT_RE = re.compile(r"\b\d+\b")

_WORD_TO_INT: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _git_commit_short() -> str | None:
    try:
        import subprocess

        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or ""
        )
        return sha[:7] if sha else None
    except Exception:
        return None


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(int(seed))


def _parse_last_number(text: str) -> float | None:
    matches = list(_LAST_NUMBER_RE.finditer(text))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except Exception:
        return None


def _parse_last_int(text: str) -> int | None:
    matches = list(_LAST_INT_RE.finditer(text))
    if not matches:
        return None
    try:
        return int(matches[-1].group(0))
    except Exception:
        return None


def _normalize_code_answer(answer: str, *, multiline: bool) -> str:
    clean = answer.strip()
    if not clean:
        return ""
    if "```" in clean:
        parts = clean.split("```")
        if len(parts) >= 2:
            body = parts[1]
            lines = [line.rstrip() for line in body.splitlines()]
            if lines and lines[0].strip().lower().startswith("python"):
                lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
            while lines and not lines[-1].strip():
                lines = lines[:-1]
            clean = "\n".join(lines).strip()
    if multiline:
        lines = [line.rstrip() for line in clean.splitlines()]
        while lines and not lines[0].strip():
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        return "\n".join(lines).strip()
    if "\n" in clean:
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if lines:
            clean = lines[-1]
    return clean.strip().strip("'\"")


def _holdout_success(family: str, target: Any, answer: str) -> bool:
    family = str(family)
    known_families = {
        "math",
        "math.sequence",
        "math.multi_step",
        "string.sort",
        "word.count",
        "json_repair",
        "logic.bool",
        "code.format",
        "regex",
        "regex.synthesis",
        "regex.debugging",
        "regex.refactoring",
        "regex.recognition",
        "regex.explanation",
        "regex.mutation_effect",
    }
    if family not in known_families:
        return answer.strip() == str(target).strip()

    try:
        task = GridTask(
            task_id="holdout",
            cell=(family, "short"),
            prompt="",
            price=0.0,
            target=target,
            family=family,
            depth="short",
            difficulty=0.0,
        )
        success, _reward = task.evaluate(answer)
        return bool(success)
    except Exception:
        return False


def _select_best_organelle_for_cell(
    population: PopulationManager, cell: tuple[str, str], candidates: list[str]
) -> tuple[str, float]:
    # Select based on global task reward (competence) to find the most accurate model.
    # We ignore cell-specific ROI because it includes energy penalties that disadvantage high-rank models.
    scored = [(oid, float(population.average_task_reward(oid, limit=10))) for oid in candidates]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    if scored:
        return scored[0][0], scored[0][1]
    return candidates[0], 0.0


def _format_holdout_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Final Holdout Evaluation")
    lines.append("")
    commit = payload.get("git_commit")
    if commit:
        lines.append(f"- git_commit: `{commit}`")
    lines.append(f"- tasks: `{payload.get('holdout_tasks_path')}` (n={payload.get('sample_size')})")
    lines.append(f"- selection_mode: `{payload.get('selection_mode')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- accuracy: {payload.get('accuracy', 0.0):.3f} ({payload.get('correct')}/{payload.get('total')})"
    )
    lines.append(f"- avg_cost: {payload.get('avg_cost', 0.0):.3f}")
    lines.append(f"- avg_latency_ms: {payload.get('avg_latency_ms', 0.0):.1f}")
    lines.append(f"- avg_tokens: {payload.get('avg_tokens', 0.0):.1f}")
    lines.append(f"- cost_per_correct: {payload.get('cost_per_correct', 0.0):.3f}")
    lines.append("")
    lines.append("## By Family")
    lines.append("")
    lines.append("| family | accuracy | correct | total | avg_cost | avg_latency_ms | avg_tokens |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    family = payload.get("family_breakdown") or {}
    if isinstance(family, dict):
        for fam, stats in family.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(fam),
                        f"{float(stats.get('accuracy', 0.0)):.3f}",
                        str(int(stats.get("correct", 0) or 0)),
                        str(int(stats.get("total", 0) or 0)),
                        f"{float(stats.get('avg_cost', 0.0)):.3f}",
                        f"{float(stats.get('avg_latency_ms', 0.0)):.1f}",
                        f"{float(stats.get('avg_tokens', 0.0)):.1f}",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def _maybe_plot_holdout(payload: dict[str, Any], output_root: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    family = payload.get("family_breakdown") or {}
    if not isinstance(family, dict) or not family:
        return

    names = list(family.keys())
    acc = [float(family[n].get("accuracy", 0.0)) for n in names]
    cost = [float(family[n].get("avg_cost", 0.0)) for n in names]

    visuals = output_root / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.bar(names, acc, color="#2563eb")
    plt.ylim(0.0, 1.0)
    plt.ylabel("accuracy")
    plt.title("Holdout accuracy by family")
    plt.tight_layout()
    plt.savefig(visuals / "final_holdout_accuracy_by_family.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.bar(names, cost, color="#6b7280")
    plt.ylabel("avg_cost")
    plt.title("Holdout cost by family")
    plt.tight_layout()
    plt.savefig(visuals / "final_holdout_cost_by_family.png", dpi=200)
    plt.close()


def _run_final_holdout(
    *,
    holdout_path: Path,
    holdout_sample_size: int | None,
    config_path: Path,
    output_root: Path,
    host: HostKernel,
    population: PopulationManager,
) -> dict[str, Any]:
    tasks = _load_jsonl(holdout_path)
    if not tasks:
        raise ValueError(f"Holdout tasks file is empty: {holdout_path}")
    if holdout_sample_size is not None and 0 < holdout_sample_size < len(tasks):
        rng = random.Random(9403)
        tasks = rng.sample(tasks, int(holdout_sample_size))

    organelle_ids = [oid for oid in host.list_organelle_ids() if oid in host.organelles]
    if not organelle_ids:
        raise RuntimeError("No organelles available for holdout evaluation")

    cells = sorted({(str(t["family"]), str(t["depth"])) for t in tasks})
    selection: dict[str, dict[str, Any]] = {}
    for family, depth in cells:
        oid, score = _select_best_organelle_for_cell(population, (family, depth), organelle_ids)
        selection[f"{family}:{depth}"] = {"organelle_id": oid, "cell_value": score}

    total = 0
    correct = 0
    cost_sum = 0.0
    latency_sum = 0.0
    tokens_sum = 0.0
    family_stats: dict[str, dict[str, float]] = {}

    for idx, item in enumerate(tasks, start=1):
        prompt = str(item["prompt"])
        target = item["target"]
        family = str(item["family"])
        depth = str(item["depth"])
        key = f"{family}:{depth}"
        chosen = str(selection.get(key, {}).get("organelle_id", organelle_ids[0]))

        result = host.step(
            prompt=prompt,
            intent="final holdout",
            max_routes=1,
            allowed_organelle_ids=[chosen],
        )
        metrics = result.responses.get(chosen)
        if metrics is None:
            continue
        success = _holdout_success(family, target, metrics.answer)
        total += 1
        if success:
            correct += 1
        cost = float(compute_route_cost(host.config.energy, metrics).total_cost)
        cost_sum += cost
        latency_sum += float(metrics.latency_ms)
        tokens_sum += float(metrics.tokens)

        fam = family_stats.setdefault(
            family,
            {"correct": 0.0, "total": 0.0, "cost_sum": 0.0, "lat_sum": 0.0, "tok_sum": 0.0},
        )
        fam["total"] += 1.0
        fam["cost_sum"] += cost
        fam["lat_sum"] += float(metrics.latency_ms)
        fam["tok_sum"] += float(metrics.tokens)
        if success:
            fam["correct"] += 1.0

        if idx % 25 == 0:
            print(f"[final-holdout] {idx}/{len(tasks)} tasks", flush=True)

    accuracy = (correct / total) if total else 0.0
    avg_cost = (cost_sum / total) if total else 0.0
    avg_latency_ms = (latency_sum / total) if total else 0.0
    avg_tokens = (tokens_sum / total) if total else 0.0
    family_breakdown: dict[str, dict[str, float]] = {}
    for fam, stats in sorted(family_stats.items(), key=lambda kv: kv[0]):
        denom = max(1.0, stats["total"])
        family_breakdown[fam] = {
            "accuracy": stats["correct"] / denom,
            "correct": float(stats["correct"]),
            "total": float(stats["total"]),
            "avg_cost": stats["cost_sum"] / denom,
            "avg_latency_ms": stats["lat_sum"] / denom,
            "avg_tokens": stats["tok_sum"] / denom,
        }

    payload: dict[str, Any] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit_short(),
        "config_path": str(config_path),
        "holdout_tasks_path": str(holdout_path),
        "sample_size": int(len(tasks)),
        "selection_mode": "best_per_cell",
        "selection": selection,
        "accuracy": accuracy,
        "correct": int(correct),
        "total": int(total),
        "avg_cost": avg_cost,
        "avg_latency_ms": avg_latency_ms,
        "avg_tokens": avg_tokens,
        "cost_sum": cost_sum,
        "cost_per_correct": (cost_sum / max(1, correct)),
        "family_breakdown": family_breakdown,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "final_holdout.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "final_holdout.md").write_text(
        _format_holdout_markdown(payload), encoding="utf-8"
    )
    _maybe_plot_holdout(payload, output_root)
    return payload


def summarize_slice(episodes_jsonl: Path, start_idx: int, end_idx: int) -> dict:
    totals, task_rewards, costs = [], [], []
    slice_tokens = 0
    slice_forward_passes = 0
    slice_hebbian_updates = 0
    with episodes_jsonl.open() as f:
        for i, line in enumerate(f):
            if i < start_idx or i >= end_idx:
                continue
            obj = json.loads(line)
            if obj.get("type") != "episode":
                continue
            rb = obj["rewards"]
            total = (
                rb["task_reward"]
                + rb["novelty_bonus"]
                + rb["competence_bonus"]
                + rb["helper_bonus"]
                - rb["risk_penalty"]
                - rb["cost_penalty"]
            )
            totals.append(total)
            task_rewards.append(rb["task_reward"])
            costs.append(rb["cost_penalty"])
            # Extract compute metrics from observations
            obs = obj.get("observations", {})
            metrics = obs.get("metrics", {})
            # Tokens can be in metrics.tokens or directly in observations
            tokens = (
                metrics.get("tokens", 0) or obs.get("tokens", 0) or obs.get("prompt_tokens", 0) or 0
            )
            slice_tokens += int(tokens)
            # Each episode = 1 forward pass + 1 hebbian update per organelle
            num_organelles = len(obj.get("organelles", []))
            slice_forward_passes += max(1, num_organelles)
            slice_hebbian_updates += max(1, num_organelles)
    return {
        "episodes": end_idx - start_idx,
        "avg_total": float(mean(totals)) if totals else 0.0,
        "avg_task_reward": float(mean(task_rewards)) if task_rewards else 0.0,
        "avg_cost_penalty": float(mean(costs)) if costs else 0.0,
        "slice_tokens": slice_tokens,
        "slice_forward_passes": slice_forward_passes,
        "slice_hebbian_updates": slice_hebbian_updates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-form evolution on a frozen host model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "ecology.yaml",
        help="Path to ecology YAML config.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=12,
        help="Number of generations to simulate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts_qwen3_long_eval"),
        help="Directory for telemetry outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed for grid environment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override synthetic batch size per generation.",
    )
    parser.add_argument(
        "--disable-human",
        action="store_true",
        help="Disable human bandit feedback for deterministic runs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional run directory containing a checkpoint.pt to resume from.",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow trusted legacy pickle checkpoints when resuming. "
            "Unsafe: untrusted pickle files can execute arbitrary code."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, write checkpoint.pt every N generations (1-indexed).",
    )
    parser.add_argument(
        "--final-holdout",
        type=Path,
        default=None,
        help="Optional JSONL holdout tasks to evaluate after the run (measurement-only).",
    )
    parser.add_argument(
        "--final-holdout-sample-size",
        type=int,
        default=None,
        help="Optional sample size for final holdout tasks (default: all).",
    )
    return parser.parse_args()


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _to_grid_key(value: object) -> GridKey | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (str(value[0]), str(value[1]))
    return None


def _serialize_grid_task(task: GridTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "cell": list(task.cell),
        "prompt": task.prompt,
        "price": float(task.price),
        "target": task.target,
        "family": task.family,
        "depth": task.depth,
        "difficulty": float(task.difficulty),
        "canary": bool(task.canary),
        "reward_bonus": float(task.reward_bonus),
        "failure_cost_scale": float(task.failure_cost_scale),
    }


def _deserialize_grid_task(payload: object) -> GridTask | None:
    if not isinstance(payload, dict):
        return None
    cell = _to_grid_key(payload.get("cell"))
    if cell is None:
        return None
    return GridTask(
        task_id=str(payload.get("task_id", "")),
        cell=cell,
        prompt=str(payload.get("prompt", "")),
        price=_to_float(payload.get("price", 0.0), 0.0),
        target=payload.get("target"),
        family=str(payload.get("family", cell[0])),
        depth=str(payload.get("depth", cell[1])),
        difficulty=_to_float(payload.get("difficulty", 0.0), 0.0),
        canary=bool(payload.get("canary", False)),
        reward_bonus=_to_float(payload.get("reward_bonus", 0.0), 0.0),
        failure_cost_scale=_to_float(payload.get("failure_cost_scale", 1.0), 1.0),
    )


def _serialize_population(population: PopulationManager) -> dict[str, object]:
    genomes: dict[str, dict[str, object]] = {}
    for oid, genome in population.population.items():
        genomes[oid] = {
            "organelle_id": genome.organelle_id,
            "drive_weights": dict(genome.drive_weights),
            "gate_bias": float(genome.gate_bias),
            "rank": int(genome.rank),
            "explore_rate": float(genome.explore_rate),
            "post_rate": float(genome.post_rate),
            "read_rate": float(genome.read_rate),
            "hint_weight": float(genome.hint_weight),
            "beta_exploit": float(genome.beta_exploit),
            "q_decay": float(genome.q_decay),
            "ucb_bonus": float(genome.ucb_bonus),
            "budget_aggressiveness": float(genome.budget_aggressiveness),
            "rank_noise": dict(genome.rank_noise),
            "adapter_dropout": sorted(genome.adapter_dropout),
            "duplication_factors": dict(genome.duplication_factors),
        }
    assimilation_history: list[dict[str, object]] = []
    for key, records in population.assimilation_history.items():
        oid, family, depth = key
        assimilation_history.append(
            {
                "organelle_id": oid,
                "family": family,
                "depth": depth,
                "records": list(records),
            }
        )
    cell_values: list[dict[str, object]] = []
    for oid, per_cell in population.cell_values.items():
        for cell, value in per_cell.items():
            cell_values.append({"organelle_id": oid, "cell": list(cell), "value": float(value)})
    cell_counts: list[dict[str, object]] = []
    for oid, per_cell in population.cell_counts.items():
        for cell, count in per_cell.items():
            cell_counts.append({"organelle_id": oid, "cell": list(cell), "count": int(count)})
    global_cell_counts = [
        {"cell": list(cell), "count": int(count)}
        for cell, count in population.global_cell_counts.items()
    ]
    return {
        "genomes": genomes,
        "history": {k: list(v) for k, v in population.history.items()},
        "score_meta": {k: list(v) for k, v in population.score_meta.items()},
        "ages": dict(population.ages),
        "energy": {k: list(v) for k, v in population.energy.items()},
        "roi": {k: list(v) for k, v in population.roi.items()},
        "adapter_usage": {k: dict(v) for k, v in population.adapter_usage.items()},
        "energy_delta": {k: list(v) for k, v in population.energy_delta.items()},
        "evidence_credit": dict(population.evidence_credit),
        "assimilation_history_limit": population.assimilation_history_limit,
        "assimilation_history": assimilation_history,
        "cell_values": cell_values,
        "cell_counts": cell_counts,
        "global_cell_counts": global_cell_counts,
    }


def _restore_population(state_obj: object, population: PopulationManager) -> PopulationManager:
    if not isinstance(state_obj, dict):
        return population
    genomes_obj = state_obj.get("genomes")
    if not isinstance(genomes_obj, dict):
        return population

    restored = PopulationManager(population.config, population.foraging)
    for oid, genome_obj in genomes_obj.items():
        if not isinstance(oid, str) or not isinstance(genome_obj, dict):
            continue
        drive_raw = genome_obj.get("drive_weights", {})
        drive_weights: dict[str, float] = {}
        if isinstance(drive_raw, dict):
            for key, value in drive_raw.items():
                if isinstance(key, str):
                    drive_weights[key] = _to_float(value, 0.0)
        rank_noise_raw = genome_obj.get("rank_noise", {})
        if not isinstance(rank_noise_raw, dict):
            rank_noise_raw = {}
        adapter_dropout_raw = genome_obj.get("adapter_dropout", [])
        if not isinstance(adapter_dropout_raw, list):
            adapter_dropout_raw = []
        duplication_raw = genome_obj.get("duplication_factors", {})
        if not isinstance(duplication_raw, dict):
            duplication_raw = {}
        genome = Genome(
            organelle_id=str(genome_obj.get("organelle_id", oid)),
            drive_weights=drive_weights,
            gate_bias=_to_float(genome_obj.get("gate_bias", 0.0), 0.0),
            rank=max(1, _to_int(genome_obj.get("rank", 1), 1)),
            explore_rate=_to_float(genome_obj.get("explore_rate", 0.5), 0.5),
            post_rate=_to_float(genome_obj.get("post_rate", 0.0), 0.0),
            read_rate=_to_float(genome_obj.get("read_rate", 0.0), 0.0),
            hint_weight=_to_float(genome_obj.get("hint_weight", 0.0), 0.0),
            beta_exploit=_to_float(genome_obj.get("beta_exploit", 1.5), 1.5),
            q_decay=_to_float(genome_obj.get("q_decay", 0.3), 0.3),
            ucb_bonus=_to_float(genome_obj.get("ucb_bonus", 0.2), 0.2),
            budget_aggressiveness=_to_float(genome_obj.get("budget_aggressiveness", 0.5), 0.5),
            rank_noise={
                str(key): _to_float(value, 0.0)
                for key, value in rank_noise_raw.items()
                if isinstance(key, str)
            },
            adapter_dropout={str(item) for item in adapter_dropout_raw if isinstance(item, str)},
            duplication_factors={
                str(key): _to_float(value, 0.0)
                for key, value in duplication_raw.items()
                if isinstance(key, str)
            },
        )
        restored.register(genome)

    history_obj = state_obj.get("history")
    if isinstance(history_obj, dict):
        restored.history = {
            str(k): [_to_float(v, 0.0) for v in list(values)]
            for k, values in history_obj.items()
            if isinstance(k, str) and isinstance(values, list)
        }
    score_meta_obj = state_obj.get("score_meta")
    if isinstance(score_meta_obj, dict):
        restored.score_meta = {
            str(k): [dict(item) for item in values if isinstance(item, dict)]
            for k, values in score_meta_obj.items()
            if isinstance(k, str) and isinstance(values, list)
        }
    ages_obj = state_obj.get("ages")
    if isinstance(ages_obj, dict):
        restored.ages = {
            str(k): max(0, _to_int(v, 0)) for k, v in ages_obj.items() if isinstance(k, str)
        }
    energy_obj = state_obj.get("energy")
    if isinstance(energy_obj, dict):
        restored.energy = {
            str(k): [_to_float(v, 0.0) for v in list(values)]
            for k, values in energy_obj.items()
            if isinstance(k, str) and isinstance(values, list)
        }
    roi_obj = state_obj.get("roi")
    if isinstance(roi_obj, dict):
        restored.roi = {
            str(k): [_to_float(v, 0.0) for v in list(values)]
            for k, values in roi_obj.items()
            if isinstance(k, str) and isinstance(values, list)
        }
    adapter_usage_obj = state_obj.get("adapter_usage")
    if isinstance(adapter_usage_obj, dict):
        restored.adapter_usage = {}
        for oid, per_module in adapter_usage_obj.items():
            if not isinstance(oid, str) or not isinstance(per_module, dict):
                continue
            clean_per_module: dict[str, list[float]] = {}
            for module_name, values in per_module.items():
                if not isinstance(module_name, str) or not isinstance(values, list):
                    continue
                clean_per_module[module_name] = [_to_float(v, 0.0) for v in values]
            restored.adapter_usage[oid] = clean_per_module
    energy_delta_obj = state_obj.get("energy_delta")
    if isinstance(energy_delta_obj, dict):
        restored.energy_delta = {
            str(k): [_to_float(v, 0.0) for v in list(values)]
            for k, values in energy_delta_obj.items()
            if isinstance(k, str) and isinstance(values, list)
        }
    evidence_credit_obj = state_obj.get("evidence_credit")
    if isinstance(evidence_credit_obj, dict):
        restored.evidence_credit = {
            str(k): max(0, _to_int(v, 0))
            for k, v in evidence_credit_obj.items()
            if isinstance(k, str)
        }
    limit_obj = state_obj.get("assimilation_history_limit")
    if isinstance(limit_obj, int):
        restored.assimilation_history_limit = limit_obj

    restored.assimilation_history = {}
    assimilation_history_obj = state_obj.get("assimilation_history")
    if isinstance(assimilation_history_obj, list):
        for item in assimilation_history_obj:
            if not isinstance(item, dict):
                continue
            oid = str(item.get("organelle_id", ""))
            family = str(item.get("family", ""))
            depth = str(item.get("depth", ""))
            if not oid or not family or not depth:
                continue
            records_raw = item.get("records", [])
            if isinstance(records_raw, list):
                restored.assimilation_history[(oid, family, depth)] = [
                    dict(record) for record in records_raw if isinstance(record, dict)
                ]

    restored.cell_values = {}
    cell_values_obj = state_obj.get("cell_values")
    if isinstance(cell_values_obj, list):
        for item in cell_values_obj:
            if not isinstance(item, dict):
                continue
            oid = str(item.get("organelle_id", ""))
            cell = _to_grid_key(item.get("cell"))
            if not oid or cell is None:
                continue
            restored.cell_values.setdefault(oid, {})[cell] = _to_float(item.get("value", 0.0), 0.0)

    restored.cell_counts = {}
    cell_counts_obj = state_obj.get("cell_counts")
    if isinstance(cell_counts_obj, list):
        for item in cell_counts_obj:
            if not isinstance(item, dict):
                continue
            oid = str(item.get("organelle_id", ""))
            cell = _to_grid_key(item.get("cell"))
            if not oid or cell is None:
                continue
            restored.cell_counts.setdefault(oid, {})[cell] = max(
                0, _to_int(item.get("count", 0), 0)
            )

    restored.global_cell_counts = {}
    global_cell_counts_obj = state_obj.get("global_cell_counts")
    if isinstance(global_cell_counts_obj, list):
        for item in global_cell_counts_obj:
            if not isinstance(item, dict):
                continue
            cell = _to_grid_key(item.get("cell"))
            if cell is None:
                continue
            restored.global_cell_counts[cell] = max(0, _to_int(item.get("count", 0), 0))

    return restored


def _serialize_ledger(ledger: ATPLedger) -> dict[str, object]:
    return {
        "accounts": {
            organelle_id: _to_float(account.balance, 0.0)
            for organelle_id, account in ledger.accounts.items()
        },
        "energy_accounts": {
            organelle_id: _to_float(balance, 0.0)
            for organelle_id, balance in ledger.energy_accounts.items()
        },
        "energy_cap": _to_float(ledger.energy_cap, 5.0),
    }


def _restore_ledger(state_obj: object, ledger: ATPLedger) -> ATPLedger:
    if not isinstance(state_obj, dict):
        return ledger
    restored = ATPLedger()
    restored.energy_cap = _to_float(
        state_obj.get("energy_cap", ledger.energy_cap), ledger.energy_cap
    )
    accounts_obj = state_obj.get("accounts")
    if isinstance(accounts_obj, dict):
        restored.accounts = {
            organelle_id: ATPAccount(_to_float(balance, 0.0))
            for organelle_id, balance in accounts_obj.items()
            if isinstance(organelle_id, str)
        }
    energy_obj = state_obj.get("energy_accounts")
    if isinstance(energy_obj, dict):
        restored.energy_accounts = {
            organelle_id: max(0.0, _to_float(balance, 0.0))
            for organelle_id, balance in energy_obj.items()
            if isinstance(organelle_id, str)
        }
    return restored


def _serialize_environment_state(environment: GridEnvironment) -> dict[str, object]:
    controller_cells: dict[tuple[str, str], dict[str, object]] = {}
    for cell, state in environment.controller.cells.items():
        controller_cells[cell] = {
            "difficulty": _to_float(state.difficulty, 0.5),
            "success_ema": _to_float(state.success_ema, 0.5),
            "price": _to_float(state.price, 1.0),
            "canary_index": max(0, _to_int(state.canary_index, 0)),
            "canaries": [_serialize_grid_task(task) for task in state.canaries],
        }
    organism_stats: dict[str, list[dict[str, object]]] = {}
    for organelle_id, per_cell in environment.organism_stats.items():
        entries: list[dict[str, object]] = []
        for cell, value in per_cell.items():
            entries.append({"cell": list(cell), "value": _to_float(value, 0.0)})
        organism_stats[organelle_id] = entries
    return {
        "controller": {
            "cells": controller_cells,
            "bandit_counts": dict(environment.controller.bandit_counts),
            "bandit_success": dict(environment.controller.bandit_success),
            "lp_progress": dict(environment.controller.lp_progress),
            "lp_prev_ema": dict(environment.controller.lp_prev_ema),
            "bandit_c": _to_float(environment.controller.bandit_c, 1.2),
            "lp_alpha": _to_float(environment.controller.lp_alpha, 0.5),
            "tau": _to_float(environment.controller.ctrl.tau, 0.5),
            "beta": _to_float(environment.controller.ctrl.beta, 0.5),
            "eta": _to_float(environment.controller.ctrl.eta, 0.1),
            "price_base": _to_float(environment.controller.pricing.base, 1.0),
            "price_k": _to_float(environment.controller.pricing.k, 1.0),
            "price_min": _to_float(environment.controller.pricing.min, 0.0),
            "price_max": _to_float(environment.controller.pricing.max, 1.0),
            "rng_state": environment.controller.rng.getstate(),
        },
        "organism_stats": organism_stats,
        "organism_canary_fail": dict(environment.organism_canary_fail),
        "rng_state": environment.rng.getstate(),
        "task_counter": max(0, _to_int(environment.task_counter, 0)),
    }


def _restore_environment_state(environment: GridEnvironment, state_obj: object) -> None:
    if not isinstance(state_obj, dict):
        return

    controller_obj = state_obj.get("controller")
    if isinstance(controller_obj, dict):
        environment.controller.bandit_c = _to_float(
            controller_obj.get("bandit_c", environment.controller.bandit_c),
            environment.controller.bandit_c,
        )
        environment.controller.lp_alpha = _to_float(
            controller_obj.get("lp_alpha", environment.controller.lp_alpha),
            environment.controller.lp_alpha,
        )
        environment.controller.ctrl.tau = _to_float(
            controller_obj.get("tau", environment.controller.ctrl.tau),
            environment.controller.ctrl.tau,
        )
        environment.controller.ctrl.beta = _to_float(
            controller_obj.get("beta", environment.controller.ctrl.beta),
            environment.controller.ctrl.beta,
        )
        environment.controller.ctrl.eta = _to_float(
            controller_obj.get("eta", environment.controller.ctrl.eta),
            environment.controller.ctrl.eta,
        )
        environment.controller.pricing.base = _to_float(
            controller_obj.get("price_base", environment.controller.pricing.base),
            environment.controller.pricing.base,
        )
        environment.controller.pricing.k = _to_float(
            controller_obj.get("price_k", environment.controller.pricing.k),
            environment.controller.pricing.k,
        )
        environment.controller.pricing.min = _to_float(
            controller_obj.get("price_min", environment.controller.pricing.min),
            environment.controller.pricing.min,
        )
        environment.controller.pricing.max = _to_float(
            controller_obj.get("price_max", environment.controller.pricing.max),
            environment.controller.pricing.max,
        )

        cells_obj = controller_obj.get("cells")
        if isinstance(cells_obj, dict):
            for raw_cell, raw_state in cells_obj.items():
                cell = _to_grid_key(raw_cell)
                if cell is None or cell not in environment.controller.cells:
                    continue
                if not isinstance(raw_state, dict):
                    continue
                canaries: list[GridTask] = []
                canaries_obj = raw_state.get("canaries")
                if isinstance(canaries_obj, list):
                    for task_payload in canaries_obj:
                        task = _deserialize_grid_task(task_payload)
                        if task is not None:
                            canaries.append(task)
                environment.controller.cells[cell] = GridCellState(
                    difficulty=_to_float(raw_state.get("difficulty", 0.5), 0.5),
                    success_ema=_to_float(raw_state.get("success_ema", 0.5), 0.5),
                    price=_to_float(raw_state.get("price", 1.0), 1.0),
                    canary_index=max(0, _to_int(raw_state.get("canary_index", 0), 0)),
                    canaries=canaries,
                )

        for map_name in ("bandit_counts", "bandit_success", "lp_progress", "lp_prev_ema"):
            raw_map = controller_obj.get(map_name)
            if not isinstance(raw_map, dict):
                continue
            target = getattr(environment.controller, map_name)
            target.clear()
            for raw_cell, value in raw_map.items():
                cell = _to_grid_key(raw_cell)
                if cell is None:
                    continue
                if map_name == "bandit_counts":
                    target[cell] = max(0, _to_int(value, 0))
                else:
                    target[cell] = _to_float(value, 0.0)

        controller_rng_state = controller_obj.get("rng_state")
        if controller_rng_state is not None:
            try:
                environment.controller.rng.setstate(controller_rng_state)
            except Exception:
                pass

    organism_stats_obj = state_obj.get("organism_stats")
    if isinstance(organism_stats_obj, dict):
        restored_stats: dict[str, dict[GridKey, float]] = {}
        for organelle_id, entries in organism_stats_obj.items():
            if not isinstance(organelle_id, str) or not isinstance(entries, list):
                continue
            per_cell: dict[GridKey, float] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                cell = _to_grid_key(entry.get("cell"))
                if cell is None:
                    continue
                per_cell[cell] = _to_float(entry.get("value", 0.0), 0.0)
            restored_stats[organelle_id] = per_cell
        environment.organism_stats = restored_stats

    organism_canary_obj = state_obj.get("organism_canary_fail")
    if isinstance(organism_canary_obj, dict):
        environment.organism_canary_fail = {
            str(key): bool(value)
            for key, value in organism_canary_obj.items()
            if isinstance(key, str)
        }

    environment.task_counter = max(0, _to_int(state_obj.get("task_counter", 0), 0))
    env_rng_state = state_obj.get("rng_state")
    if env_rng_state is not None:
        try:
            environment.rng.setstate(env_rng_state)
        except Exception:
            pass


def _load_checkpoint(path: Path, *, allow_unsafe_pickle: bool = False) -> dict[str, Any]:
    return load_checkpoint(path, allow_unsafe_pickle=allow_unsafe_pickle)


def _save_checkpoint(
    path: Path,
    generation: int,
    host: HostKernel,
    population: PopulationManager,
    environment: GridEnvironment,
    ledger: ATPLedger,
    loop: EcologyLoop,
    random_state: tuple,
    telemetry_bytes: dict[str, int] | None = None,
    compute_budget: ComputeBudget | None = None,
) -> None:
    adapter_states: dict[str, object] = {}
    export_errors: list[str] = []
    for oid, org in host.organelles.items():
        try:
            adapter_states[oid] = org.export_adapter_state()  # type: ignore[attr-defined]
        except Exception as exc:
            export_errors.append(f"{oid}: {exc}")
    if export_errors:
        joined = "; ".join(export_errors)
        raise RuntimeError(f"Failed to export adapter state for checkpoint save: {joined}")
    state = {
        "checkpoint_version": 2,
        "generation": generation,
        "population_state": _serialize_population(population),
        "environment_state": _serialize_environment_state(environment),
        "ledger_state": _serialize_ledger(ledger),
        "adapter_states": adapter_states,
        "random_state": random_state,
        "assimilation_cooldown": loop.assimilation_cooldown,
        "tau_relief": getattr(loop, "_tau_relief", {}),
        "telemetry_bytes": dict(telemetry_bytes or {}),
        "compute_budget": compute_budget.model_dump() if compute_budget else None,
    }
    save_checkpoint(path, state)


def _truncate_file_to_bytes(path: Path, size_bytes: int | None) -> None:
    if size_bytes is None:
        return
    try:
        expected = int(size_bytes)
    except Exception:
        return
    if expected < 0:
        return
    if not path.exists():
        return
    try:
        current = path.stat().st_size
    except Exception:
        return
    if current <= expected:
        return
    try:
        with path.open("r+b") as handle:
            handle.truncate(expected)
    except Exception:
        return


def _make_assimilation_tester(config) -> AssimilationTester:
    assim = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )
    try:
        tuning = config.assimilation_tuning
        assim.bootstrap_enabled = bool(getattr(tuning, "bootstrap_uplift_enabled", False))
        assim.bootstrap_n = int(getattr(tuning, "bootstrap_samples", 0))
        assim.permutation_n = int(getattr(tuning, "permutation_samples", 0))
        assim.min_samples = int(getattr(tuning, "min_uplift_samples", 2))
        assim.dr_enabled = bool(getattr(tuning, "dr_enabled", False))
        assim.dr_strata = list(getattr(tuning, "dr_strata", assim.dr_strata))
        assim.dr_min_stratum = int(getattr(tuning, "dr_min_stratum_size", assim.dr_min_stratum))
        assim.dr_min_power = float(getattr(tuning, "dr_min_power", assim.dr_min_power))
    except Exception:
        pass
    return assim


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    resume_root = args.resume_from
    if resume_root is not None:
        args.output = resume_root

    config = load_ecology_config(args.config)
    config.metrics.root = args.output
    config.metrics.root.mkdir(parents=True, exist_ok=True)
    if args.batch_size is not None:
        config.environment.synthetic_batch_size = args.batch_size

    # Respect model IDs provided by the experiment config; only default if unset
    if not getattr(config.host, "backbone_model", None):
        config.host.backbone_model = "Qwen/Qwen3-0.6B"
    if not getattr(config.host, "tokenizer", None):
        config.host.tokenizer = config.host.backbone_model

    if args.device is not None:
        config.host.device = args.device
    elif config.host.device == "cpu":
        config.host.device = "auto"

    compute_budget = ComputeBudget()
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger, compute_budget=compute_budget)
    host.freeze_host()

    population = PopulationManager(config.evolution, config.foraging)

    start_generation = 0
    prev_n = 0
    gen_summaries: list[dict[str, object]] = []
    gen_summaries_path = config.metrics.root / "gen_summaries.jsonl"
    episodes_path = config.metrics.root / config.metrics.episodes_file
    assimilation_path = config.metrics.root / config.metrics.assimilation_file
    if resume_root is None and gen_summaries_path.exists():
        gen_summaries_path.write_text("", encoding="utf-8")

    checkpoint_path = config.metrics.root / "checkpoint.pt"
    if resume_root is not None and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path, allow_unsafe_pickle=args.allow_unsafe_pickle)
        start_generation = int(state.get("generation", 0))
        telemetry_bytes = state.get("telemetry_bytes") if isinstance(state, dict) else None
        if isinstance(telemetry_bytes, dict):
            _truncate_file_to_bytes(episodes_path, telemetry_bytes.get("episodes"))
            _truncate_file_to_bytes(assimilation_path, telemetry_bytes.get("assimilation"))
            _truncate_file_to_bytes(gen_summaries_path, telemetry_bytes.get("gen_summaries"))
        # restore population
        population_state = state.get("population_state")
        if population_state is not None:
            population = _restore_population(population_state, population)
        else:
            population = state.get("population", population)
        # respawn organelles with saved IDs and adapter states
        adapter_states = state.get("adapter_states", {}) or {}
        population.population = dict(population.population)
        restore_errors: list[str] = []
        for genome in population.population.values():
            oid = genome.organelle_id
            host.spawn_organelle(rank=genome.rank, organelle_id=oid)
            if oid in adapter_states:
                state_obj = adapter_states[oid]
                try:
                    if isinstance(state_obj, (str, Path)):
                        host.import_organelle_adapter(oid, state_obj)
                    else:
                        org = host.organelles.get(oid)
                        if org is not None:
                            org.import_adapter_state(state_obj, alpha=1.0)  # type: ignore[attr-defined]
                except Exception as exc:
                    restore_errors.append(f"{oid}: {exc}")
            ledger.ensure(oid, 0.0)
            ledger.ensure_energy(oid, 0.0)
        if restore_errors:
            joined = "; ".join(restore_errors)
            raise RuntimeError(f"Failed to restore one or more organelle adapters: {joined}")
        # restore ledger balances
        if state.get("ledger_state") is not None:
            ledger = _restore_ledger(state.get("ledger_state"), ledger)
            host.ledger = ledger
        else:
            saved_ledger: ATPLedger = state.get("ledger", ledger)
            ledger.accounts = saved_ledger.accounts
            ledger.energy_accounts = saved_ledger.energy_accounts
            ledger.energy_cap = saved_ledger.energy_cap
        # restore compute budget
        saved_compute = state.get("compute_budget")
        if saved_compute is not None:
            compute_budget = ComputeBudget(**saved_compute)
            host.compute_budget = compute_budget
        # environment state
        env_state = state.get("environment_state", {})
        environment = GridEnvironment(
            grid_cfg=config.grid,
            controller_cfg=config.controller,
            pricing_cfg=config.pricing,
            canary_cfg=config.canary,
            seed=args.seed,
            reward_bonus=config.environment.success_reward_bonus,
            failure_cost_multiplier=config.environment.failure_cost_multiplier,
            lp_alpha=getattr(config.curriculum, "lp_alpha", 0.5),
        )
        if "checkpoint_version" in state:
            _restore_environment_state(environment, env_state)
        else:
            try:
                environment.controller = env_state.get("controller", environment.controller)
                environment.organism_stats = env_state.get(
                    "organism_stats", environment.organism_stats
                )
                environment.organism_canary_fail = env_state.get(
                    "organism_canary_fail", environment.organism_canary_fail
                )
                environment.task_counter = env_state.get("task_counter", environment.task_counter)
                rng_state = env_state.get("rng_state")
                if rng_state is not None:
                    environment.rng.setstate(rng_state)
            except Exception:
                pass
        assim = _make_assimilation_tester(config)
        loop = EcologyLoop(
            config=config,
            host=host,
            environment=environment,
            population=population,
            assimilation=assim,
            human_bandit=None,
            sink=TelemetrySink(
                root=config.metrics.root,
                episodes_file=config.metrics.episodes_file,
                assimilation_file=config.metrics.assimilation_file,
            ),
        )
        try:
            loop.assimilation_cooldown = state.get("assimilation_cooldown", {})
            if hasattr(loop, "_tau_relief"):
                loop._tau_relief = state.get("tau_relief", {})
        except Exception:
            pass
        try:
            random_state = state.get("random_state")
            if random_state is not None:
                random.setstate(random_state)
        except Exception:
            pass
        # bootstrap existing gen summaries
        if gen_summaries_path.exists():
            with gen_summaries_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    gen_summaries.append(json.loads(line))
        if episodes_path.exists():
            with episodes_path.open() as f:
                prev_n = sum(1 for _ in f)
    else:
        initial_orgs = getattr(config.population_strategy, "initial_orgs", 4)
        for _ in range(initial_orgs):
            oid = host.spawn_organelle(rank=config.host.max_lora_rank)
            population.register(
                Genome(
                    organelle_id=oid,
                    drive_weights={"novelty": 0.4},
                    gate_bias=0.0,
                    rank=config.host.max_lora_rank,
                )
            )

        # bootstrap uplift configuration
        assim = _make_assimilation_tester(config)
        sink = TelemetrySink(
            root=config.metrics.root,
            episodes_file=config.metrics.episodes_file,
            assimilation_file=config.metrics.assimilation_file,
        )
        environment = GridEnvironment(
            grid_cfg=config.grid,
            controller_cfg=config.controller,
            pricing_cfg=config.pricing,
            canary_cfg=config.canary,
            seed=args.seed,
            reward_bonus=config.environment.success_reward_bonus,
            failure_cost_multiplier=config.environment.failure_cost_multiplier,
            lp_alpha=getattr(config.curriculum, "lp_alpha", 0.5),
        )
        if args.disable_human or not config.human_bandit.enabled:
            human_bandit = None
        else:
            human_bandit = HumanBandit(
                preference_weight=config.human_bandit.preference_weight,
                helper_weight=config.human_bandit.helper_weight,
                frequency=config.human_bandit.frequency,
            )

        loop = EcologyLoop(
            config=config,
            host=host,
            environment=environment,
            population=population,
            assimilation=assim,
            human_bandit=human_bandit,
            sink=sink,
        )

    episodes_path = config.metrics.root / config.metrics.episodes_file
    if start_generation > 0 and episodes_path.exists():
        with episodes_path.open() as f:
            prev_n = sum(1 for _ in f)
    total_generations = start_generation + args.generations
    for gen in range(start_generation, total_generations):
        total_before = compute_budget.total_tokens
        total_generated_before = compute_budget.total_generated_tokens
        total_forwards_before = compute_budget.total_forward_passes
        total_updates_before = compute_budget.total_hebbian_updates
        train_before = compute_budget.train_tokens
        train_generated_before = compute_budget.train_generated_tokens
        train_forwards_before = compute_budget.train_forward_passes
        train_updates_before = compute_budget.train_hebbian_updates

        t_gen0 = time.time()
        summary = loop.run_generation(batch_size=config.environment.synthetic_batch_size)
        gen_wall = float(time.time() - t_gen0)
        compute_budget.add_wall_clock(gen_wall)
        if summary is None:
            summary = {}
        # summarize this generation's episodes
        curr_n = 0
        if episodes_path.exists():
            with episodes_path.open() as f:
                curr_n = sum(1 for _ in f)
        episode_slice = (
            summarize_slice(episodes_path, prev_n, curr_n)
            if curr_n > prev_n
            else {
                "episodes": 0,
                "avg_total": 0.0,
                "avg_task_reward": 0.0,
                "avg_cost_penalty": 0.0,
                "slice_tokens": 0,
                "slice_forward_passes": 0,
                "slice_hebbian_updates": 0,
            }
        )
        generation_record: dict[str, object] = {
            "generation": gen + 1,
            "population": len(population.population),
        }
        generation_record.update(summary)
        generation_record.update(
            {
                "episodes": int(episode_slice.get("episodes", 0)),
                "avg_total": float(episode_slice.get("avg_total", 0.0)),
                "avg_task_reward": float(episode_slice.get("avg_task_reward", 0.0)),
                "avg_cost_penalty": float(episode_slice.get("avg_cost_penalty", 0.0)),
            }
        )
        generation_record.update(
            {
                "slice_tokens": int(compute_budget.train_tokens - train_before),
                "slice_generated_tokens": int(
                    compute_budget.train_generated_tokens - train_generated_before
                ),
                "slice_forward_passes": int(
                    compute_budget.train_forward_passes - train_forwards_before
                ),
                "slice_hebbian_updates": int(
                    compute_budget.train_hebbian_updates - train_updates_before
                ),
                "slice_total_tokens": int(compute_budget.total_tokens - total_before),
                "slice_total_generated_tokens": int(
                    compute_budget.total_generated_tokens - total_generated_before
                ),
                "slice_total_forward_passes": int(
                    compute_budget.total_forward_passes - total_forwards_before
                ),
                "slice_total_hebbian_updates": int(
                    compute_budget.total_hebbian_updates - total_updates_before
                ),
                "gen_wall_clock_seconds": gen_wall,
            }
        )
        generation_record["compute_budget"] = compute_budget.summary()
        gen_summaries.append(generation_record)
        try:
            with gen_summaries_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(generation_record) + "\n")
        except Exception:
            pass
        prev_n = curr_n
        tuning = summary.get("assimilation_energy_tuning") or {}
        gating = summary.get("assimilation_gating") or {}
        message = (
            f"Generation {gen+1:03d} | ROI {summary.get('avg_roi', 0.0):.3f} | merges {summary.get('merges', 0)} "
            f"| energy floor {tuning.get('energy_floor', 0.0):.2f} (ROI≥{tuning.get('energy_floor_roi', 0.0):.2f}) "
            f"| gating low-energy {gating.get('low_energy', 0)} cooldown {gating.get('cooldown', 0)} "
            f"| episodes {generation_record['episodes']}"
        )
        # Optional trial/promotion counters for visibility
        trials = summary.get("trials_created")
        promos = summary.get("promotions")
        if isinstance(trials, int) and isinstance(promos, int):
            if trials or promos:
                message += f" | trials {trials} promotions {promos}"
        # Team metrics in ticker
        tr = summary.get("team_routes")
        tp = summary.get("team_promotions")
        if isinstance(tr, int) and isinstance(tp, int) and (tr or tp):
            message += f" | team routes {tr} promos {tp}"
        if "evaluation" in generation_record:
            eval_info = generation_record["evaluation"]
            message += (
                f" | eval {eval_info['accuracy']:.3f} ({eval_info['correct']}/{eval_info['total']})"
            )
        print(message, flush=True)

        if args.checkpoint_every and (gen + 1) % args.checkpoint_every == 0:
            telemetry = {}
            try:
                if episodes_path.exists():
                    telemetry["episodes"] = int(episodes_path.stat().st_size)
                if assimilation_path.exists():
                    telemetry["assimilation"] = int(assimilation_path.stat().st_size)
                if gen_summaries_path.exists():
                    telemetry["gen_summaries"] = int(gen_summaries_path.stat().st_size)
            except Exception:
                telemetry = {}
            _save_checkpoint(
                checkpoint_path,
                gen + 1,
                host,
                population,
                environment,
                ledger,
                loop,
                random.getstate(),
                telemetry,
                compute_budget,
            )

    if args.final_holdout is not None:
        try:
            print("[final-holdout] starting", flush=True)
            _run_final_holdout(
                holdout_path=args.final_holdout,
                holdout_sample_size=args.final_holdout_sample_size,
                config_path=args.config,
                output_root=config.metrics.root,
                host=host,
                population=population,
            )
            print("[final-holdout] done", flush=True)
        except Exception as exc:
            print(f"[final-holdout] failed: {exc}", flush=True)

    config.metrics.root.mkdir(parents=True, exist_ok=True)
    json_path = config.metrics.root / "gen_summaries.jsonl"
    jsonl = "\n".join(json.dumps(s) for s in gen_summaries)
    if jsonl:
        jsonl += "\n"
    json_path.write_text(jsonl, encoding="utf-8")

    fieldnames = sorted({key for record in gen_summaries for key in record.keys()})
    with (config.metrics.root / "gen_summaries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in gen_summaries:
            row = {}
            for key in fieldnames:
                value = record.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                row[key] = value
            writer.writerow(row)
    print("Telemetry root:", config.metrics.root)

    # Save final checkpoint
    telemetry = {}
    try:
        if episodes_path.exists():
            telemetry["episodes"] = int(episodes_path.stat().st_size)
        if assimilation_path.exists():
            telemetry["assimilation"] = int(assimilation_path.stat().st_size)
        if json_path.exists():
            telemetry["gen_summaries"] = int(json_path.stat().st_size)
    except Exception:
        telemetry = {}
    _save_checkpoint(
        checkpoint_path,
        total_generations,
        host,
        population,
        environment,
        ledger,
        loop,
        random.getstate(),
        telemetry,
        compute_budget,
    )
    # Print final compute summary
    print(f"Compute budget: {compute_budget.summary()}")


if __name__ == "__main__":
    main()
