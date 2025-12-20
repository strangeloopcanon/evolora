#!/usr/bin/env python3
"""Long Gemma evolution with per-generation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

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
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome

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


def _parse_last_number(text: str) -> float | None:
    matches = list(re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except Exception:
        return None


def _parse_last_int(text: str) -> int | None:
    matches = list(re.finditer(r"\b\d+\b", text))
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
    if family in {"math", "math.sequence", "math.multi_step"}:
        predicted = _parse_last_number(answer)
        if predicted is None:
            return False
        try:
            expected = float(target)
        except Exception:
            return False
        return math.isclose(predicted, expected, rel_tol=1e-3)
    if family == "word.count":
        predicted = _parse_last_int(answer)
        if predicted is None:
            tokens = answer.strip().lower().split()
            words_map = {
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
            for tok in tokens[::-1]:
                if tok in words_map:
                    predicted = words_map[tok]
                    break
        if predicted is None:
            return False
        try:
            expected = int(target)
        except Exception:
            return False
        return predicted == expected
    if family == "code.format":
        expected_raw = str(target).strip()
        if "\n" in expected_raw:
            normalized = _normalize_code_answer(answer, multiline=True)
            expected_lines = [line.rstrip() for line in expected_raw.splitlines()]
            expected = "\n".join(expected_lines).strip()
            return normalized == expected
        normalized = _normalize_code_answer(answer, multiline=False)
        expected = expected_raw.strip().strip("'\"")
        return normalized == expected
    return answer.strip() == str(target).strip()


def _select_best_organelle_for_cell(
    population: PopulationManager, cell: tuple[str, str], candidates: list[str]
) -> tuple[str, float]:
    best_id = candidates[0]
    best_score = float("-inf")
    for oid in candidates:
        score = population.cell_values.get(oid, {}).get(cell)
        if score is None:
            continue
        try:
            val = float(score)
        except Exception:
            continue
        if val > best_score:
            best_score = val
            best_id = oid
    if best_score == float("-inf"):
        scored = [(oid, float(population.average_roi(oid, limit=10))) for oid in candidates]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        best_id, best_score = scored[0][0], scored[0][1]
    return best_id, best_score


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
    return {
        "episodes": end_idx - start_idx,
        "avg_total": float(mean(totals)) if totals else 0.0,
        "avg_task_reward": float(mean(task_rewards)) if task_rewards else 0.0,
        "avg_cost_penalty": float(mean(costs)) if costs else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-form evolution on Gemma host.")
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
        default=Path("artifacts_gemma_long_eval"),
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


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return pickle.loads(path.read_bytes())


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
) -> None:
    adapter_states: dict[str, object] = {}
    for oid, org in host.organelles.items():
        try:
            adapter_states[oid] = org.export_adapter_state()  # type: ignore[attr-defined]
        except Exception:
            adapter_states[oid] = {}
    state = {
        "generation": generation,
        "population": population,
        "environment_state": {
            "controller": environment.controller,
            "organism_stats": environment.organism_stats,
            "organism_canary_fail": environment.organism_canary_fail,
            "rng_state": environment.rng.getstate(),
            "task_counter": environment.task_counter,
        },
        "ledger": ledger,
        "adapter_states": adapter_states,
        "random_state": random_state,
        "assimilation_cooldown": loop.assimilation_cooldown,
        "tau_relief": getattr(loop, "_tau_relief", {}),
        "telemetry_bytes": dict(telemetry_bytes or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(state))


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
        config.host.backbone_model = "google/gemma-3-270m-it"
    if not getattr(config.host, "tokenizer", None):
        config.host.tokenizer = config.host.backbone_model

    if args.device is not None:
        config.host.device = args.device
    elif config.host.device == "cpu":
        config.host.device = "auto"

    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
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
        state = _load_checkpoint(checkpoint_path)
        start_generation = int(state.get("generation", 0))
        telemetry_bytes = state.get("telemetry_bytes") if isinstance(state, dict) else None
        if isinstance(telemetry_bytes, dict):
            _truncate_file_to_bytes(episodes_path, telemetry_bytes.get("episodes"))
            _truncate_file_to_bytes(assimilation_path, telemetry_bytes.get("assimilation"))
            _truncate_file_to_bytes(gen_summaries_path, telemetry_bytes.get("gen_summaries"))
        # restore population
        population = state.get("population", population)
        # respawn organelles with saved IDs and adapter states
        adapter_states = state.get("adapter_states", {}) or {}
        population.population = dict(population.population)
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
                except Exception:
                    pass
            ledger.ensure(oid, 0.0)
            ledger.ensure_energy(oid, 0.0)
        # restore ledger balances
        saved_ledger: ATPLedger = state.get("ledger", ledger)
        ledger.accounts = saved_ledger.accounts
        ledger.energy_accounts = saved_ledger.energy_accounts
        ledger.energy_cap = saved_ledger.energy_cap
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
        try:
            environment.controller = env_state.get("controller", environment.controller)
            environment.organism_stats = env_state.get("organism_stats", environment.organism_stats)
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

    assim = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )
    # bootstrap uplift configuration
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

    episodes_path = config.metrics.root / config.metrics.episodes_file
    if start_generation > 0 and episodes_path.exists():
        with episodes_path.open() as f:
            prev_n = sum(1 for _ in f)
    total_generations = start_generation + args.generations
    for gen in range(start_generation, total_generations):
        summary = loop.run_generation(batch_size=config.environment.synthetic_batch_size)
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
            else {"episodes": 0, "avg_total": 0.0, "avg_task_reward": 0.0, "avg_cost_penalty": 0.0}
        )
        generation_record: dict[str, object] = {
            "generation": gen + 1,
            "population": len(population.population),
        }
        generation_record.update(summary)
        generation_record.update(episode_slice)
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
    json_path.write_text("\n".join(json.dumps(s) for s in gen_summaries))

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
    )


if __name__ == "__main__":
    main()
