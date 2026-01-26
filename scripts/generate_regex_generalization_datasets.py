#!/usr/bin/env python3
"""Generate distribution-matched regex_generalization task-mix datasets.

This produces three JSONL files in an output directory:
  - sft_train.jsonl: {"prompt": "...", "completion": "...", "capability": "...", "task_id": "..."}
  - selection_tasks.jsonl: RegexTask schema (see src/symbiont_ecology/evaluation/regex_generalization.py)
  - holdout_tasks.jsonl: RegexTask schema (same as selection_tasks.jsonl)

These datasets are meant for the regex_generalization evo-vs-SFT track:
  - SFT trains on sft_train.jsonl (mixed output formats: regex patterns, yes/no, explanations)
  - Evolved organelle selection uses selection_tasks.jsonl
  - Final in-distribution evaluation uses holdout_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from symbiont_ecology import load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment, GridTask


def _alloc_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not weights:
        raise ValueError("weights must be non-empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0:
        raise ValueError("weights sum must be positive")
    normalized = {k: float(v) / weight_sum for k, v in weights.items()}
    raw = {k: float(total) * p for k, p in normalized.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = int(total - sum(counts.values()))
    if remainder <= 0:
        return counts
    fractional = sorted(
        ((k, raw[k] - counts[k]) for k in counts.keys()),
        key=lambda item: item[1],
        reverse=True,
    )
    for i in range(remainder):
        counts[fractional[i % len(fractional)][0]] += 1
    return counts


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _stable_target_signature(target: object) -> str:
    try:
        return json.dumps(target, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(target)


def _grid_task_key(task: GridTask) -> tuple[str, str, str]:
    return (str(task.family), str(task.prompt), _stable_target_signature(task.target))


def _capability_from_family(family: str) -> str:
    fam = str(family).strip().lower()
    if fam in {"regex", "regex.synthesis"}:
        return "synthesis"
    if fam == "regex.debugging":
        return "debugging"
    if fam == "regex.recognition":
        return "recognition"
    if fam == "regex.explanation":
        return "explanation"
    if fam == "regex.refactoring":
        return "refactoring"
    if fam == "regex.mutation_effect":
        # Match the benchmark file, where these tasks are under the recognition axis.
        return "recognition"
    return "synthesis"


def _grid_task_to_regex_task(task: GridTask, *, task_id: str) -> dict[str, Any]:
    family = str(task.family)
    capability = _capability_from_family(family)
    target: dict[str, Any] = task.target if isinstance(task.target, dict) else {}

    out: dict[str, Any] = {
        "task_id": task_id,
        "prompt": str(task.prompt),
        "capability": capability,
        "holdout_type": None,
        "mutation_type": None,
        "target_regex": None,
        "test_cases": [],
        "expected_answer": None,
        "metadata": {},
    }

    if family in {"regex", "regex.synthesis"}:
        out["target_regex"] = str(target.get("pattern") or "")
        out["test_cases"] = list(target.get("test_strings") or [])
        return out

    if family == "regex.debugging":
        out["target_regex"] = str(target.get("broken_pattern") or target.get("pattern") or "")
        out["test_cases"] = list(target.get("test_strings") or [])
        out["metadata"] = {"bug_description": target.get("bug_description")}
        return out

    if family == "regex.refactoring":
        out["target_regex"] = str(target.get("original_pattern") or "")
        out["test_cases"] = list(target.get("test_strings") or [])
        return out

    if family == "regex.recognition":
        expected = target.get("expected")
        expected_bool = bool(expected) if expected is not None else False
        out["target_regex"] = str(target.get("pattern") or "")
        out["expected_answer"] = "yes" if expected_bool else "no"
        out["metadata"] = {"test_string": target.get("test_string")}
        return out

    if family == "regex.explanation":
        out["target_regex"] = str(target.get("pattern") or "")
        out["metadata"] = {"required_keywords": list(target.get("required_keywords") or [])}
        return out

    if family == "regex.mutation_effect":
        out["target_regex"] = str(target.get("pattern") or "")
        out["metadata"] = {
            "mutated_regex": target.get("mutated_pattern"),
            "required_keywords": list(target.get("required_keywords") or []),
        }
        return out

    return out


def _completion_for_task(task: GridTask) -> str | None:
    family = str(task.family)
    if family in {"regex", "regex.synthesis", "regex.debugging", "regex.refactoring"}:
        if isinstance(task.target, dict):
            value = str(task.target.get("pattern") or "")
            return value or None
        return None

    if family == "regex.recognition":
        if isinstance(task.target, dict):
            expected = task.target.get("expected")
            return "yes" if bool(expected) else "no"
        if isinstance(task.target, bool):
            return "yes" if task.target else "no"
        return None

    if family in {"regex.explanation", "regex.mutation_effect"}:
        if isinstance(task.target, dict):
            value = str(task.target.get("reference_answer") or "")
            return value or None
        return None

    return None


def _generate_grid_tasks(
    env: GridEnvironment,
    *,
    families: list[str],
    depths: list[str],
    counts: dict[str, int],
    rng: random.Random,
    exclude: set[tuple[str, str, str]],
    dedupe_within_split: bool,
) -> list[GridTask]:
    tasks: list[GridTask] = []
    local_seen: set[tuple[str, str, str]] = set() if dedupe_within_split else set()
    attempts = 0
    total_target = sum(int(counts.get(f, 0)) for f in families)
    family_order = [f for f in families if int(counts.get(f, 0)) > 0]
    if not family_order:
        return []

    for family in family_order:
        needed = int(counts.get(family, 0))
        for _ in range(needed):
            while True:
                attempts += 1
                if attempts > total_target * 200:
                    raise RuntimeError(
                        f"Failed to generate enough tasks (got {len(tasks)}/{total_target})."
                    )
                depth = rng.choice(depths)
                task = env._make_task((family, depth), canary=False)  # noqa: SLF001
                key = _grid_task_key(task)
                if key in exclude:
                    continue
                if dedupe_within_split and key in local_seen:
                    continue
                if dedupe_within_split:
                    local_seen.add(key)
                tasks.append(task)
                break

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mixed regex_generalization datasets from GridEnvironment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/qwen3_regex_generalization.yaml"),
        help="EcologyConfig YAML used for grid families/depths.",
    )
    parser.add_argument("--seed", type=int, default=777, help="Base RNG seed (default: 777).")
    parser.add_argument(
        "--train-size", type=int, default=20000, help="SFT train task count (default: 20000)."
    )
    parser.add_argument(
        "--selection-size",
        type=int,
        default=64,
        help="Selection/validation task count for organelle picking (default: 64).",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=512,
        help="Final in-distribution holdout task count (default: 512).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated JSONL files.",
    )
    args = parser.parse_args()

    cfg = load_ecology_config(Path(args.config))
    families = [str(f) for f in (cfg.grid.families or [])]
    depths = [str(d) for d in (cfg.grid.depths or ["short", "medium", "long"])]
    if not families:
        raise ValueError("Config has empty grid.families")
    if not depths:
        raise ValueError("Config has empty grid.depths")

    # Task-mix weights (roughly aligned to regex_generalization.jsonl proportions).
    synthesis_family = "regex" if "regex" in families else "regex.synthesis"
    weights: dict[str, float] = {
        "regex.recognition": 0.25,
        "regex.mutation_effect": 0.10,
        synthesis_family: 0.25,
        "regex.debugging": 0.20,
        "regex.refactoring": 0.10,
        "regex.explanation": 0.10,
    }
    # Keep only families present in the config.
    weights = {f: w for f, w in weights.items() if f in families}
    if not weights:
        raise ValueError("Config grid.families is incompatible with generator weights")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reserved: set[tuple[str, str, str]] = set()
    train_rng = random.Random(int(args.seed) + 1)
    sel_rng = random.Random(int(args.seed) + 2)
    holdout_rng = random.Random(int(args.seed) + 3)

    env_train = GridEnvironment(
        cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=int(args.seed) + 11
    )
    env_sel = GridEnvironment(
        cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=int(args.seed) + 22
    )
    env_holdout = GridEnvironment(
        cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=int(args.seed) + 33
    )

    train_counts = _alloc_counts(int(args.train_size), weights)
    selection_counts = _alloc_counts(int(args.selection_size), weights)
    holdout_counts = _alloc_counts(int(args.holdout_size), weights)

    selection_tasks = _generate_grid_tasks(
        env_sel,
        families=list(selection_counts.keys()),
        depths=depths,
        counts=selection_counts,
        rng=sel_rng,
        exclude=reserved,
        dedupe_within_split=True,
    )
    reserved.update(_grid_task_key(task) for task in selection_tasks)
    holdout_tasks = _generate_grid_tasks(
        env_holdout,
        families=list(holdout_counts.keys()),
        depths=depths,
        counts=holdout_counts,
        rng=holdout_rng,
        exclude=reserved,
        dedupe_within_split=False,
    )
    reserved.update(_grid_task_key(task) for task in holdout_tasks)
    # Train split can contain duplicates, but must not overlap selection/holdout tasks.
    train_tasks = _generate_grid_tasks(
        env_train,
        families=list(train_counts.keys()),
        depths=depths,
        counts=train_counts,
        rng=train_rng,
        exclude=reserved,
        dedupe_within_split=False,
    )

    sft_rows: list[dict[str, Any]] = []
    for idx, task in enumerate(train_tasks):
        completion = _completion_for_task(task)
        if not completion:
            continue
        sft_rows.append(
            {
                "task_id": f"train_{idx:06d}",
                "capability": _capability_from_family(str(task.family)),
                "prompt": str(task.prompt),
                "completion": completion,
            }
        )

    selection_rows = [
        _grid_task_to_regex_task(task, task_id=f"sel_{idx:05d}")
        for idx, task in enumerate(selection_tasks)
    ]
    holdout_rows = [
        _grid_task_to_regex_task(task, task_id=f"id_{idx:05d}")
        for idx, task in enumerate(holdout_tasks)
    ]

    _atomic_write_jsonl(out_dir / "sft_train.jsonl", sft_rows)
    _atomic_write_jsonl(out_dir / "selection_tasks.jsonl", selection_rows)
    _atomic_write_jsonl(out_dir / "holdout_tasks.jsonl", holdout_rows)
    _atomic_write_jsonl(
        out_dir / "dataset_metadata.json",
        [
            {
                "config": str(Path(args.config)),
                "seed": int(args.seed),
                "depths": depths,
                "families": list(weights.keys()),
                "weights": weights,
                "train_size": int(args.train_size),
                "selection_size": int(args.selection_size),
                "holdout_size": int(args.holdout_size),
                "train_counts": train_counts,
                "selection_counts": selection_counts,
                "holdout_counts": holdout_counts,
                "written_sft_rows": len(sft_rows),
            }
        ],
    )

    print(f"Wrote: {out_dir / 'sft_train.jsonl'}")
    print(f"Wrote: {out_dir / 'selection_tasks.jsonl'}")
    print(f"Wrote: {out_dir / 'holdout_tasks.jsonl'}")


if __name__ == "__main__":
    main()
