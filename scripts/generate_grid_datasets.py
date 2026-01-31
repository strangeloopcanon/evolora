#!/usr/bin/env python3
"""Generate distribution-matched datasets from GridEnvironment for arbitrary task families.

Produces three JSONL files in an output directory:
  - sft_train.jsonl: {"task_id": "...", "family": "...", "depth": "...", "prompt": "...", "completion": "..."}
  - selection_tasks.jsonl: {"task_id": "...", "family": "...", "depth": "...", "prompt": "...", "target": ...}
  - holdout_tasks.jsonl: same schema as selection_tasks.jsonl

These datasets are meant for multi-objective evo-vs-SFT experiments:
  - SFT trains on sft_train.jsonl (targets from GridTask.supervised_completion)
  - Evolved organelle selection uses selection_tasks.jsonl
  - Final evaluation uses holdout_tasks.jsonl
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


def _grid_task_key(task: GridTask) -> tuple[str, str, str, str]:
    return (
        str(task.family),
        str(task.depth),
        str(task.prompt),
        _stable_target_signature(task.target),
    )


def _generate_grid_tasks(
    env: GridEnvironment,
    *,
    families: list[str],
    depths: list[str],
    counts: dict[str, int],
    rng: random.Random,
    exclude: set[tuple[str, str, str, str]],
    dedupe_within_split: bool,
) -> list[GridTask]:
    tasks: list[GridTask] = []
    local_seen: set[tuple[str, str, str, str]] = set() if dedupe_within_split else set()
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
                if attempts > max(1, total_target) * 200:
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
        description="Generate distribution-matched datasets from GridEnvironment across families."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/qwen3_endogenous.yaml"),
        help="EcologyConfig YAML used for grid families/depths.",
    )
    parser.add_argument("--seed", type=int, default=777, help="Base RNG seed (default: 777).")
    parser.add_argument(
        "--train-size", type=int, default=20000, help="SFT train task count (default: 20000)."
    )
    parser.add_argument(
        "--selection-size",
        type=int,
        default=256,
        help="Selection/validation task count for organelle picking (default: 256).",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=512,
        help="Final in-distribution holdout task count (default: 512).",
    )
    parser.add_argument(
        "--weights-json",
        type=str,
        default="",
        help="Optional JSON mapping of family -> weight. Default is uniform over config families.",
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

    weights: dict[str, float]
    if str(args.weights_json).strip():
        raw = json.loads(str(args.weights_json))
        if not isinstance(raw, dict):
            raise ValueError("--weights-json must be a JSON object mapping family->weight")
        weights = {}
        for key, value in raw.items():
            fam = str(key)
            if fam not in families:
                raise ValueError(f"--weights-json contains unknown family: {fam}")
            weights[fam] = float(value)
        if not weights:
            raise ValueError("--weights-json produced an empty weights mapping")
    else:
        weights = {fam: 1.0 for fam in families}

    train_counts = _alloc_counts(int(args.train_size), weights)
    selection_counts = _alloc_counts(int(args.selection_size), weights)
    holdout_counts = _alloc_counts(int(args.holdout_size), weights)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reserved: set[tuple[str, str, str, str]] = set()
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

    selection_tasks = _generate_grid_tasks(
        env_sel,
        families=families,
        depths=depths,
        counts=selection_counts,
        rng=sel_rng,
        exclude=reserved,
        dedupe_within_split=False,
    )
    for task in selection_tasks:
        reserved.add(_grid_task_key(task))
    holdout_tasks = _generate_grid_tasks(
        env_holdout,
        families=families,
        depths=depths,
        counts=holdout_counts,
        rng=holdout_rng,
        exclude=reserved,
        dedupe_within_split=False,
    )
    for task in holdout_tasks:
        reserved.add(_grid_task_key(task))
    train_tasks = _generate_grid_tasks(
        env_train,
        families=families,
        depths=depths,
        counts=train_counts,
        rng=train_rng,
        exclude=reserved,
        dedupe_within_split=False,
    )

    sft_rows: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for idx, task in enumerate(train_tasks):
        completion = task.supervised_completion()
        if completion is None:
            missing.append((str(task.family), str(task.depth)))
            continue
        sft_rows.append(
            {
                "task_id": f"train_{idx:06d}",
                "family": str(task.family),
                "depth": str(task.depth),
                "prompt": str(task.prompt),
                "completion": str(completion),
            }
        )
    if missing:
        # This typically means GridTask.supervised_completion() lacks support for a family.
        missing_preview = ", ".join(sorted({f"{fam}:{depth}" for fam, depth in missing})[:10])
        raise RuntimeError(f"Missing supervised_completion for some tasks: {missing_preview}")

    selection_rows = [
        {
            "task_id": f"sel_{idx:05d}",
            "family": str(task.family),
            "depth": str(task.depth),
            "prompt": str(task.prompt),
            "target": task.target,
        }
        for idx, task in enumerate(selection_tasks)
    ]
    holdout_rows = [
        {
            "task_id": f"id_{idx:05d}",
            "family": str(task.family),
            "depth": str(task.depth),
            "prompt": str(task.prompt),
            "target": task.target,
        }
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
