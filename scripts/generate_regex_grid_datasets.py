#!/usr/bin/env python3
"""Generate distribution-matched regex datasets from the GridEnvironment generator.

Produces three JSONL files in an output directory:
  - sft_train.jsonl: {"prompt": "...", "completion": "<regex pattern>"}
  - selection_tasks.jsonl: {"prompt": "...", "target": {"pattern": "...", "test_strings": [...]}, "family": "regex", "depth": "..."}
  - holdout_tasks.jsonl: same schema as selection_tasks.jsonl

These datasets are meant for the regex evo-vs-SFT track:
  - SFT trains on sft_train.jsonl
  - Evolved organelle selection uses selection_tasks.jsonl
  - Final in-distribution evaluation uses holdout_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from symbiont_ecology import load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment


def _validate_task(pattern: str, test_strings: list[dict[str, object]]) -> bool:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    if not test_strings:
        return False
    for tc in test_strings:
        text = str(tc.get("string", ""))
        should_match = bool(tc.get("should_match", False))
        matched = bool(compiled.fullmatch(text))
        if matched != should_match:
            return False
    return True


def _generate_tasks(
    env: GridEnvironment,
    *,
    depths: list[str],
    count: int,
    rng: random.Random,
    seen: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    attempts = 0
    target = int(count)
    while len(tasks) < target:
        attempts += 1
        if attempts > target * 50:
            raise RuntimeError(
                f"Failed to generate enough unique tasks (got {len(tasks)}/{target})."
            )
        depth = rng.choice(depths)
        prompt, pattern, test_strings = env._make_regex_task(depth)  # noqa: SLF001
        key = (prompt, pattern)
        if key in seen:
            continue
        if not _validate_task(pattern, test_strings):
            continue
        seen.add(key)
        tasks.append(
            {
                "prompt": prompt,
                "target": {"pattern": pattern, "test_strings": test_strings},
                "family": "regex",
                "depth": depth,
            }
        )
    return tasks


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sft_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    rows = []
    for task in tasks:
        target = task.get("target") or {}
        pattern = str(target.get("pattern", "") or "")
        prompt = str(task.get("prompt", "") or "")
        if not prompt or not pattern:
            continue
        rows.append({"prompt": prompt, "completion": pattern})
    _write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate distribution-matched regex datasets from GridEnvironment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/qwen3_regex.yaml"),
        help="EcologyConfig YAML used for depths (default: config/experiments/qwen3_regex.yaml).",
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
        default=256,
        help="Final in-distribution holdout task count (default: 256).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated JSONL files.",
    )
    args = parser.parse_args()

    cfg = load_ecology_config(Path(args.config))
    depths = [str(depth) for depth in (cfg.grid.depths or ["short", "medium", "long"])]
    if not depths:
        raise ValueError("Config has empty grid.depths")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use distinct RNG streams per split to avoid accidental overlap.
    seen: set[tuple[str, str]] = set()
    train_rng = random.Random(int(args.seed) + 1)
    sel_rng = random.Random(int(args.seed) + 2)
    holdout_rng = random.Random(int(args.seed) + 3)

    # Instantiate fresh environments per split to keep internal RNG independent.
    env_train = GridEnvironment(
        cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=int(args.seed) + 11
    )
    env_sel = GridEnvironment(
        cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=int(args.seed) + 22
    )
    env_holdout = GridEnvironment(
        cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=int(args.seed) + 33
    )

    train_tasks = _generate_tasks(
        env_train, depths=depths, count=int(args.train_size), rng=train_rng, seen=seen
    )
    selection_tasks = _generate_tasks(
        env_sel, depths=depths, count=int(args.selection_size), rng=sel_rng, seen=seen
    )
    holdout_tasks = _generate_tasks(
        env_holdout, depths=depths, count=int(args.holdout_size), rng=holdout_rng, seen=seen
    )

    _write_sft_jsonl(out_dir / "sft_train.jsonl", train_tasks)
    _write_jsonl(out_dir / "selection_tasks.jsonl", selection_tasks)
    _write_jsonl(out_dir / "holdout_tasks.jsonl", holdout_tasks)
    _write_jsonl(
        out_dir / "dataset_metadata.json",
        [
            {
                "config": str(Path(args.config)),
                "seed": int(args.seed),
                "depths": depths,
                "train_size": int(args.train_size),
                "selection_size": int(args.selection_size),
                "holdout_size": int(args.holdout_size),
            }
        ],
    )

    print(f"Wrote: {out_dir / 'sft_train.jsonl'}")
    print(f"Wrote: {out_dir / 'selection_tasks.jsonl'}")
    print(f"Wrote: {out_dir / 'holdout_tasks.jsonl'}")


if __name__ == "__main__":
    main()
