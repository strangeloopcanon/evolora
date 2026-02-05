#!/usr/bin/env python3
"""Quick diagnostic: does the base Gemma+LoRA setup solve our grid tasks?"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from symbiont_ecology import (
    ATPLedger,
    BanditRouter,
    HostKernel,
    load_ecology_config,
)
from symbiont_ecology.environment.grid import GridEnvironment, GridKey
from symbiont_ecology.evaluation.manager import EvaluationManager


def run_grid_diagnostics(cfg_path: Path, per_cell: int) -> None:
    cfg = load_ecology_config(cfg_path)
    # Force known backbone identifiers and device resolution from YAML
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()
    # One fresh organelle with max rank — initial LoRA is effectively neutral
    oid = host.spawn_organelle(rank=cfg.host.max_lora_rank)

    env = GridEnvironment(
        grid_cfg=cfg.grid,
        controller_cfg=cfg.controller,
        pricing_cfg=cfg.pricing,
        canary_cfg=cfg.canary,
        seed=1234,
        reward_bonus=cfg.environment.success_reward_bonus,
        failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
        lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5),
    )

    families = list(cfg.grid.families)
    depths = list(cfg.grid.depths)
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    total = 0
    correct = 0

    for fam in families:
        for dep in depths:
            cell: GridKey = (str(fam), str(dep))
            wins = 0
            tried = 0
            for _i in range(per_cell):
                task = env.sample_task_from_cell(cell)
                result = host.step(
                    prompt=task.prompt,
                    intent="diagnostic",
                    max_routes=1,
                    allowed_organelle_ids=[oid],
                )
                answer = result.envelope.observation.state.get("answer", "")
                success, _ = task.evaluate(answer)
                wins += 1 if success else 0
                tried += 1
                # update controller so pricing/difficulty tracks behaviour
                env.register_result(oid, task, success)
            summary[(fam, dep)] = {
                "acc": (wins / tried) if tried else 0.0,
                "n": tried,
            }
            total += tried
            correct += wins

    print("Base model diagnostic (fresh LoRA, no training)")
    print("Per-cell accuracy (family,depth → acc @ n):")
    for fam in families:
        for dep in depths:
            key = (fam, dep)
            stats = summary[key]
            print(f"  {fam:14s} {dep:7s} → {stats['acc']:.2f} @ {int(stats['n'])}")
    overall = (correct / total) if total else 0.0
    print(f"Overall grid accuracy: {overall:.2f} ({correct}/{total})")

    # Optional: evaluate on configured holdout tasks if enabled
    if cfg.evaluation.enabled and cfg.evaluation.tasks_path:
        eval_runtime = EvaluationManager.from_file(
            enabled=True,
            cadence=cfg.evaluation.cadence,
            tasks_path=cfg.evaluation.tasks_path,
            sample_size=cfg.evaluation.sample_size,
            reward_weight=cfg.evaluation.reward_weight,
        )
        manager = EvaluationManager(eval_runtime)
        eval_stats = manager.evaluate(host, env)
        print(
            "Holdout eval accuracy:",
            f"{eval_stats['accuracy']:.2f} ({eval_stats['correct']}/{eval_stats['total']})",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe base model capability on grid cells and holdout tasks."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/gemma_endogenous.yaml"),
        help="Path to ecology YAML config",
    )
    ap.add_argument(
        "--per-cell",
        type=int,
        default=10,
        help="Number of diagnostic samples per (family,depth)",
    )
    args = ap.parse_args()
    run_grid_diagnostics(args.config, per_cell=args.per_cell)


if __name__ == "__main__":
    main()
