#!/usr/bin/env python3
"""Run a quick best-of-two inference over a colony (set of organelles).

Usage:
  .venv/bin/python scripts/colony_infer.py --config config/experiments/qwen3_endogenous.yaml \
      --members auto --prompt "Sort words: pear banana apple"

Notes:
  - If --members auto, we spawn a small population and pick any 2 live organelles.
  - This is a convenience utility; for querying trained adapters from a past run,
    add persistence of organelle adapter states and reload before calling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from symbiont_ecology import ATPLedger, BanditRouter, HostKernel, load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colony inference (best-of-two)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--members", type=str, default="auto", help="comma-separated organelle IDs or 'auto'"
    )
    p.add_argument("--prompt", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_ecology_config(args.config)
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()
    population = PopulationManager(cfg.evolution, cfg.foraging)
    # Spawn a minimal pop when auto
    if args.members.strip().lower() == "auto":
        for _ in range(2):
            oid = host.spawn_organelle(rank=cfg.host.max_lora_rank)
            population.register(
                Genome(
                    organelle_id=oid,
                    drive_weights={"novelty": 0.4},
                    gate_bias=0.0,
                    rank=cfg.host.max_lora_rank,
                )
            )
        members: List[str] = host.list_organelle_ids()[:2]
    else:
        members = [m.strip() for m in args.members.split(",") if m.strip()]
    env = GridEnvironment(
        cfg.grid,
        cfg.controller,
        cfg.pricing,
        cfg.canary,
        seed=123,
        reward_bonus=cfg.environment.success_reward_bonus,
        failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
        lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5),
    )
    loop = EcologyLoop(config=cfg, host=host, environment=env, population=population, assimilation=None)  # type: ignore[arg-type]
    result = loop.run_colony_inference(members, args.prompt, strategy="best_of_two")
    print("Selected:", result["selected_id"])  # noqa: T201
    print("Answer:\n", result["selected_answer"])  # noqa: T201
    print("All answers:")  # noqa: T201
    for k, v in result["answers"].items():
        print(f"  - {k}: {v}")  # noqa: T201


if __name__ == "__main__":
    main()
