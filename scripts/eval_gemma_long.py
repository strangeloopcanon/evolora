#!/usr/bin/env python3
"""Long Gemma evolution with per-generation summaries."""

from __future__ import annotations

import json
import argparse
import csv
from pathlib import Path
from statistics import mean
import warnings

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="You are trying to modify a model with PEFT for a second time")
warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute")

from symbiont_ecology import (
    ATPLedger,
    AssimilationTester,
    BanditRouter,
    HostKernel,
    HumanBandit,
    PopulationManager,
    TelemetrySink,
    load_ecology_config,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


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
            total = rb["task_reward"] + rb["novelty_bonus"] + rb["competence_bonus"] + rb["helper_bonus"] - rb["risk_penalty"] - rb["cost_penalty"]
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_ecology_config(args.config)
    config.metrics.root = args.output
    config.metrics.root.mkdir(parents=True, exist_ok=True)
    if args.batch_size is not None:
        config.environment.synthetic_batch_size = args.batch_size

    config.host.backbone_model = "google/gemma-3-270m-it"
    config.host.tokenizer = "google/gemma-3-270m-it"

    if args.device is not None:
        config.host.device = args.device
    elif config.host.device == "cpu":
        config.host.device = "auto"

    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    population = PopulationManager(config.evolution)
    for _ in range(4):
        oid = host.spawn_organelle(rank=config.host.max_lora_rank)
        population.register(Genome(organelle_id=oid, drive_weights={"novelty": 0.4}, gate_bias=0.0, rank=config.host.max_lora_rank))

    assim = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )

    sink = TelemetrySink(root=config.metrics.root, episodes_file=config.metrics.episodes_file, assimilation_file=config.metrics.assimilation_file)
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=args.seed,
        reward_bonus=config.environment.success_reward_bonus,
        failure_cost_multiplier=config.environment.failure_cost_multiplier,
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
    gen_summaries = []
    prev_n = 0
    for gen in range(args.generations):
        summary = loop.run_generation(batch_size=config.environment.synthetic_batch_size)
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
        prev_n = curr_n
        tuning = summary.get("assimilation_energy_tuning") or {}
        gating = summary.get("assimilation_gating") or {}
        message = (
            f"Generation {gen+1:03d} | ROI {summary.get('avg_roi', 0.0):.3f} | merges {summary.get('merges', 0)} "
            f"| energy floor {tuning.get('energy_floor', 0.0):.2f} (ROI≥{tuning.get('energy_floor_roi', 0.0):.2f}) "
            f"| gating low-energy {gating.get('low_energy', 0)} cooldown {gating.get('cooldown', 0)} "
            f"| episodes {generation_record['episodes']}"
        )
        if "evaluation" in generation_record:
            eval_info = generation_record["evaluation"]
            message += f" | eval {eval_info['accuracy']:.3f} ({eval_info['correct']}/{eval_info['total']})"
        print(message, flush=True)

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


if __name__ == "__main__":
    main()
