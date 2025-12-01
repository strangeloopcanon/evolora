#!/usr/bin/env python3
"""Long Gemma evolution with per-generation summaries."""

from __future__ import annotations

import json
import argparse
import csv
from pathlib import Path
import pickle
import random
from statistics import mean
import warnings

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="You are trying to modify a model with PEFT for a second time")
warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute")
warnings.filterwarnings(
    "ignore",
    message=r"Adapter .* was active which is now deleted. Setting active adapter to default.",
)

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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(state))


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

    checkpoint_path = config.metrics.root / "checkpoint.pt"
    if resume_root is not None and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path)
        start_generation = int(state.get("generation", 0))
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
            environment.organism_canary_fail = env_state.get("organism_canary_fail", environment.organism_canary_fail)
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
            sink=TelemetrySink(root=config.metrics.root, episodes_file=config.metrics.episodes_file, assimilation_file=config.metrics.assimilation_file),
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
        gs_path = config.metrics.root / "gen_summaries.jsonl"
        if gs_path.exists():
            with gs_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    gen_summaries.append(json.loads(line))
        episodes_path = config.metrics.root / config.metrics.episodes_file
        if episodes_path.exists():
            with episodes_path.open() as f:
                prev_n = sum(1 for _ in f)
    else:
        for _ in range(4):
            oid = host.spawn_organelle(rank=config.host.max_lora_rank)
            population.register(Genome(organelle_id=oid, drive_weights={"novelty": 0.4}, gate_bias=0.0, rank=config.host.max_lora_rank))

        # bootstrap uplift configuration
        assim = _make_assimilation_tester(config)
        sink = TelemetrySink(root=config.metrics.root, episodes_file=config.metrics.episodes_file, assimilation_file=config.metrics.assimilation_file)
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
        # Optional trial/promotion counters for visibility
        trials = summary.get("trials_created")
        promos = summary.get("promotions")
        if isinstance(trials, int) and isinstance(promos, int):
            if trials or promos:
                message += f" | trials {trials} promotions {promos}"
        # Team metrics in ticker
        tr = summary.get("team_routes"); tp = summary.get("team_promotions")
        if isinstance(tr, int) and isinstance(tp, int) and (tr or tp):
            message += f" | team routes {tr} promos {tp}"
        if "evaluation" in generation_record:
            eval_info = generation_record["evaluation"]
            message += f" | eval {eval_info['accuracy']:.3f} ({eval_info['correct']}/{eval_info['total']})"
        print(message, flush=True)

        if args.checkpoint_every and (gen + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path,
                gen + 1,
                host,
                population,
                environment,
                ledger,
                loop,
                random.getstate(),
            )

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
    _save_checkpoint(checkpoint_path, total_generations, host, population, environment, ledger, loop, random.getstate())


if __name__ == "__main__":
    main()
