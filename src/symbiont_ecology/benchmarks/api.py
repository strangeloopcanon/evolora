"""Benchmark suite runner.

The benchmark harness is designed to compare multiple configurations (frozen,
single-adapter, full ecology) on the same codebase, producing a small JSON report.

For CI, a stub backend is provided so the harness can run without downloading models.
"""

from __future__ import annotations

import json
import random
import subprocess
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

import torch
from pydantic import BaseModel, Field

from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    HostKernel,
    PopulationManager,
    TelemetrySink,
    load_ecology_config,
)
from symbiont_ecology.benchmarks.stubs import BenchmarkStubBackbone, BenchmarkStubOrganelle
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome

Backend = Literal["stub", "hf"]


class BenchmarkCase(BaseModel):
    name: str
    config_path: Path
    generations: int = Field(1, ge=1)
    seed: int = Field(1337, ge=0)
    batch_size: int = Field(4, ge=1)
    disable_human: bool = True
    device: str | None = None
    backend: Backend = "stub"


class RunMetrics(BaseModel):
    episodes: int = 0
    success_rate: float = 0.0
    avg_total_reward: float = 0.0
    avg_task_reward: float = 0.0
    avg_cost_penalty: float = 0.0
    avg_energy_spent: float = 0.0
    avg_roi: float = 0.0
    avg_tokens: float = 0.0


class AssimilationMetrics(BaseModel):
    events: int = 0
    decisions_true: int = 0
    passed_true: int = 0


class BenchmarkCaseResult(BaseModel):
    name: str
    run_dir: Path
    config_path: Path
    backend: Backend
    metrics: RunMetrics
    assimilation: AssimilationMetrics
    population_size_final: int


class BenchmarkSuite(BaseModel):
    cases: list[BenchmarkCase]


class BenchmarkReport(BaseModel):
    started_at: datetime
    finished_at: datetime
    git_commit: str | None
    suite: BenchmarkSuite
    results: list[BenchmarkCaseResult]


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or None
        )
    except Exception:
        return None


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def _summarize_episodes(path: Path) -> RunMetrics:
    episodes = 0
    successes = 0
    totals: list[float] = []
    task_rewards: list[float] = []
    cost_penalties: list[float] = []
    energy_spent: list[float] = []
    rois: list[float] = []
    tokens: list[int] = []
    if not path.exists():
        return RunMetrics()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "episode":
                continue
            episodes += 1
            rewards = obj.get("rewards", {}) or {}
            try:
                task_reward = float(rewards.get("task_reward", 0.0))
                novelty = float(rewards.get("novelty_bonus", 0.0))
                competence = float(rewards.get("competence_bonus", 0.0))
                helper = float(rewards.get("helper_bonus", 0.0))
                risk = float(rewards.get("risk_penalty", 0.0))
                cost_penalty = float(rewards.get("cost_penalty", 0.0))
                total = task_reward + novelty + competence + helper - risk - cost_penalty
            except Exception:
                continue
            totals.append(total)
            task_rewards.append(task_reward)
            cost_penalties.append(cost_penalty)
            try:
                energy_spent.append(float(obj.get("energy_spent", 0.0)))
            except Exception:
                pass
            obs = obj.get("observations", {}) or {}
            if bool(obs.get("success", task_reward > 0.0)):
                successes += 1
            try:
                rois.append(float(obs.get("roi", 0.0)))
            except Exception:
                pass
            try:
                metrics = obs.get("metrics", {}) or {}
                tokens.append(int(metrics.get("tokens", 0)))
            except Exception:
                pass
    if episodes == 0:
        return RunMetrics()
    denom = float(episodes)
    return RunMetrics(
        episodes=episodes,
        success_rate=float(successes) / denom,
        avg_total_reward=sum(totals) / denom if totals else 0.0,
        avg_task_reward=sum(task_rewards) / denom if task_rewards else 0.0,
        avg_cost_penalty=sum(cost_penalties) / denom if cost_penalties else 0.0,
        avg_energy_spent=sum(energy_spent) / denom if energy_spent else 0.0,
        avg_roi=sum(rois) / denom if rois else 0.0,
        avg_tokens=sum(tokens) / denom if tokens else 0.0,
    )


def _summarize_assimilation(path: Path) -> AssimilationMetrics:
    if not path.exists():
        return AssimilationMetrics()
    events = 0
    decisions_true = 0
    passed_true = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "assimilation":
                continue
            events += 1
            if bool(obj.get("decision")):
                decisions_true += 1
            if bool(obj.get("passed")):
                passed_true += 1
    return AssimilationMetrics(
        events=events, decisions_true=decisions_true, passed_true=passed_true
    )


@contextmanager
def _patched_stub_backend() -> Iterator[None]:
    import symbiont_ecology.host.kernel as kernel_module
    import symbiont_ecology.organelles.peft_hebbian as peft_module

    original_backbone = getattr(kernel_module, "HFBackbone")  # noqa: B009
    original_organelle = getattr(peft_module, "HebbianPEFTOrganelle")  # noqa: B009
    setattr(kernel_module, "HFBackbone", BenchmarkStubBackbone)  # noqa: B010
    setattr(peft_module, "HebbianPEFTOrganelle", BenchmarkStubOrganelle)  # noqa: B010
    try:
        yield
    finally:
        setattr(kernel_module, "HFBackbone", original_backbone)  # noqa: B010
        setattr(peft_module, "HebbianPEFTOrganelle", original_organelle)  # noqa: B010


def _configure_assimilation(assim: AssimilationTester, config: object) -> None:
    try:
        tuning = getattr(config, "assimilation_tuning", None)
        if tuning is None:
            return
        assim.bootstrap_enabled = bool(getattr(tuning, "bootstrap_uplift_enabled", False))
        assim.bootstrap_n = int(getattr(tuning, "bootstrap_samples", 0))
        assim.permutation_n = int(getattr(tuning, "permutation_samples", 0))
        assim.min_samples = int(getattr(tuning, "min_uplift_samples", 2))
        assim.dr_enabled = bool(getattr(tuning, "dr_enabled", False))
        assim.dr_strata = list(getattr(tuning, "dr_strata", assim.dr_strata))
        assim.dr_min_stratum = int(getattr(tuning, "dr_min_stratum_size", assim.dr_min_stratum))
        assim.dr_min_power = float(getattr(tuning, "dr_min_power", assim.dr_min_power))
    except Exception:
        return


def run_benchmark_suite(suite: BenchmarkSuite, output_root: Path) -> BenchmarkReport:
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(tz=timezone.utc)
    results: list[BenchmarkCaseResult] = []
    commit = _git_commit()

    for case in suite.cases:
        run_dir = output_root / case.name
        run_dir.mkdir(parents=True, exist_ok=True)
        _set_seeds(case.seed)

        ctx = _patched_stub_backend() if case.backend == "stub" else nullcontext()
        with ctx:
            config = load_ecology_config(case.config_path)
            config.metrics.root = run_dir
            config.metrics.root.mkdir(parents=True, exist_ok=True)
            config.environment.synthetic_batch_size = int(case.batch_size)
            if case.disable_human:
                config.human_bandit.enabled = False
            if case.device is not None:
                config.host.device = case.device
            elif case.backend == "stub":
                config.host.device = "cpu"

            # Prefer deterministic decoding unless an experiment explicitly opts out.
            config.host.temperature = 0.0
            config.host.top_p = 1.0
            config.host.team_probe_temperature = 0.0
            config.host.team_probe_top_p = 1.0

            ledger = ATPLedger()
            router = BanditRouter()
            host = HostKernel(config=config, router=router, ledger=ledger)
            host.freeze_host()

            population = PopulationManager(config.evolution, config.foraging)
            initial_orgs = int(getattr(config.population_strategy, "initial_orgs", 4))
            for _ in range(max(1, initial_orgs)):
                oid = host.spawn_organelle(rank=config.host.max_lora_rank)
                population.register(
                    Genome(
                        organelle_id=oid,
                        drive_weights={"novelty": 0.4},
                        gate_bias=0.0,
                        rank=config.host.max_lora_rank,
                    )
                )

            assim = AssimilationTester(
                uplift_threshold=config.evolution.assimilation_threshold,
                p_value_threshold=config.evolution.assimilation_p_value,
                safety_budget=0,
            )
            _configure_assimilation(assim, config)

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
                seed=case.seed,
                reward_bonus=config.environment.success_reward_bonus,
                failure_cost_multiplier=config.environment.failure_cost_multiplier,
                lp_alpha=getattr(config.curriculum, "lp_alpha", 0.5),
            )
            human_bandit = None
            if not case.disable_human and bool(getattr(config.human_bandit, "enabled", False)):
                from symbiont_ecology.environment.human import HumanBandit

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
            for _ in range(int(case.generations)):
                loop.run_generation(batch_size=int(case.batch_size))

        episodes_path = run_dir / config.metrics.episodes_file
        assim_path = run_dir / config.metrics.assimilation_file
        results.append(
            BenchmarkCaseResult(
                name=case.name,
                run_dir=run_dir,
                config_path=case.config_path,
                backend=case.backend,
                metrics=_summarize_episodes(episodes_path),
                assimilation=_summarize_assimilation(assim_path),
                population_size_final=len(population.population),
            )
        )

    finished = datetime.now(tz=timezone.utc)
    return BenchmarkReport(
        started_at=started,
        finished_at=finished,
        git_commit=commit,
        suite=suite,
        results=results,
    )
