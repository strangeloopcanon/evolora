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
from statistics import mean, pstdev
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
    seeds: list[int] | None = None
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


class OpenEndednessMetrics(BaseModel):
    merges_total: int = 0
    merges_per_generation: float = 0.0
    qd_archive_size_final: int = 0
    qd_archive_coverage_final: float = 0.0
    qd_archive_coverage_max: float = 0.0
    colonies_final: int = 0
    colonies_max: int = 0
    diversity_energy_gini_final: float = 0.0
    diversity_effective_population_final: float = 0.0
    diversity_max_species_share_final: float = 0.0


class BenchmarkReplicateResult(BaseModel):
    seed: int
    run_dir: Path
    metrics: RunMetrics
    assimilation: AssimilationMetrics
    open_endedness: OpenEndednessMetrics
    population_size_final: int


class BenchmarkCaseResult(BaseModel):
    name: str
    run_dir: Path
    config_path: Path
    backend: Backend
    metrics: RunMetrics
    metrics_std: RunMetrics = Field(default_factory=RunMetrics)
    assimilation: AssimilationMetrics
    assimilation_std: AssimilationMetrics = Field(default_factory=AssimilationMetrics)
    open_endedness: OpenEndednessMetrics = Field(default_factory=OpenEndednessMetrics)
    open_endedness_std: OpenEndednessMetrics = Field(default_factory=OpenEndednessMetrics)
    population_size_final: int
    population_size_final_mean: float = 0.0
    population_size_final_std: float = 0.0
    replicates: list[BenchmarkReplicateResult] = Field(default_factory=list)


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


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _summarize_open_endedness(
    summaries: list[dict[str, object]], generations: int
) -> OpenEndednessMetrics:
    merges_total = 0
    qd_archive_size_final = 0
    qd_archive_coverage_final = 0.0
    qd_archive_coverage_max = 0.0
    colonies_final = 0
    colonies_max = 0
    diversity_energy_gini_final = 0.0
    diversity_effective_population_final = 0.0
    diversity_max_species_share_final = 0.0

    for summary in summaries:
        merges_total += _safe_int(summary.get("merges", 0))
        colonies_final = _safe_int(summary.get("colonies", colonies_final), colonies_final)
        colonies_max = max(colonies_max, colonies_final)

        qd_archive_size_final = _safe_int(
            summary.get("qd_archive_size", qd_archive_size_final), qd_archive_size_final
        )
        coverage = summary.get("qd_archive_coverage")
        if coverage is None:
            coverage = summary.get("qd_coverage_ratio")
        qd_archive_coverage_final = _safe_float(coverage, qd_archive_coverage_final)
        qd_archive_coverage_max = max(qd_archive_coverage_max, qd_archive_coverage_final)

        diversity = summary.get("diversity")
        if isinstance(diversity, dict):
            diversity_energy_gini_final = _safe_float(
                diversity.get("energy_gini"), diversity_energy_gini_final
            )
            diversity_effective_population_final = _safe_float(
                diversity.get("effective_population"), diversity_effective_population_final
            )
            diversity_max_species_share_final = _safe_float(
                diversity.get("max_species_share"), diversity_max_species_share_final
            )

    return OpenEndednessMetrics(
        merges_total=merges_total,
        merges_per_generation=float(merges_total) / max(generations, 1),
        qd_archive_size_final=qd_archive_size_final,
        qd_archive_coverage_final=qd_archive_coverage_final,
        qd_archive_coverage_max=qd_archive_coverage_max,
        colonies_final=colonies_final,
        colonies_max=colonies_max,
        diversity_energy_gini_final=diversity_energy_gini_final,
        diversity_effective_population_final=diversity_effective_population_final,
        diversity_max_species_share_final=diversity_max_species_share_final,
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def _aggregate_metrics(replicates: list[BenchmarkReplicateResult]) -> tuple[RunMetrics, RunMetrics]:
    episodes = [float(r.metrics.episodes) for r in replicates]
    success = [float(r.metrics.success_rate) for r in replicates]
    total_reward = [float(r.metrics.avg_total_reward) for r in replicates]
    task_reward = [float(r.metrics.avg_task_reward) for r in replicates]
    cost_penalty = [float(r.metrics.avg_cost_penalty) for r in replicates]
    energy_spent = [float(r.metrics.avg_energy_spent) for r in replicates]
    roi = [float(r.metrics.avg_roi) for r in replicates]
    tokens = [float(r.metrics.avg_tokens) for r in replicates]

    e_mean, e_std = _mean_std(episodes)
    s_mean, s_std = _mean_std(success)
    tr_mean, tr_std = _mean_std(total_reward)
    task_mean, task_std = _mean_std(task_reward)
    cp_mean, cp_std = _mean_std(cost_penalty)
    es_mean, es_std = _mean_std(energy_spent)
    roi_mean, roi_std = _mean_std(roi)
    tok_mean, tok_std = _mean_std(tokens)

    return (
        RunMetrics(
            episodes=int(round(e_mean)),
            success_rate=s_mean,
            avg_total_reward=tr_mean,
            avg_task_reward=task_mean,
            avg_cost_penalty=cp_mean,
            avg_energy_spent=es_mean,
            avg_roi=roi_mean,
            avg_tokens=tok_mean,
        ),
        RunMetrics(
            episodes=int(round(e_std)),
            success_rate=s_std,
            avg_total_reward=tr_std,
            avg_task_reward=task_std,
            avg_cost_penalty=cp_std,
            avg_energy_spent=es_std,
            avg_roi=roi_std,
            avg_tokens=tok_std,
        ),
    )


def _aggregate_assimilation(
    replicates: list[BenchmarkReplicateResult],
) -> tuple[AssimilationMetrics, AssimilationMetrics]:
    events = [float(r.assimilation.events) for r in replicates]
    decisions = [float(r.assimilation.decisions_true) for r in replicates]
    passed = [float(r.assimilation.passed_true) for r in replicates]

    e_mean, e_std = _mean_std(events)
    d_mean, d_std = _mean_std(decisions)
    p_mean, p_std = _mean_std(passed)

    return (
        AssimilationMetrics(
            events=int(round(e_mean)),
            decisions_true=int(round(d_mean)),
            passed_true=int(round(p_mean)),
        ),
        AssimilationMetrics(
            events=int(round(e_std)),
            decisions_true=int(round(d_std)),
            passed_true=int(round(p_std)),
        ),
    )


def _aggregate_open_endedness(
    replicates: list[BenchmarkReplicateResult],
) -> tuple[OpenEndednessMetrics, OpenEndednessMetrics]:
    merges = [float(r.open_endedness.merges_total) for r in replicates]
    merges_per_gen = [float(r.open_endedness.merges_per_generation) for r in replicates]
    qd_size = [float(r.open_endedness.qd_archive_size_final) for r in replicates]
    qd_final = [float(r.open_endedness.qd_archive_coverage_final) for r in replicates]
    qd_max = [float(r.open_endedness.qd_archive_coverage_max) for r in replicates]
    colonies_final = [float(r.open_endedness.colonies_final) for r in replicates]
    colonies_max = [float(r.open_endedness.colonies_max) for r in replicates]
    gini = [float(r.open_endedness.diversity_energy_gini_final) for r in replicates]
    eff_pop = [float(r.open_endedness.diversity_effective_population_final) for r in replicates]
    max_share = [float(r.open_endedness.diversity_max_species_share_final) for r in replicates]

    merges_mean, merges_std = _mean_std(merges)
    mpg_mean, mpg_std = _mean_std(merges_per_gen)
    qd_size_mean, qd_size_std = _mean_std(qd_size)
    qd_final_mean, qd_final_std = _mean_std(qd_final)
    qd_max_mean, qd_max_std = _mean_std(qd_max)
    colonies_final_mean, colonies_final_std = _mean_std(colonies_final)
    colonies_max_mean, colonies_max_std = _mean_std(colonies_max)
    gini_mean, gini_std = _mean_std(gini)
    eff_mean, eff_std = _mean_std(eff_pop)
    share_mean, share_std = _mean_std(max_share)

    return (
        OpenEndednessMetrics(
            merges_total=int(round(merges_mean)),
            merges_per_generation=mpg_mean,
            qd_archive_size_final=int(round(qd_size_mean)),
            qd_archive_coverage_final=qd_final_mean,
            qd_archive_coverage_max=qd_max_mean,
            colonies_final=int(round(colonies_final_mean)),
            colonies_max=int(round(colonies_max_mean)),
            diversity_energy_gini_final=gini_mean,
            diversity_effective_population_final=eff_mean,
            diversity_max_species_share_final=share_mean,
        ),
        OpenEndednessMetrics(
            merges_total=int(round(merges_std)),
            merges_per_generation=mpg_std,
            qd_archive_size_final=int(round(qd_size_std)),
            qd_archive_coverage_final=qd_final_std,
            qd_archive_coverage_max=qd_max_std,
            colonies_final=int(round(colonies_final_std)),
            colonies_max=int(round(colonies_max_std)),
            diversity_energy_gini_final=gini_std,
            diversity_effective_population_final=eff_std,
            diversity_max_species_share_final=share_std,
        ),
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
        seeds = list(case.seeds) if case.seeds else [int(case.seed)]
        case_root = output_root / case.name
        case_root.mkdir(parents=True, exist_ok=True)

        replicate_results: list[BenchmarkReplicateResult] = []
        for seed in seeds:
            run_dir = case_root if len(seeds) == 1 else case_root / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _set_seeds(seed)

            generation_summaries: list[dict[str, object]] = []
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
                    seed=seed,
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
                    summary = loop.run_generation(batch_size=int(case.batch_size))
                    if isinstance(summary, dict):
                        generation_summaries.append(summary)

            episodes_path = run_dir / config.metrics.episodes_file
            assim_path = run_dir / config.metrics.assimilation_file
            replicate_results.append(
                BenchmarkReplicateResult(
                    seed=seed,
                    run_dir=run_dir,
                    metrics=_summarize_episodes(episodes_path),
                    assimilation=_summarize_assimilation(assim_path),
                    open_endedness=_summarize_open_endedness(
                        generation_summaries, generations=int(case.generations)
                    ),
                    population_size_final=len(population.population),
                )
            )

        metrics, metrics_std = _aggregate_metrics(replicate_results)
        assimilation, assimilation_std = _aggregate_assimilation(replicate_results)
        open_endedness, open_endedness_std = _aggregate_open_endedness(replicate_results)
        pop_mean, pop_std = _mean_std([float(r.population_size_final) for r in replicate_results])
        results.append(
            BenchmarkCaseResult(
                name=case.name,
                run_dir=case_root,
                config_path=case.config_path,
                backend=case.backend,
                metrics=metrics,
                metrics_std=metrics_std,
                assimilation=assimilation,
                assimilation_std=assimilation_std,
                open_endedness=open_endedness,
                open_endedness_std=open_endedness_std,
                population_size_final=int(round(pop_mean)),
                population_size_final_mean=pop_mean,
                population_size_final_std=pop_std,
                replicates=replicate_results,
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
