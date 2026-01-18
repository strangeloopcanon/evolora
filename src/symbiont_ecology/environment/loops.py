"""Outer training loops for the ecology."""

from __future__ import annotations

import math
import random
import textwrap
from dataclasses import dataclass, field
from statistics import mean, pstdev, quantiles
from typing import Dict, Optional, Tuple

import torch

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.economics.api import compute_route_cost
from symbiont_ecology.environment.colony_utils import colony_c2c_debit
from symbiont_ecology.environment.grid import GridEnvironment, GridKey, GridTask
from symbiont_ecology.environment.human import HumanBandit, HumanFeedbackResult
from symbiont_ecology.environment.policy import parse_policy_json
from symbiont_ecology.environment.stats import compute_mean_ci, power_proxy, team_accept
from symbiont_ecology.environment.telemetry_utils import sanitize_telemetry
from symbiont_ecology.evaluation import EvaluationManager
from symbiont_ecology.evaluation.manager import EvaluationTask
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.meta import MetaEvolver
from symbiont_ecology.evolution.morphogenesis import MorphogenesisController
from symbiont_ecology.evolution.population import Genome, PopulationManager
from symbiont_ecology.host.kernel import HostKernel, RouteMetrics
from symbiont_ecology.metrics.persistence import TelemetrySink
from symbiont_ecology.metrics.telemetry import AssimilationEvent, EpisodeLog, RewardBreakdown


@dataclass
class EnergyGuardBanditArm:
    weight: float
    floor: float = 0.0
    roi: float = 0.0
    pulls: int = 0
    reward: float = 0.0


class EnergyGuardBandit:
    def __init__(self, weights: list[float], seed: int = 4242) -> None:
        self.arms: list[EnergyGuardBanditArm] = [EnergyGuardBanditArm(weight=w) for w in weights]
        self.last_choice: int | None = None
        self.total_pulls = 0
        self.rng = random.Random(seed)
        self.last_reward: float = 0.0

    def record_reward(self, reward: float) -> None:
        if self.last_choice is None:
            return
        arm = self.arms[self.last_choice]
        arm.pulls += 1
        arm.reward += reward
        self.total_pulls += 1
        self.last_reward = reward
        self.last_choice = None

    def select(self, epsilon: float = 0.1) -> int:
        # ensure each arm is tried at least once
        for idx, arm in enumerate(self.arms):
            if arm.pulls == 0 and self.last_choice is None:
                self.last_choice = idx
                return idx
        if self.rng.random() < epsilon:
            choice = self.rng.randint(0, len(self.arms) - 1)
            self.last_choice = choice
            return choice
        total = max(1, self.total_pulls)
        best_idx = 0
        best_score = float("-inf")
        for idx, arm in enumerate(self.arms):
            avg = arm.reward / arm.pulls if arm.pulls > 0 else 0.0
            bonus = (
                math.sqrt(2.0 * math.log(total + 1) / arm.pulls) if arm.pulls > 0 else float("inf")
            )
            score = avg + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        self.last_choice = best_idx
        return best_idx


@dataclass
class EcologyLoop:
    config: EcologyConfig
    host: HostKernel
    environment: GridEnvironment
    population: PopulationManager
    assimilation: AssimilationTester
    human_bandit: Optional[HumanBandit] = None
    sink: Optional[TelemetrySink] = None
    logs: list[EpisodeLog] = field(default_factory=list)
    generation_index: int = 0
    assimilation_cooldown: Dict[Tuple[str, GridKey], int] = field(default_factory=dict)
    bankruptcy_strikes: Dict[str, int] = field(default_factory=dict)
    morphogenesis: MorphogenesisController | None = None
    meta_evolver: MetaEvolver | None = None
    evaluation_manager: EvaluationManager | None = None
    bandit_counter: int = 0
    holdout_rng: random.Random = field(default_factory=lambda: random.Random(7331))
    _holdout_tasks_cache: list[EvaluationTask] | None = field(default=None, init=False, repr=False)
    _diversity_snapshot: dict[str, float] | None = field(default=None, init=False, repr=False)
    no_merge_counter: int = 0
    assim_fail_streak: int = 0
    assim_gating_samples: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    assim_attempt_samples: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _pending_hints: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    trial_offspring: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )
    promotions_this_gen: int = 0
    trial_creations_this_gen: int = 0
    _qd_archive: dict[tuple[str, str, int], dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _synergy_window: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _synergy_sustain: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _team_probe_candidates_gen: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _team_gate_samples: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _team_gate_counts_gen: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    colonies: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    _lp_mix_history: list[float] = field(default_factory=list, init=False, repr=False)
    _last_lp_mix: float = field(default=0.0, init=False, repr=False)
    _base_lp_mix: float = field(default=0.0, init=False, repr=False)
    _tau_relief: dict[GridKey, float] = field(default_factory=dict, init=False, repr=False)
    _tau_fail_counts: dict[GridKey, int] = field(default_factory=dict, init=False, repr=False)
    _roi_relief: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _population_refresh_gen: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _topup_fail_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _active_policies: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _assim_attempt_total: int = field(default=0, init=False, repr=False)
    _policy_cost_total: float = field(default=0.0, init=False, repr=False)
    _policy_attempts_gen: int = field(default=0, init=False, repr=False)
    _policy_parsed_gen: int = field(default=0, init=False, repr=False)
    _policy_fail_counts: int = field(default=0, init=False, repr=False)
    _policy_failure_samples: list[dict[str, str]] = field(
        default_factory=list, init=False, repr=False
    )
    _co_routing_counts: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _reserve_state: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _hazard_state: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _hazard_cooldown: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _survival_events: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _comms_credit_queue: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _comms_events_history: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _knowledge_store: dict[str, list[dict[str, object]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _knowledge_stats_gen: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _budget_snapshot_gen: dict[str, object] | None = field(default=None, init=False, repr=False)
    _power_econ_stats: dict[str, float | int] = field(default_factory=dict, init=False, repr=False)
    _winter_active: bool = field(default=False, init=False, repr=False)
    _winter_timer: int = field(default=0, init=False, repr=False)
    _winter_counter: int = field(default=0, init=False, repr=False)
    _winter_price_factor: float = field(default=1.0, init=False, repr=False)
    _winter_ticket_multiplier: float = field(default=1.0, init=False, repr=False)
    _winter_pre_roi: float = field(default=0.0, init=False, repr=False)
    _winter_pre_assim_attempts: int = field(default=0, init=False, repr=False)
    _last_assim_attempts: int = field(default=0, init=False, repr=False)
    _colony_events_archive: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _colony_selection_pool: dict[str, object] = field(
        default_factory=lambda: {"members": [], "pot": 0.0, "events": []},
        init=False,
        repr=False,
    )
    _colony_selection_stats: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _mutation_stats_gen: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _merge_audits_gen: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if hasattr(self.environment, "message_board"):
            self.environment.comms_history_cap = max(
                1, int(getattr(self.config.comms, "history_cap", 64))
            )
            self.environment.default_comm_priority = float(
                getattr(self.config.comms, "default_priority", 0.0)
            )
        history_limit = getattr(self.config.assimilation_tuning, "assimilation_history_limit", None)
        if isinstance(history_limit, int) and history_limit > 0:
            self.population.assimilation_history_limit = int(history_limit)
        else:
            self.population.assimilation_history_limit = None
        self._mutation_stats_gen = {"rank_noise": 0, "dropout": 0, "duplication": 0}
        self._merge_audits_gen = []

    def run_generation(self, batch_size: int) -> dict[str, object]:
        config = self.config
        host = self.host
        population = self.population
        energy_cfg = config.energy

        self.generation_index += 1
        self.promotions_this_gen = 0
        self.trial_creations_this_gen = 0
        self._assim_attempt_total = 0
        self._policy_attempts_gen = 0
        self._policy_parsed_gen = 0
        self._policy_fail_counts = 0
        self._policy_failure_samples = []
        self._comms_stats_gen = {"posts": 0, "reads": 0, "credits": 0}
        self._comms_events_gen: list[dict[str, object]] = []
        self._budget_snapshot_gen = None
        self._power_econ_stats = {
            "episodes": 0,
            "power_sum": 0.0,
            "price_multiplier_sum": 0.0,
            "tokens_minted": 0,
            "tokens_used": 0,
            "info_topups": 0,
        }
        self._mutation_stats_gen = {"rank_noise": 0, "dropout": 0, "duplication": 0}
        self._merge_audits_gen = []
        self._knowledge_stats_gen = {
            "writes": 0,
            "write_denied": 0,
            "reads": 0,
            "read_denied": 0,
            "hits": 0,
            "expired": 0,
        }
        self._prune_knowledge_cache()
        # per-generation caps
        self._team_handoff_used = 0
        self._prompt_scaffold_counts = {}
        self._team_probe_candidates_gen = []
        self._winter_events_gen = []
        self._population_refresh_gen = {}
        pool = self._colony_selection_pool
        pool_members = list(pool.get("members", []))
        pool_pot = float(pool.get("pot", 0.0))
        self._colony_selection_stats = {
            "dissolved": 0,
            "replicated": 0,
            "pool_members": len(pool_members),
            "pool_pot": round(pool_pot, 4),
            "events": [],
        }
        if self.morphogenesis is None:
            self.morphogenesis = MorphogenesisController(config=config, host=host)
        if config.evaluation.enabled and self.evaluation_manager is None:
            runtime = EvaluationManager.from_file(
                enabled=config.evaluation.enabled,
                cadence=config.evaluation.cadence,
                tasks_path=config.evaluation.tasks_path,
                sample_size=config.evaluation.sample_size,
                reward_weight=config.evaluation.reward_weight,
            )
            self.evaluation_manager = EvaluationManager(runtime)
        organelle_ids = host.list_organelle_ids()
        if not organelle_ids:
            return {}

        self._update_winter_cycle()
        ticket_base = energy_cfg.m
        ticket = ticket_base * self._winter_ticket_multiplier
        active: list[str] = []
        bankrupt: list[str] = []
        grace = max(1, energy_cfg.bankruptcy_grace)
        for organelle_id in organelle_ids:
            balance = host.ledger.energy_balance(organelle_id)
            if balance < ticket:
                bankrupt.append(organelle_id)
                continue
            host.ledger.consume_energy(organelle_id, ticket)
            active.append(organelle_id)

        culled_bankrupt: list[str] = []
        bankrupt_set = set(bankrupt)
        for organelle_id in organelle_ids:
            if organelle_id in bankrupt_set:
                strikes = self.bankruptcy_strikes.get(organelle_id, 0) + 1
                self.bankruptcy_strikes[organelle_id] = strikes
            else:
                self.bankruptcy_strikes[organelle_id] = 0
        for organelle_id, strikes in list(self.bankruptcy_strikes.items()):
            if strikes >= grace:
                culled_bankrupt.append(organelle_id)
                self.bankruptcy_strikes.pop(organelle_id, None)
                if organelle_id in active:
                    active.remove(organelle_id)
                if organelle_id in bankrupt:
                    bankrupt.remove(organelle_id)
                self.environment.organism_stats.pop(organelle_id, None)
                for key in list(self.assimilation_cooldown.keys()):
                    if key[0] == organelle_id:
                        self.assimilation_cooldown.pop(key, None)
                host.retire_organelle(organelle_id)
                population.remove(organelle_id)

        if self._winter_active and culled_bankrupt:
            preview = culled_bankrupt[:5]
            self._winter_events_gen.append(
                {
                    "gen": self.generation_index,
                    "type": "winter_cull",
                    "count": len(culled_bankrupt),
                    "preview": list(preview),
                }
            )

        for organelle_id in bankrupt:
            population.record_score(organelle_id, 0.0)
            population.record_energy(organelle_id, 0.0)
            population.record_roi(organelle_id, 0.0)

        if self.colonies:
            try:
                self._apply_colony_bankruptcy_guard(culled_bankrupt)
            except Exception:
                pass

        # Prepare colony comms budgets (bandwidth + per-gen caps)
        if self.colonies:
            tune = config.assimilation_tuning
            bw_base = float(getattr(tune, "colony_bandwidth_base", 2.0))
            bw_frac = float(getattr(tune, "colony_bandwidth_frac", 0.02))
            hazard_scale = float(getattr(tune, "colony_hazard_bandwidth_scale", 0.3))
            post_cap_base = int(getattr(tune, "colony_post_cap", 2))
            read_cap_base = int(getattr(tune, "colony_read_cap", 3))
            post_cap_hazard = int(
                getattr(tune, "colony_post_cap_hazard", max(0, post_cap_base // 2))
            )
            if post_cap_hazard <= 0:
                post_cap_hazard = max(0, min(post_cap_base, 1))
            read_cap_hazard = int(
                getattr(tune, "colony_read_cap_hazard", max(0, read_cap_base // 2))
            )
            if read_cap_hazard <= 0:
                read_cap_hazard = max(0, min(read_cap_base, 1))
            hazard_state = self._hazard_state
            for _cid, meta in self.colonies.items():
                members = [str(x) for x in meta.get("members", [])]
                pot = max(0.0, float(meta.get("pot", 0.0)))
                hazard_members = sum(
                    1 for m in members if bool(hazard_state.get(m, {}).get("active"))
                )
                hazard_active = hazard_members > 0
                bandwidth = min(bw_base, pot * bw_frac)
                if hazard_active:
                    bandwidth *= hazard_scale
                bandwidth = max(0.0, bandwidth)
                posts_cap = post_cap_hazard if hazard_active else post_cap_base
                reads_cap = read_cap_hazard if hazard_active else read_cap_base
                reserve_active = bool(meta.get("reserve_active"))
                if reserve_active:
                    reserve_scale = float(getattr(tune, "colony_reserve_bandwidth_scale", 0.3))
                    bandwidth *= max(0.0, min(1.0, reserve_scale))
                    posts_cap = min(
                        posts_cap, int(getattr(tune, "colony_reserve_post_cap", posts_cap))
                    )
                    reads_cap = min(
                        reads_cap, int(getattr(tune, "colony_reserve_read_cap", reads_cap))
                    )
                winter_mode = bool(meta.get("winter_mode"))
                if winter_mode:
                    winter_scale = float(getattr(tune, "colony_winter_bandwidth_scale", 0.1))
                    bandwidth *= max(0.0, min(1.0, winter_scale))
                    posts_cap = min(
                        posts_cap, int(getattr(tune, "colony_winter_post_cap", posts_cap))
                    )
                    reads_cap = min(
                        reads_cap, int(getattr(tune, "colony_winter_read_cap", reads_cap))
                    )
                tier = int(meta.get("tier", 0))
                if tier > 0:
                    boost = float(getattr(tune, "colony_tier_bandwidth_boost", 0.0))
                    if boost > 0.0:
                        scale = 1.0 + tier * boost
                        bandwidth *= scale
                        posts_cap = max(1, int(round(posts_cap * scale)))
                        reads_cap = max(1, int(round(reads_cap * scale)))
                meta["bandwidth_left"] = bandwidth
                meta["posts_left"] = posts_cap
                meta["reads_left"] = reads_cap
                meta["bandwidth_budget"] = bandwidth
                meta["posts_budget"] = posts_cap
                meta["reads_budget"] = reads_cap
                meta["hazard_members"] = hazard_members
                # C2C latent comms share the same budget unless overridden later
                meta["c2c_bandwidth_left"] = bandwidth
                meta["c2c_posts_left"] = posts_cap
                meta["c2c_reads_left"] = reads_cap
                self._log_colony_event(
                    meta,
                    self.generation_index,
                    "bandwidth",
                    budget=float(bandwidth),
                    posts=int(posts_cap),
                    reads=int(reads_cap),
                    hazard=int(hazard_members),
                    pot=float(pot),
                )

        try:
            self._update_survival_states(active)
        except Exception:
            pass

        # Optional: organism policy step (JSON intents)
        if bool(getattr(self.config.policy, "enabled", False)):
            for organelle_id in active:
                try:
                    self._request_and_apply_policy(organelle_id)
                except Exception:
                    pass

        # Lightweight co‑routing probes to inform team pairing
        try:
            self._probe_co_routing(active)
        except Exception:
            pass

        # Optional: communication read step once per active organelle
        comms_enabled = getattr(self.config.comms, "enabled", False)
        post_cost = float(getattr(self.config.comms, "post_cost", 0.2))
        read_cost = float(getattr(self.config.comms, "read_cost", 0.1))
        credit_frac = float(getattr(self.config.comms, "credit_frac", 0.2))
        # C2C latent comms
        c2c_enabled = bool(getattr(self.config.comms, "c2c_enabled", False))
        c2c_post_cost = float(getattr(self.config.comms, "c2c_post_cost", 0.1))
        c2c_read_cost = float(getattr(self.config.comms, "c2c_read_cost", 0.05))
        c2c_ttl = int(getattr(self.config.comms, "c2c_ttl", 5))
        c2c_mix = float(getattr(self.config.comms, "c2c_mix", 0.5))
        self._pending_latents: dict[str, list[list[float]]] = {}
        if comms_enabled:
            read_gen_cap = int(getattr(self.config.comms, "read_gen_cap", len(active) * 2))
            reads_used = 0
            for organelle_id in active:
                if reads_used >= read_gen_cap:
                    break
                genome = self.population.population.get(organelle_id)
                trait = 0.0
                if genome is not None and isinstance(getattr(genome, "read_rate", 0.0), float):
                    trait = max(0.0, min(1.0, float(genome.read_rate)))
                max_attempts = max(0, int(round(2 * trait)))
                if max_attempts <= 0:
                    continue
                remaining_budget = max(0, read_gen_cap - reads_used)
                if remaining_budget <= 0:
                    break
                member_colony = None
                if self.colonies:
                    for cid, meta in self.colonies.items():
                        if organelle_id in meta.get("members", []):
                            member_colony = (cid, meta)
                            break
                colony_reads_left = remaining_budget
                if member_colony is not None:
                    _cid, _meta = member_colony
                    colony_reads_left = min(
                        colony_reads_left, int(_meta.get("reads_left", remaining_budget))
                    )
                max_reads = min(colony_reads_left, max_attempts, remaining_budget)
                if max_reads <= 0:
                    continue
                preferred_topics: list[str] | None = None
                cell_topic = None
                if hasattr(self.environment, "best_cell_score"):
                    try:
                        cell_info = self.environment.best_cell_score(organelle_id)
                        if isinstance(cell_info, tuple) and cell_info[0]:
                            cell_topic = f"{cell_info[0][0]}:{cell_info[0][1]}"
                    except Exception:
                        cell_topic = None
                if cell_topic:
                    preferred_topics = [cell_topic]
                messages = self.environment.read_messages(
                    max_items=max_reads,
                    topics=preferred_topics,
                    exclude={organelle_id},
                    reader=organelle_id,
                )
                for msg in messages:
                    if reads_used >= read_gen_cap:
                        break
                    bal = self.host.ledger.energy_balance(organelle_id)
                    colony_ok = True
                    if member_colony is not None:
                        _cid, _meta = member_colony
                        colony_ok = (
                            float(_meta.get("bandwidth_left", 0.0)) >= read_cost
                            and int(_meta.get("reads_left", 1)) > 0
                        )
                    if bal < read_cost or not colony_ok:
                        continue
                    try:
                        self.host.ledger.consume_energy(organelle_id, read_cost)
                    except Exception:
                        continue
                    if member_colony is not None:
                        _cid, _meta = member_colony
                        _meta["bandwidth_left"] = max(
                            0.0, float(_meta.get("bandwidth_left", 0.0)) - read_cost
                        )
                        _meta["reads_left"] = max(0, int(_meta.get("reads_left", 1)) - 1)
                    reads_used += 1
                    self._comms_stats_gen["reads"] = self._comms_stats_gen.get("reads", 0) + 1
                    poster = msg.get("organelle_id") if isinstance(msg, dict) else None
                    poster_id = str(poster) if isinstance(poster, str) else None
                    baseline = self._estimate_power(organelle_id)
                    self._queue_comms_credit(
                        poster_id, organelle_id, baseline, credit_frac * read_cost
                    )
                    self._log_comms_event(
                        "read",
                        reader=organelle_id,
                        poster=poster_id,
                        topic=msg.get("topic"),
                        priority=msg.get("priority"),
                    )
                    genome = self.population.population.get(organelle_id)
                    if genome is not None:
                        genome.gate_bias += 0.01
                    hint_text = ""
                    if isinstance(msg, dict):
                        hint_text = str(msg.get("text", "")).strip()
                    if hint_text:
                        bucket = self._pending_hints.setdefault(organelle_id, [])
                        if len(bucket) < 3:
                            bucket.append(hint_text)
                if c2c_enabled:  # pragma: no cover - integration exercised in long runs
                    self._consume_c2c_latents(organelle_id, member_colony, c2c_read_cost)

        # endogenous batch size option
        bs = self._compute_batch_size(batch_size)
        base_lp_mix = float(getattr(self.config.curriculum, "lp_mix", 0.0))
        self._base_lp_mix = base_lp_mix
        lp_mix_value = self._resolve_lp_mix(base_lp_mix)
        team_enabled = bool(getattr(self.config.assimilation_tuning, "team_router_enabled", False))
        team_max = int(getattr(self.config.assimilation_tuning, "team_max_routes_per_gen", 0))
        team_used = 0
        used_pairs: set[tuple[str, str]] = set()
        budget_map, budget_meta = self._compute_budget_map(active, bs)
        global_cap_value = int(budget_meta.get("global_cap", 0))
        if budget_map:
            capped_total = int(budget_meta.get("capped_total", sum(budget_map.values())))
        else:
            capped_total = len(active) * bs
        global_left: int | None = capped_total if global_cap_value > 0 else None
        executed_budget: dict[str, int] = {}
        boost_record: dict[str, int] = {}
        # Evidence boost budget
        boost_cap = int(getattr(self.config.assimilation_tuning, "evidence_boost_cap", 0))
        boost_left = boost_cap
        for organelle_id in active:
            per_org_bs = int(budget_map.get(organelle_id, bs)) if budget_map else bs
            # Evidence boost for near-threshold candidates
            try:
                if boost_left > 0 and self._should_boost(organelle_id):
                    inc = int(getattr(self.config.assimilation_tuning, "evidence_boost_factor", 1))
                    inc = min(inc, boost_left)
                    per_org_bs += inc
                    boost_left = max(0, boost_left - inc)
                    if inc > 0:
                        boost_record[organelle_id] = boost_record.get(organelle_id, 0) + inc
            except Exception:
                pass
            try:
                per_org_bs = self._resolve_per_org_batch(organelle_id, per_org_bs)
            except Exception:
                per_org_bs = max(0, per_org_bs)
            per_org_bs = max(0, per_org_bs)
            if global_left is not None:
                allowed = min(per_org_bs, max(0, global_left))
            else:
                allowed = per_org_bs
            if global_left is not None:
                global_left = max(0, global_left - allowed)
            if allowed <= 0:
                executed_budget[organelle_id] = 0
                continue
            for _ in range(allowed):
                task = self._sample_task_for_org(organelle_id, lp_mix_value)
                # Optional team routing: pick a synergy pair and run team episode
                if team_enabled and team_used < team_max and len(self.population.population) >= 2:
                    pair = self._select_synergy_pair(
                        fallback_organelle=organelle_id, excluded=used_pairs
                    )
                    if pair is not None and pair not in used_pairs:
                        if self._run_team_episode(pair, task):
                            team_used += 1
                            used_pairs.add(pair)
                            continue
                # apply any pending hints to the prompt
                prompt_text = task.prompt
                hints = self._pending_hints.get(organelle_id, [])
                if hints:
                    joined = "; ".join(hints)
                    prompt_text = f"Hints: {joined}\n\n{task.prompt}"
                    # clear after use
                    self._pending_hints[organelle_id] = []
                knowledge_block = self._prepare_knowledge_prompt(organelle_id, task)
                if knowledge_block:
                    prompt_text = f"{knowledge_block}\n\n{prompt_text}"
                # optional C2C latent prefix consumption
                latent_prefix = None
                if c2c_enabled and self._pending_latents.get(organelle_id):
                    latent_prefix = self._pending_latents[organelle_id].pop(0)
                prompt_text, scaffold_applied = self._apply_prompt_scaffold(task, prompt_text)
                if scaffold_applied:
                    self._prompt_scaffold_counts[task.family] = (
                        self._prompt_scaffold_counts.get(task.family, 0) + 1
                    )
                result = self.host.step(
                    prompt=prompt_text,
                    intent="solve task",
                    max_routes=1,
                    allowed_organelle_ids=[organelle_id],
                    latent_prefix=latent_prefix,
                    latent_mix=c2c_mix,
                )
                metrics = result.responses.get(organelle_id)
                if metrics is None:
                    continue
                success, reward = task.evaluate(metrics.answer)
                if success:
                    self._record_knowledge_entry(organelle_id, task, metrics)
                responses = {organelle_id: (metrics.answer, float(metrics.tokens))}
                human_feedback = self._collect_human_feedback(task.prompt, responses)
                if human_feedback:
                    blended = self._blend_rewards({organelle_id: reward}, human_feedback)
                    reward = blended.get(organelle_id, reward)
                self.environment.register_result(organelle_id, task, success)
                settlement = self._settle_episode(organelle_id, task, reward, metrics)
                reward = reward.model_copy(update={"cost_penalty": settlement["cost"]})
                task_cell_family, task_cell_depth = task.cell
                meta = {
                    "family": task.family,
                    "depth": task.depth,
                    "task_id": getattr(task, "task_id", None),
                    "cell_family": task_cell_family,
                    "cell_depth": task_cell_depth,
                    "price": float(getattr(task, "price", 0.0)),
                    "difficulty": float(getattr(task, "difficulty", 0.0)),
                    "success": bool(success),
                    "generation": self.generation_index,
                }
                self.population.record_score(organelle_id, reward.total, meta=meta)
                self.population.record_energy(organelle_id, settlement["cost"])
                self.population.record_roi(organelle_id, settlement["roi"])
                self.population.record_adapter_usage(
                    organelle_id, metrics.active_adapters, metrics.tokens
                )
                utilisation_snapshot = {
                    module: self.population.average_adapter_usage(organelle_id, module)
                    for module in metrics.active_adapters
                    if module not in {"rank", "total"}
                }
                self._record_episode(
                    task, organelle_id, reward, metrics, settlement, success, utilisation_snapshot
                )
                self.host.apply_reward(result.envelope, {organelle_id: reward})
                # C2C: post latent capsule from this step
                if c2c_enabled:  # pragma: no cover - integration exercised in long runs
                    latent = result.envelope.observation.state.get("latent")
                    member_colony_post = None
                    if self.colonies:
                        for cid, meta in self.colonies.items():
                            if organelle_id in meta.get("members", []):
                                member_colony_post = (cid, meta)
                                break
                    self._post_c2c_latent(
                        organelle_id, latent, c2c_post_cost, c2c_ttl, member_colony_post
                    )
            executed_budget[organelle_id] = allowed
        budget_meta["final"] = executed_budget
        budget_meta["final_total"] = int(sum(executed_budget.values()))
        budget_meta["boost_applied"] = boost_record
        budget_meta["cap_exhausted"] = bool(global_left == 0) if global_left is not None else False
        self._budget_snapshot_gen = budget_meta

        self._enforce_diversity()
        if comms_enabled and active:
            post_gen_cap = int(getattr(self.config.comms, "post_gen_cap", 1))
            posts_used = 0
            rng = getattr(self.environment, "rng", None)
            if rng is None:
                rng = random.Random(self.generation_index)
            candidates = sorted(
                active, key=lambda oid: self.population.average_roi(oid, limit=5), reverse=True
            )
            for organelle_id in candidates:
                if posts_used >= post_gen_cap:
                    break
                genome = self.population.population.get(organelle_id)
                trait = 0.0
                if genome is not None and isinstance(getattr(genome, "post_rate", 0.0), float):
                    trait = max(0.0, min(1.0, float(genome.post_rate)))
                if trait <= 0.0:
                    continue
                if rng.random() > trait:
                    continue
                colony_ok = True
                member_colony = None
                if self.colonies:
                    for cid, meta in self.colonies.items():
                        if organelle_id in meta.get("members", []):
                            member_colony = (cid, meta)
                            break
                if member_colony is not None:
                    _cid, _meta = member_colony
                    colony_ok = (
                        float(_meta.get("bandwidth_left", 0.0)) >= post_cost
                        and int(_meta.get("posts_left", 1)) > 0
                    )
                if not colony_ok:
                    continue
                if self.host.ledger.energy_balance(organelle_id) < post_cost:
                    continue
                try:
                    self.host.ledger.consume_energy(organelle_id, post_cost)
                except Exception:
                    continue
                hint = ""
                topic = None
                cell_meta = None
                if hasattr(self.environment, "best_cell_score"):
                    try:
                        cell_score = self.environment.best_cell_score(organelle_id)
                        if isinstance(cell_score, tuple) and cell_score[0]:
                            cell_meta = cell_score[0]
                            topic = f"{cell_meta[0]}:{cell_meta[1]}"
                    except Exception:
                        topic = None
                hint = self._build_comms_hint(organelle_id, cell_meta)
                if not hint or not hint.strip():
                    continue
                priority = float(trait)
                try:
                    self.environment.post_message(
                        organelle_id,
                        hint,
                        cost=post_cost,
                        ttl=int(getattr(self.config.comms, "ttl", 10)),
                        priority=priority,
                        topic=topic,
                        cell=cell_meta,
                        meta={"source": "auto_hint"},
                    )
                except Exception:
                    continue
                if member_colony is not None:
                    _cid, _meta = member_colony
                    _meta["bandwidth_left"] = max(
                        0.0, float(_meta.get("bandwidth_left", 0.0)) - post_cost
                    )
                    _meta["posts_left"] = max(0, int(_meta.get("posts_left", 1)) - 1)
                posts_used += 1
                self._comms_stats_gen["posts"] = self._comms_stats_gen.get("posts", 0) + 1
                colony_id = member_colony[0] if member_colony is not None else None
                self._log_comms_event(
                    "post", poster=organelle_id, colony=colony_id, topic=topic, priority=priority
                )
        merges = self._attempt_assimilation(capped=self.config.evolution.max_merges_per_gen)
        # review trial offspring for potential promotion or cull
        self._review_trial_offspring()
        # Optional: team probes to promote colonies via CI gate
        try:
            team_promotions = self._maybe_team_probes()
        except Exception:
            team_promotions = 0
        # Colonies: promotion and upkeep (pooled pot, reserve top-ups)
        try:
            if bool(getattr(self.config.assimilation_tuning, "colonies_enabled", False)):
                self._maybe_promote_colonies()
                self._tick_colonies()
                self._colony_selection_step()
                self._colony_tier_migration()
        except Exception:
            pass
        if comms_enabled:
            try:
                self._process_comms_credit()
            except Exception:
                pass
            self._comms_events_history.append(
                {
                    "generation": self.generation_index,
                    "posts": int(self._comms_stats_gen.get("posts", 0)),
                    "reads": int(self._comms_stats_gen.get("reads", 0)),
                    "credits": int(self._comms_stats_gen.get("credits", 0)),
                    "events": list(self._comms_events_gen),
                }
            )
            if len(self._comms_events_history) > 120:
                self._comms_events_history = self._comms_events_history[-120:]
        if merges > 0:
            self.no_merge_counter = 0
        else:
            self.no_merge_counter += 1
        viability_map = self._compute_viability_map()
        survivors = self._mu_lambda_selection(viability_map)
        self._apply_morphogenesis(survivors)
        self._spawn_offspring(survivors)
        self._maybe_refresh_population(survivors)
        self.population.increment_ages()
        summary = {
            "active": len(active),
            "bankrupt": len(bankrupt),
            "culled_bankrupt": len(culled_bankrupt),
            "merges": merges + self.promotions_this_gen,
            "team_promotions": int(team_promotions),
            "population": len(self.population.population),
            "team_routes": int(team_used),
            "avg_roi": self.population.aggregate_roi(),
            "avg_energy_cost": self.population.aggregate_energy(),
            "cells": {
                f"{family}:{depth}": {
                    "difficulty": state.difficulty,
                    "success_ema": state.success_ema,
                    "price": state.price,
                }
                for (family, depth), state in self.environment.controller.cells.items()
            },
            "mutation_stats": dict(self._mutation_stats_gen),
        }
        if self._population_refresh_gen:
            summary["population_refresh"] = dict(self._population_refresh_gen)
        # Promotion acceptance governor (adjust team thresholds toward target)
        try:
            target = float(getattr(self.config.assimilation_tuning, "promotion_target_rate", 0.0))
            step = float(getattr(self.config.assimilation_tuning, "promotion_adjust_step", 0.0))
            if step > 0.0 and target > 0.0:
                observed_raw = float(team_promotions)
                # Normalize against 1.0 (aim ~target per gen)
                observed_norm = min(observed_raw, 1.0)
                # EMA smoothing
                alpha = float(getattr(self.config.assimilation_tuning, "promotion_ema_alpha", 0.3))
                prev = float(getattr(self, "_promotions_ema", observed_norm))
                smoothed = alpha * observed_norm + (1.0 - alpha) * prev
                self._promotions_ema = smoothed
                err = target - smoothed
                # Adjust team_holdout_margin and team_min_power gently
                margin = float(
                    getattr(
                        self.config.assimilation_tuning,
                        "team_holdout_margin",
                        getattr(self.config.assimilation_tuning, "holdout_margin", 0.02),
                    )
                )
                power = float(getattr(self.config.assimilation_tuning, "team_min_power", 0.2))
                margin_min = float(getattr(self.config.assimilation_tuning, "team_margin_min", 0.0))
                margin_max = float(getattr(self.config.assimilation_tuning, "team_margin_max", 0.1))
                power_min = float(getattr(self.config.assimilation_tuning, "team_power_min", 0.05))
                power_max = float(getattr(self.config.assimilation_tuning, "team_power_max", 0.5))
                # If under target, reduce margin and power; if over, increase
                margin = max(
                    margin_min, min(margin_max, margin - step * (1.0 if err > 0 else -1.0))
                )
                power = max(power_min, min(power_max, power - step * (1.0 if err > 0 else -1.0)))
                # Write back
                self.config.assimilation_tuning.team_holdout_margin = margin
                self.config.assimilation_tuning.team_min_power = power
                summary["promotion_controller"] = {
                    "team_holdout_margin": round(margin, 4),
                    "team_min_power": round(power, 4),
                    "target": target,
                    "observed": observed_raw,
                    "observed_smooth": round(smoothed, 3),
                }
        except Exception:
            pass
        # Top co-routing pairs this gen (best-effort)
        try:
            if self._co_routing_counts:
                top_pairs = sorted(
                    self._co_routing_counts.items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                summary["co_routing_top"] = {f"{a}:{b}": int(c) for (a, b), c in top_pairs}
                self._co_routing_counts = {}
        except Exception:
            pass
        # Learning progress snapshot per cell (for analyzer LP heatmaps)
        try:
            lp_map = {
                f"{family}:{depth}": float(
                    self.environment.controller.lp_progress.get((family, depth), 0.0)
                )
                for (family, depth) in self.environment.controller.cells.keys()
            }
            summary["lp_progress"] = lp_map
        except Exception:
            pass
        if bool(getattr(self.config.policy, "enabled", False)):
            summary["policy_applied"] = int(len(self._active_policies) > 0)
            if self._active_policies:
                # Aggregate simple field usage and a couple of numeric knobs
                field_counts: dict[str, int] = {}
                budgets: list[float] = []
                reserves: list[float] = []
                for pol in self._active_policies.values():
                    if isinstance(pol, dict):
                        for k in pol.keys():
                            field_counts[k] = field_counts.get(k, 0) + 1
                        bf = pol.get("budget_frac")
                        rr = pol.get("reserve_ratio")
                        if isinstance(bf, (int, float)):
                            budgets.append(float(bf))
                        if isinstance(rr, (int, float)):
                            reserves.append(float(rr))
                if field_counts:
                    summary["policy_fields_used"] = field_counts
                if budgets:
                    summary["policy_budget_frac_avg"] = float(sum(budgets) / max(1, len(budgets)))
                if reserves:
                    summary["policy_reserve_ratio_avg"] = float(
                        sum(reserves) / max(1, len(reserves))
                    )
                if getattr(self, "_policy_cost_total", 0.0) > 0.0:
                    summary["policy_cost_total"] = float(self._policy_cost_total)
            summary["policy_attempts"] = int(self._policy_attempts_gen)
            summary["policy_parsed"] = int(self._policy_parsed_gen)
            if getattr(self, "_policy_fail_counts", 0):
                summary["policy_failures"] = int(self._policy_fail_counts)
            if getattr(self, "_policy_failure_samples", None):
                summary["policy_failure_samples"] = list(self._policy_failure_samples[-5:])
        if getattr(self, "_prompt_scaffold_counts", None):
            summary["prompt_scaffolds"] = dict(self._prompt_scaffold_counts)
        # QD coverage (family-depth cells with observed stats)
        try:
            if getattr(self.config.qd, "enabled", False):
                total_bins = len(self.environment.controller.cells)
                populated = 0
                seen = set()
                for stats in self.environment.organism_stats.values():
                    for cell in stats.keys():
                        seen.add(cell)
                populated = len(seen)
                summary["qd_coverage"] = f"{populated}/{total_bins}"
                summary["qd_coverage_ratio"] = float(populated / max(total_bins, 1))
        except Exception:
            pass
        if hasattr(self, "assim_gating_counts"):
            summary["assimilation_gating"] = self.assim_gating_counts
        samples = getattr(self, "assim_gating_samples_snapshot", None)
        if samples:
            summary["assimilation_gating_samples"] = samples
        attempt_samples = getattr(self, "assim_attempt_samples_snapshot", None)
        if attempt_samples:
            summary["assimilation_attempts"] = attempt_samples
        if getattr(self, "_team_gate_counts_gen", None):
            summary["team_gate_counts"] = dict(self._team_gate_counts_gen)
        if self._team_gate_samples:
            summary["team_gate_samples"] = list(self._team_gate_samples[-24:])
        if self._merge_audits_gen:
            summary["merge_audits"] = list(self._merge_audits_gen)
        if self.population.assimilation_history:
            summary_limit = max(
                1,
                int(getattr(self.config.assimilation_tuning, "assimilation_history_summary", 6)),
            )
            history_snapshot: dict[str, list[dict[str, object]]] = {}
            for (
                organelle_id,
                family,
                depth,
            ), records in self.population.assimilation_history.items():
                if not records:
                    continue
                trimmed = records[-summary_limit:]
                history_snapshot[f"{organelle_id}:{family}:{depth}"] = [
                    self._sanitize_telemetry(entry) for entry in trimmed
                ]
            summary["assimilation_history"] = history_snapshot
        summary["roi_by_organelle"] = {
            organelle_id: float(self.population.average_roi(organelle_id, limit=5))
            for organelle_id in self.population.population
        }
        if getattr(self.config, "foraging", None) and getattr(
            self.config.foraging, "enabled", False
        ):
            top_k = int(getattr(self.config.foraging, "telemetry_top_k", 3))
            max_orgs = int(getattr(self.config.foraging, "telemetry_max_orgs", 10))
            ids_sorted = sorted(
                self.population.population.keys(),
                key=lambda oid: self.population.average_roi(oid, limit=5),
                reverse=True,
            )[: max(1, max_orgs)]
            trait_snapshot: dict[str, dict[str, float]] = {}
            top_cells_snapshot: dict[str, list[dict[str, float | str]]] = {}
            for oid in ids_sorted:
                genome = self.population.population.get(oid)
                if genome is None:
                    continue
                trait_snapshot[oid] = {
                    "beta": round(float(genome.beta_exploit), 3),
                    "decay": round(float(genome.q_decay), 3),
                    "ucb": round(float(genome.ucb_bonus), 3),
                    "budget": round(float(genome.budget_aggressiveness), 3),
                }
                top_cells = self.population.top_cells(oid, limit=top_k)
                if top_cells:
                    top_cells_snapshot[oid] = [
                        {"family": fam, "depth": depth, "q": round(val, 4)}
                        for fam, depth, val in top_cells
                    ]
            summary["foraging"] = {
                "traits": trait_snapshot,
                "top_cells": top_cells_snapshot,
            }
        summary["lp_mix_base"] = float(self._base_lp_mix)
        summary["lp_mix_active"] = float(self._last_lp_mix)
        if self.colonies:
            tier_counts: dict[int, int] = {}
            summary["colonies_meta"] = {
                cid: {
                    "members": list(meta.get("members", [])),
                    "pot": float(meta.get("pot", 0.0)),
                    "holdout_passes": int(meta.get("holdout_passes", 0)),
                    "holdout_failures": int(meta.get("holdout_failures", 0)),
                    "last_delta": float(meta.get("last_delta", 0.0)),
                    "reserve_active": bool(meta.get("reserve_active", False)),
                    "reserve_floor": float(meta.get("reserve_floor", 0.0)),
                    "winter_mode": bool(meta.get("winter_mode", False)),
                    "hazard_z": float(meta.get("hazard_z", 0.0)),
                    "variance_guard": bool(meta.get("variance_guard", False)),
                    "bandwidth_budget": float(
                        meta.get("bandwidth_budget", meta.get("bandwidth_left", 0.0))
                    ),
                    "hazard_members": int(meta.get("hazard_members", 0)),
                    "bankrupt_violations": int(meta.get("bankrupt_violations", 0)),
                    "roi_mean": float(meta.get("roi_mean", 0.0)),
                    "fitness": float(meta.get("fitness", 0.0)),
                    "tax_rate": float(
                        meta.get(
                            "tax_rate",
                            getattr(self.config.assimilation_tuning, "colony_tax_rate", 0.1),
                        )
                    ),
                    "subsidy_frac": float(
                        meta.get(
                            "subsidy_frac",
                            getattr(
                                self.config.assimilation_tuning, "colony_subsidy_fraction", 0.25
                            ),
                        )
                    ),
                    "events": list(meta.get("events", [])),
                    "tier": int(meta.get("tier", 0)),
                    "tier_cooldown": int(meta.get("tier_cooldown", 0)),
                }
                for cid, meta in self.colonies.items()
            }
            for meta in self.colonies.values():
                tier = int(meta.get("tier", 0))
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if self._colony_selection_stats:
                stats_copy = dict(self._colony_selection_stats)
                events = list(stats_copy.pop("events", []))
                summary["colony_selection"] = stats_copy
                if events:
                    summary["colony_selection_events"] = events[-5:]
                pool_events = list(self._colony_selection_pool.get("events", []))
                if pool_events:
                    summary["colony_selection_pool"] = pool_events[-5:]
            if tier_counts:
                summary["colony_tier_counts"] = {str(k): v for k, v in sorted(tier_counts.items())}
                total_colonies = sum(tier_counts.values())
                tier_mean = sum(level * count for level, count in tier_counts.items()) / max(
                    1, total_colonies
                )
                summary["colony_tier_mean"] = float(tier_mean)
        winter_block = {
            "active": bool(self._winter_active),
            "timer": int(self._winter_timer),
            "counter": int(self._winter_counter),
            "price_multiplier": float(self._winter_price_factor),
            "ticket_multiplier": float(self._winter_ticket_multiplier),
            "events": list(self._winter_events_gen),
        }
        summary["winter"] = winter_block
        if getattr(self, "_team_probe_candidates_gen", None):
            summary["team_probe_candidates"] = [
                {**{k: (list(v) if k == "pair" else v) for k, v in cand.items()}}
                for cand in self._team_probe_candidates_gen
            ]
        stats = getattr(self, "_power_econ_stats", None)
        if stats is not None and stats.get("episodes", 0):
            episodes = max(1, int(stats.get("episodes", 0)))
            summary["power_economics"] = {
                "episodes": episodes,
                "avg_power_need": float(stats.get("power_sum", 0.0)) / episodes,
                "avg_price_multiplier": float(stats.get("price_multiplier_sum", 0.0)) / episodes,
                "tokens_minted": int(stats.get("tokens_minted", 0)),
                "tokens_used": int(stats.get("tokens_used", 0)),
                "info_topups": int(stats.get("info_topups", 0)),
            }
        if comms_enabled:
            summary["comms"] = {
                "posts": int(self._comms_stats_gen.get("posts", 0)),
                "reads": int(self._comms_stats_gen.get("reads", 0)),
                "credits": int(self._comms_stats_gen.get("credits", 0)),
                "events": list(self._comms_events_gen),
            }
            if hasattr(self.environment, "peek_messages"):
                try:
                    summary["comms_board"] = self.environment.peek_messages(limit=5)
                except Exception:
                    summary["comms_board"] = []
        if self._knowledge_enabled():
            entries_total = sum(len(v) for v in self._knowledge_store.values())
            knowledge_block = dict(self._knowledge_stats_gen)
            knowledge_block["entries"] = entries_total
            summary["knowledge"] = knowledge_block
        # Learning KPI: ROI volatility (rolling std across organelles)
        try:
            roi_vals = list(summary["roi_by_organelle"].values())
            summary["roi_volatility"] = float(pstdev(roi_vals)) if len(roi_vals) >= 2 else 0.0
        except Exception:
            summary["roi_volatility"] = 0.0
        summary["energy_balance"] = {
            organelle_id: float(self.host.ledger.energy_balance(organelle_id))
            for organelle_id in self.population.population
        }
        summary["mean_energy_balance"] = (
            float(sum(summary["energy_balance"].values()) / max(len(summary["energy_balance"]), 1))
            if summary["energy_balance"]
            else 0.0
        )
        if self._budget_snapshot_gen is not None:
            summary["budget"] = dict(self._budget_snapshot_gen)
        if self._diversity_snapshot is not None:
            summary["diversity"] = dict(self._diversity_snapshot)
        summary["assimilation_fail_streak"] = self.assim_fail_streak
        summary["trial_offspring_active"] = len(self.trial_offspring)
        summary["trials_created"] = int(getattr(self, "trial_creations_this_gen", 0))
        summary["promotions"] = int(getattr(self, "promotions_this_gen", 0))
        if self.colonies:
            summary["colony_metrics"] = {
                cid: {
                    "size": len(meta.get("members", [])),
                    "pot": float(meta.get("pot", 0.0)),
                    "bandwidth_budget": float(meta.get("bandwidth_budget", 0.0)),
                    "posts_budget": int(meta.get("posts_budget", 0)),
                    "reads_budget": int(meta.get("reads_budget", 0)),
                    "last_delta": float(meta.get("last_delta", 0.0)),
                    "variance_ratio": float(meta.get("last_variance_ratio", 1.0)),
                    "hazard_members": int(meta.get("hazard_members", 0)),
                    "tier": int(meta.get("tier", 0)),
                    "tier_cooldown": int(meta.get("tier_cooldown", 0)),
                }
                for cid, meta in self.colonies.items()
            }
        summary["colony_events"] = list(self._colony_events_archive[-240:])
        # Update QD archive and expose size
        try:
            archive_size = int(self._update_qd_archive())
            summary["qd_archive_size"] = archive_size
            if getattr(self.config.qd, "enabled", False):
                total_bins = max(
                    1,
                    len(self.environment.controller.cells)
                    * max(1, int(getattr(self.config.qd, "cost_bins", 1))),
                )
                summary["qd_archive_coverage"] = float(archive_size / total_bins)
                top_entries = sorted(
                    self._qd_archive.items(),
                    key=lambda kv: float(kv[1].get("roi", 0.0)),
                    reverse=True,
                )[:5]
                summary["qd_archive_top"] = [
                    {
                        "cell": f"{key[0]}:{key[1]}",
                        "bin": int(key[2]),
                        "organelle": value.get("organelle_id"),
                        "roi": float(value.get("roi", 0.0)),
                        "ema": float(value.get("ema", 0.0)),
                        "novelty": float(value.get("novelty", 0.0)),
                    }
                    for key, value in top_entries
                ]
        except Exception:
            summary["qd_archive_size"] = 0
        self._auto_tune_assimilation_energy(summary)
        # Auto‑nudge evidence settings when assimilation stalls (no merges, low power)
        try:
            self._auto_nudge_evidence(summary)
        except Exception:
            pass
        # Ephemeral team synergy sampling (log only)
        try:
            self._sample_team_synergy()
            if self._synergy_window:
                summary["synergy_samples"] = list(self._synergy_window[-5:])
        except Exception:
            pass
        # Colonies: ensure count is present even if disabled
        summary["colonies"] = int(len(self.colonies))
        summary["assimilation_attempt_total"] = int(getattr(self, "_assim_attempt_total", 0))
        survival_snapshot = self._snapshot_survival_state()
        if survival_snapshot is not None:
            summary["survival"] = survival_snapshot
        if self._tau_relief:
            relief_snapshot = {
                f"{cell[0]}:{cell[1]}": round(float(relief), 3)
                for cell, relief in list(self._tau_relief.items())[:8]
                if relief > 0.0
            }
            if relief_snapshot:
                summary["tau_relief_active"] = relief_snapshot
        if self._roi_relief:
            roi_snapshot = {
                organelle: round(float(relief), 3)
                for organelle, relief in list(self._roi_relief.items())[:8]
                if relief > 0.0
            }
            if roi_snapshot:
                summary["roi_relief_active"] = roi_snapshot
        if (
            self.evaluation_manager
            and self.generation_index % self.evaluation_manager.config.cadence == 0
        ):
            evaluation_result = self.evaluation_manager.evaluate(self.host, self.environment)
            summary["evaluation"] = evaluation_result
            summary["evaluation"]["reward_weight"] = self.evaluation_manager.config.reward_weight
        if self.config.meta.enabled:
            if self.meta_evolver is None:
                self.meta_evolver = MetaEvolver(
                    config=self.config,
                    environment=self.environment,
                    assimilation=self.assimilation,
                )
            meta_info = self.meta_evolver.step(self.generation_index, summary["avg_roi"])
            summary.update(meta_info)
        self._last_assim_attempts = int(getattr(self, "_assim_attempt_total", 0))
        return summary

    def _compute_batch_size(self, default_bs: int) -> int:
        bs = default_bs
        try:
            if bool(getattr(self.config.environment, "auto_batch", False)):
                roi_sample = self.population.aggregate_roi(limit=5)
                bmin = int(max(1, getattr(self.config.environment, "batch_min", 1)))
                bmax = int(max(bmin, getattr(self.config.environment, "batch_max", 4)))
                if roi_sample <= 0.5:
                    bs = bmin
                elif roi_sample >= 1.5:
                    bs = bmax
                else:
                    frac = (roi_sample - 0.5) / 1.0
                    bs = int(max(bmin, min(bmax, round(bmin + frac * (bmax - bmin)))))
        except Exception:
            bs = default_bs
        return bs

    def _resolve_lp_mix(self, base_mix: float) -> float:
        cfg = self.config.curriculum
        lp_min = float(getattr(cfg, "lp_mix_min", base_mix))
        lp_max = float(getattr(cfg, "lp_mix_max", max(lp_min, base_mix)))
        lp_min = max(0.0, min(1.0, lp_min))
        lp_max = max(lp_min, min(1.0, lp_max))
        mix = max(lp_min, min(lp_max, base_mix))
        if bool(getattr(cfg, "alp_auto_mix", False)):
            progress_values = list(self.environment.controller.lp_progress.values())
            if progress_values:
                spread = max(progress_values) - min(progress_values)
                denom = max(max(progress_values), 1e-6)
                ratio = max(0.0, min(1.0, spread / denom))
                mix = lp_min + (lp_max - lp_min) * ratio
        window = max(1, int(getattr(cfg, "lp_window", 5)))
        self._lp_mix_history.append(mix)
        if len(self._lp_mix_history) > window:
            self._lp_mix_history = self._lp_mix_history[-window:]
        smoothed = sum(self._lp_mix_history) / len(self._lp_mix_history)
        self._last_lp_mix = max(lp_min, min(lp_max, smoothed))
        return self._last_lp_mix

    def _settle_episode(
        self,
        organelle_id: str,
        task: GridTask,
        reward: RewardBreakdown,
        metrics: RouteMetrics,
    ) -> Dict[str, float]:
        energy_before = self.host.ledger.energy_balance(organelle_id)
        price = task.price * self._winter_price_factor
        tuning = self.config.assimilation_tuning
        power_need = self._power_need(organelle_id)
        premium_alpha = float(getattr(tuning, "price_premium_alpha", 0.0))
        premium_cap = float(getattr(tuning, "price_premium_cap", 1.0))
        price_multiplier = 1.0
        if premium_alpha > 0.0:
            price_multiplier = 1.0 + premium_alpha * power_need
            price_multiplier = max(1.0, price_multiplier)
            price_multiplier = min(max(1.0, premium_cap), price_multiplier)
        config = self.config.energy
        revenue = (price * price_multiplier) * reward.total
        cost = compute_route_cost(config, metrics).total_cost
        if cost > 0:
            roi = revenue / max(cost, 1e-6)
        elif revenue > 0:
            roi = float("inf")
        else:
            roi = 0.0
        if not math.isfinite(roi):
            roi = 0.0
        else:
            roi = max(-10.0, min(roi, 10.0))
        delta = revenue - cost
        energy_after = max(0.0, min(self.host.ledger.energy_cap, energy_before + delta))
        self.host.ledger.set_energy(organelle_id, energy_after)
        self.population.record_energy_delta(organelle_id, delta)
        stats = self._power_econ_stats
        if stats is not None:
            stats["episodes"] = int(stats.get("episodes", 0)) + 1
            stats["power_sum"] = float(stats.get("power_sum", 0.0)) + power_need
            stats["price_multiplier_sum"] = (
                float(stats.get("price_multiplier_sum", 0.0)) + price_multiplier
            )
        settlement = {
            "energy_before": energy_before,
            "energy_after": energy_after,
            "revenue": revenue,
            "cost": cost,
            "roi": roi,
            "delta": delta,
            "price_multiplier": price_multiplier,
            "power_need": power_need,
        }
        minted = self._maybe_mint_evidence_token(organelle_id, power_need)
        if minted:
            settlement["evidence_tokens_minted"] = minted
        return settlement

    def _apply_colony_tax(self, organelle_id: str, settlement: Dict[str, float]) -> None:
        if not getattr(self.config.assimilation_tuning, "colonies_enabled", False):
            return
        colony = self._find_member_colony(organelle_id)
        if colony is None:
            return
        cid, meta = colony
        cfg = self.config.assimilation_tuning
        tax_rate = float(meta.get("tax_rate", getattr(cfg, "colony_tax_rate", 0.1)))
        if tax_rate <= 0.0:
            return
        delta = float(settlement.get("delta", 0.0))
        if delta <= 0.0:
            return
        tax = max(0.0, delta * tax_rate)
        if tax <= 0.0:
            return
        try:
            balance = float(self.host.ledger.energy_balance(organelle_id))
        except Exception:
            balance = 0.0
        tax = min(tax, balance)
        if tax <= 0.0:
            return
        try:
            if not self.host.ledger.consume_energy(organelle_id, tax):
                return
        except Exception:
            return
        meta["pot"] = float(meta.get("pot", 0.0)) + tax
        meta["_pot_earn_gen"] = float(meta.get("_pot_earn_gen", 0.0)) + tax
        settlement["energy_after"] = float(self.host.ledger.energy_balance(organelle_id))
        settlement["colony_tax"] = tax
        self._log_colony_event(
            meta, self.generation_index, "tax", member=organelle_id, amount=float(tax), colony=cid
        )

    def _record_episode(
        self,
        task: GridTask,
        organelle_id: str,
        reward: RewardBreakdown,
        metrics: RouteMetrics,
        settlement: Dict[str, float],
        success: bool,
        adapter_utilisation: dict[str, float],
    ) -> None:
        try:
            self._apply_colony_tax(organelle_id, settlement)
        except Exception:
            pass
        episode = EpisodeLog(
            episode_id=f"epi_{len(self.logs)}",
            task_id=task.task_id,
            organelles=[organelle_id],
            rewards=reward,
            energy_spent=settlement["cost"],
            observations={
                "prompt": task.prompt,
                "answer": metrics.answer,
                "cell": {
                    "family": task.family,
                    "depth": task.depth,
                },
                "price": task.price,
                "success": success,
                "energy_before": settlement["energy_before"],
                "energy_after": settlement["energy_after"],
                "colony_tax": settlement.get("colony_tax"),
                "roi": settlement["roi"],
                "metrics": {
                    "tokens": metrics.tokens,
                    "prompt_tokens": getattr(metrics, "prompt_tokens", 0),
                    "generated_tokens": getattr(metrics, "generated_tokens", 0),
                    "recurrent_passes": getattr(metrics, "recurrent_passes", 1),
                    "latency_ms": metrics.latency_ms,
                    "flops_estimate": metrics.flops_estimate,
                    "memory_gb": metrics.memory_gb,
                    "trainable_params": metrics.trainable_params,
                    "active_adapters": metrics.active_adapters,
                    "adapter_utilisation": adapter_utilisation,
                },
            },
        )
        self.logs.append(episode)
        if getattr(self.config, "foraging", None) and getattr(
            self.config.foraging, "enabled", False
        ):
            genome = self.population.population.get(organelle_id)
            decay = float(getattr(self.config.foraging, "q_decay_default", 0.3))
            if genome is not None and math.isfinite(getattr(genome, "q_decay", decay)):
                decay = float(genome.q_decay)
            q_init = float(getattr(self.config.foraging, "q_init", 0.0))
            try:
                roi_value = float(settlement.get("roi", 0.0))
            except Exception:
                roi_value = 0.0
            cell_key = (str(task.family), str(task.depth))
            try:
                self.population.update_cell_value(
                    organelle_id,
                    cell_key,
                    roi=roi_value,
                    decay=decay,
                    q_init=q_init,
                )
            except Exception:
                pass
        # Keep a bounded in-memory cache to avoid RAM growth on long runs
        try:
            limit = int(getattr(self.config.metrics, "in_memory_log_limit", 256))
        except Exception:
            limit = 256
        if limit <= 0:
            self.logs = []
        elif len(self.logs) > limit:
            self.logs = self.logs[-limit:]
        if self.sink:
            self.sink.log_episode(episode)

    def _select_synergy_pair(
        self,
        fallback_organelle: str | None = None,
        excluded: set[tuple[str, str]] | None = None,
    ) -> tuple[str, str] | None:
        # Prefer top co-routing pairs; else use top-ROI fallback including the provided organelle.
        # Skips any pairs present in `excluded`.
        excluded = excluded or set()
        try:
            if hasattr(self, "_co_routing_counts") and self._co_routing_counts:
                pairs_sorted = sorted(
                    self._co_routing_counts.items(), key=lambda kv: kv[1], reverse=True
                )
                for (a, b), _c in pairs_sorted:
                    pair = (min(a, b), max(a, b))
                    if pair in excluded:
                        continue
                    if a in self.population.population and b in self.population.population:
                        return pair
        except Exception:
            pass
        # Fallback by ROI
        try:
            scored = [
                (oid, float(self.population.average_roi(oid, limit=5)))
                for oid in self.population.population.keys()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [oid for oid, _ in scored[: min(6, len(scored))]]
            if fallback_organelle and fallback_organelle in top and len(top) >= 2:
                # Try each candidate paired with fallback until one is not excluded
                for other in top:
                    if other == fallback_organelle:
                        continue
                    pair = tuple(sorted((fallback_organelle, other)))  # type: ignore[assignment]
                    if pair not in excluded:
                        return pair  # type: ignore[return-value]
            if len(top) >= 2:
                # Return the first non-excluded pair in rank order
                for i in range(len(top)):
                    for j in range(i + 1, len(top)):
                        pair = tuple(sorted((top[i], top[j])))  # type: ignore[assignment]
                        if pair not in excluded:
                            return pair  # type: ignore[return-value]
        except Exception:
            pass
        return None

    def _run_team_episode(self, pair: tuple[str, str], task: GridTask) -> bool:
        # Best-of-two answer (vote) with individual energy settlement
        if len(pair) != 2:
            return False
        a_id, b_id = pair
        results: list[tuple[str, RouteMetrics, bool, RewardBreakdown, dict[str, float]]] = []
        rng = getattr(self.environment, "rng", random)
        order = [a_id, b_id]
        try:
            rng.shuffle(order)
        except Exception:
            random.shuffle(order)
        for oid in order:
            result = self.host.step(
                prompt=task.prompt,
                intent="team episode",
                max_routes=1,
                allowed_organelle_ids=[oid],
            )
            metrics = result.responses.get(oid)
            if metrics is None:
                continue
            success, reward = task.evaluate(metrics.answer)
            settlement = self._settle_episode(oid, task, reward, metrics)
            results.append((oid, metrics, success, reward, settlement))
            # record episode per member
            adapter_utilisation = (
                {k: float(v) for k, v in metrics.active_adapters.items()}
                if isinstance(metrics.active_adapters, dict)
                else {}
            )
            self._record_episode(
                task, oid, reward, metrics, settlement, success, adapter_utilisation
            )
            # C2C: post latent capsule from this team step
            try:
                if bool(
                    getattr(self.config.comms, "c2c_enabled", False)
                ):  # pragma: no cover - integration exercised in long runs
                    ttl = int(getattr(self.config.comms, "c2c_ttl", 5))
                    cost = float(getattr(self.config.comms, "c2c_post_cost", 0.1))
                    latent = result.envelope.observation.state.get("latent")
                    member_colony_post = None
                    if self.colonies:
                        for cid, meta in self.colonies.items():
                            if oid in meta.get("members", []):
                                member_colony_post = (cid, meta)
                                break
                    self._post_c2c_latent(oid, latent, cost, ttl, member_colony_post)
            except Exception:
                pass
        if not results:
            return False
        # Optional: single handoff (solver→checker)
        try:
            if (
                bool(getattr(self.config.assimilation_tuning, "team_handoff_enabled", False))
                and len(results) == 2
            ):
                # Cap and cost
                cap = int(getattr(self.config.assimilation_tuning, "team_handoff_cap_per_gen", 4))
                cost = float(getattr(self.config.assimilation_tuning, "team_handoff_cost", 0.05))
                if getattr(self, "_team_handoff_used", 0) >= cap:
                    raise RuntimeError("handoff_cap_reached")
                # pick current winner and let the other revise
                best_pair = max(results, key=lambda tup: float(tup[4].get("roi", 0.0)))
                winner_id = best_pair[0]
                checker_id = b_id if winner_id == a_id else a_id
                winner_answer = str(best_pair[1].answer)
                prompt_template = str(
                    getattr(
                        self.config.assimilation_tuning,
                        "team_handoff_prompt",
                        "Partner answer:\n{answer}\nRespond with a critique and improved answer that differs.",
                    )
                )
                handoff_prompt = f"{prompt_template.format(answer=winner_answer)}\n\nTask:\n{task.prompt}".strip()
                # charge small energy to checker if possible
                try:
                    if self.host.ledger.energy_balance(checker_id) >= cost:
                        self.host.ledger.consume_energy(checker_id, cost)
                except Exception:
                    pass
                result_rev = self.host.step(
                    prompt=handoff_prompt,
                    intent="team handoff",
                    max_routes=1,
                    allowed_organelle_ids=[checker_id],
                )
                m_rev = result_rev.responses.get(checker_id)
                if m_rev is not None:
                    success_rev, reward_rev = task.evaluate(m_rev.answer)
                    st_rev = self._settle_episode(checker_id, task, reward_rev, m_rev)
                    # Do not record a separate episode for the revision to keep accounting simple
                    results.append((checker_id, m_rev, success_rev, reward_rev, st_rev))
                    self._team_handoff_used = int(getattr(self, "_team_handoff_used", 0)) + 1
        except Exception:
            pass
        # Choose winner by highest ROI
        best = None
        best_roi = float("-inf")
        for oid, metrics, success, reward, settlement in results:
            roi = float(settlement.get("roi", 0.0))
            if roi > best_roi:
                best_roi = roi
                best = (oid, metrics, success, reward, settlement)
        # post a small hint if the winner succeeded
        if best is not None and bool(best[2]):
            try:
                hint = "Team tip: cross-check units and counts."
                self.environment.post_message(
                    best[0],
                    hint,
                    cost=float(getattr(self.config.comms, "post_cost", 0.2)),
                    ttl=int(getattr(self.config.comms, "ttl", 10)),
                )
            except Exception:
                pass
        return True

    def _curriculum_allowed_cells(
        self,
    ) -> list[GridKey] | None:  # pragma: no cover - exercised indirectly in long runs
        cfg = getattr(self.config, "curriculum", None)
        if cfg is None:
            return None
        try:
            warmup = int(getattr(cfg, "warmup_generations", 0) or 0)
        except Exception:
            warmup = 0
        if warmup <= 0 or self.generation_index > warmup:
            return None
        families = [str(f) for f in getattr(cfg, "warmup_families", []) if f]
        depths = [str(d) for d in getattr(cfg, "warmup_depths", []) if d]
        controller_cells = getattr(self.environment.controller, "cells", {})
        if not controller_cells:
            return None
        allowed: list[GridKey] = []
        for cell in controller_cells.keys():
            fam_ok = not families or cell[0] in families
            depth_ok = not depths or cell[1] in depths
            if fam_ok and depth_ok:
                allowed.append(cell)
        return allowed or None

    def _sample_cell_with_curriculum(
        self, lp_mix: float
    ) -> GridKey:  # pragma: no cover - randomness hinders determinism
        allowed = self._curriculum_allowed_cells()
        controller = self.environment.controller
        if not allowed:
            return controller.sample_cell(lp_mix=lp_mix)
        for _ in range(8):
            cell = controller.sample_cell(lp_mix=lp_mix)
            if cell in allowed:
                return cell
        rng = getattr(self.environment, "rng", random)
        return rng.choice(allowed)

    def _sample_task_lp(self, lp_mix: float) -> GridTask:
        try:
            cell = self._sample_cell_with_curriculum(lp_mix)
            state = self.environment.controller.get_state(cell)
            use_canary = (
                state.success_ema > self.environment.canary_q_min
                and self.environment.rng.random() < 0.1
            )
            return self.environment.sample_task_from_cell(cell, canary=use_canary)
        except Exception:
            return self.environment.sample_task()

    def _foraging_select_cell(self, organelle_id: str) -> tuple[str, str]:
        forage_cfg = getattr(self.config, "foraging", None)
        cells = list(getattr(self.environment.controller, "cells", {}).keys())
        allowed = self._curriculum_allowed_cells()
        if allowed:
            allowed_set = set(allowed)
            filtered = [cell for cell in cells if cell in allowed_set]
            if filtered:
                cells = filtered
        if not cells:
            return self.environment.controller.sample_cell(lp_mix=0.0)
        genome = self.population.population.get(organelle_id)
        q_map = self.population.cell_values.get(organelle_id, {})
        counts = self.population.cell_counts.get(organelle_id, {})
        q_init = float(getattr(forage_cfg, "q_init", 0.0)) if forage_cfg else 0.0
        beta = (
            genome.beta_exploit
            if genome is not None
            else (forage_cfg.beta_default if forage_cfg else 1.5)
        )
        if not math.isfinite(beta) or beta <= 0.0:
            beta = forage_cfg.beta_default if forage_cfg else 1.5
        total_visits = sum(counts.values()) + len(cells)
        scores: list[float] = []
        for cell in cells:
            q_value = float(q_map.get(cell, q_init))
            bonus = 0.0
            if genome is not None and genome.ucb_bonus > 0.0 and total_visits > 0:
                count = counts.get(cell, 0)
                bonus = genome.ucb_bonus * math.sqrt(math.log(total_visits + 1) / (count + 1))
            scores.append(q_value + bonus)
        max_score = max(scores) if scores else 0.0
        exps = [math.exp(beta * (score - max_score)) for score in scores]
        total = sum(exps)
        rng = getattr(self.environment, "rng", random)
        if total <= 0.0:
            return rng.choice(cells)
        probs = [val / total for val in exps]
        pol = self._active_policies.get(organelle_id)
        if pol and isinstance(pol.get("cell_pref"), dict):
            pref = (str(pol["cell_pref"].get("family")), str(pol["cell_pref"].get("depth")))
            bias_strength = float(getattr(self.config.policy, "bias_strength", 0.3))
            cap = float(getattr(forage_cfg, "policy_bias_cap", 0.5)) if forage_cfg else 0.5
            boost = 1.0 + max(0.0, min(cap, bias_strength))
            for idx, cell in enumerate(cells):
                if cell == pref:
                    probs[idx] *= boost
                    break
            total_prob = sum(probs)
            if total_prob > 0.0:
                probs = [p / total_prob for p in probs]
        try:
            choice = rng.choices(cells, weights=probs, k=1)[0]
        except Exception:
            choice = rng.choice(cells)
        return choice

    def _sample_task_with_policy(self, lp_mix: float, organelle_id: str) -> GridTask:
        if getattr(self.config, "foraging", None) and getattr(
            self.config.foraging, "enabled", False
        ):
            rng = getattr(self.environment, "rng", random)
            if lp_mix > 0.0 and rng.random() < lp_mix:
                cell = self._sample_cell_with_curriculum(lp_mix)
            else:
                cell = self._foraging_select_cell(organelle_id)
            return self.environment.sample_task_from_cell(cell)
        pol = self._active_policies.get(organelle_id)
        if pol and isinstance(pol.get("cell_pref"), dict):
            try:
                fam = str(pol["cell_pref"].get("family"))
                dep = str(pol["cell_pref"].get("depth"))
                preferred = (fam, dep)
                allowed = self._curriculum_allowed_cells()
                if allowed and preferred not in allowed:
                    preferred = random.choice(allowed)
                bias = float(getattr(self.config.policy, "bias_strength", 0.3))
                if 0.0 < bias <= 1.0 and self.environment.rng.random() < bias:
                    return self.environment.sample_task_from_cell(preferred)
            except Exception:
                pass
        return self._sample_task_lp(lp_mix)

    @staticmethod
    def _parse_policy_json(text: str, allowed: list[str]) -> dict[str, object]:
        return parse_policy_json(text=text, allowed=allowed)

    def run_colony_inference(
        self, member_ids: list[str], prompt: str, strategy: str = "best_of_two"
    ) -> dict[str, object]:
        """Run an ad-hoc inference over a set of members and return the selected answer.

        - strategy="best_of_two" chooses the answer with the larger number of non-whitespace characters
          (heuristic when no ground-truth reward is available).
        - Returns a dict: {"selected_id", "selected_answer", "answers": {id: answer}, "tokens": {id: int}}
        """
        answers: dict[str, str] = {}
        tokens: dict[str, int] = {}
        for oid in member_ids:
            result = self.host.step(
                prompt=prompt, intent="colony infer", max_routes=1, allowed_organelle_ids=[oid]
            )
            metrics = result.responses.get(oid)
            ans = ""
            tok = 0
            if metrics is not None:
                ans = str(metrics.answer)
                tok = int(getattr(metrics, "tokens", 0) or 0)
            answers[oid] = ans
            tokens[oid] = tok
        selected_id = member_ids[0] if member_ids else ""
        if strategy == "best_of_two" and len(member_ids) >= 2:

            def _score(a: str) -> int:
                return len(str(a).strip())

            selected_id = max(member_ids, key=lambda oid: _score(answers.get(oid, "")))
        selected_answer = answers.get(selected_id, "")
        return {
            "selected_id": selected_id,
            "selected_answer": selected_answer,
            "answers": answers,
            "tokens": tokens,
        }

    def _probe_co_routing(self, active_ids: list[str]) -> None:
        """Run a few light routing probes to populate co‑routing counts per generation.

        Uses host routing with k=2 to observe which organelles tend to co‑route on sampled tasks.
        Keeps the count in `_co_routing_counts` and avoids energy settlement/logging.
        """
        try:
            per_gen = int(getattr(self.config.assimilation_tuning, "team_routing_probe_per_gen", 0))
        except Exception:
            per_gen = 0
        if per_gen <= 0 or len(active_ids) < 2:
            return
        # Ensure counter exists
        if not hasattr(self, "_co_routing_counts"):
            self._co_routing_counts = {}
        for _ in range(per_gen):
            try:
                task = self.environment.sample_task()
                # Ask router to pick two candidates via normal step (no settlement here)
                result = self.host.step(
                    prompt=task.prompt,
                    intent="routing probe",
                    max_routes=2,
                    allowed_organelle_ids=active_ids,
                )
                # Collect first two unique routed organelles
                picked: list[str] = []
                for evt in result.routes:
                    oid = getattr(evt, "organelle_id", None)
                    if isinstance(oid, str) and oid not in picked:
                        picked.append(oid)
                    if len(picked) >= 2:
                        break
                if len(picked) == 2:
                    a, b = sorted(picked)
                    key = (a, b)
                    self._co_routing_counts[key] = int(self._co_routing_counts.get(key, 0)) + 1
            except Exception:
                # Best‑effort only; continue
                continue

    def _should_boost(self, organelle_id: str) -> bool:
        """Heuristic: boost evidence for near-threshold or promising organelles.

        Uses recent ROI vs aggregate ROI as a proxy. Best-effort and safe.
        """
        try:
            recent = float(self.population.average_roi(organelle_id, limit=5))
        except Exception:
            return False
        try:
            aggregate = float(self.population.aggregate_roi())
        except Exception:
            aggregate = 0.0
        # Boost when recent ROI is close to or above 90% of aggregate
        threshold = max(0.0, 0.9 * aggregate)
        return recent >= threshold

    def _compute_budget_map(
        self, active_ids: list[str], base_bs: int
    ) -> tuple[dict[str, int], dict[str, object]]:
        env_cfg = self.config.environment
        if not getattr(env_cfg, "budget_enabled", True):
            budgets = {organelle_id: max(0, base_bs) for organelle_id in active_ids}
            population_by_id = self.population.population
            per_org = {}
            for organelle_id in active_ids:
                genome = population_by_id.get(organelle_id)
                trait = float(getattr(genome, "explore_rate", 0.0)) if genome is not None else 0.0
                per_org[organelle_id] = {
                    "energy": 0.0,
                    "trait": trait,
                    "policy": 1.0,
                    "raw": budgets[organelle_id],
                }
            raw_total = int(sum(budgets.values()))
            meta = {
                "base": base_bs,
                "global_cap": int(getattr(env_cfg, "global_episode_cap", 0)),
                "per_org": per_org,
                "raw_total": raw_total,
                "capped_total": raw_total,
                "cap_hit": False,
            }
            return budgets, meta
        ticket = max(1e-6, float(self.config.energy.m))
        energy_floor = max(0.0, float(getattr(env_cfg, "budget_energy_floor", 0.4)))
        energy_ceiling = max(energy_floor, float(getattr(env_cfg, "budget_energy_ceiling", 3.0)))
        trait_bonus = max(0.0, float(getattr(env_cfg, "budget_trait_bonus", 1.0)))
        policy_floor = max(0.0, float(getattr(env_cfg, "budget_policy_floor", 0.3)))
        policy_ceiling = max(policy_floor, float(getattr(env_cfg, "budget_policy_ceiling", 2.0)))
        global_cap = int(max(0, getattr(env_cfg, "global_episode_cap", 0)))
        active_policies = getattr(self, "_active_policies", None)
        budgets: dict[str, int] = {}
        per_org_meta: dict[str, dict[str, float]] = {}
        raw_total = 0
        require_policy = bool(getattr(env_cfg, "budget_policy_requires_parse", False))
        for organelle_id in active_ids:
            try:
                balance = float(self.host.ledger.energy_balance(organelle_id))
            except Exception:
                balance = 0.0
            energy_ratio = balance / ticket if ticket > 0 else 0.0
            clamped_ratio = max(energy_floor, min(energy_ceiling, energy_ratio))
            energy_factor = math.sqrt(clamped_ratio)
            genome = self.population.population.get(organelle_id)
            if genome is not None:
                trait = float(getattr(genome, "budget_aggressiveness", genome.explore_rate))
            else:
                trait = 0.5
            trait_factor = max(0.1, 1.0 + trait_bonus * (trait - 0.5))
            policy_frac = 1.0
            pol = active_policies.get(organelle_id) if isinstance(active_policies, dict) else None
            if isinstance(pol, dict) and isinstance(pol.get("budget_frac"), (int, float)):
                policy_frac = float(pol["budget_frac"])
            elif require_policy:
                policy_frac = policy_floor
            policy_frac = max(policy_floor, min(policy_ceiling, policy_frac))
            raw_budget = base_bs * energy_factor * trait_factor * policy_frac
            per_org = max(1, int(round(raw_budget)))
            budgets[organelle_id] = per_org
            raw_total += per_org
            per_org_meta[organelle_id] = {
                "energy": round(energy_ratio, 4),
                "trait": round(trait, 4),
                "policy": round(policy_frac, 4),
                "raw": per_org,
            }
        capped_total = raw_total
        cap_hit = False
        if global_cap > 0 and raw_total > global_cap:
            cap_hit = True
            scaled: dict[str, int] = {}
            sorted_ids = sorted(active_ids, key=lambda oid: budgets.get(oid, 0), reverse=True)
            remaining_cap = global_cap
            for oid in sorted_ids:
                raw = budgets.get(oid, 0)
                if remaining_cap <= 0:
                    alloc = 0
                else:
                    share = (raw / raw_total) if raw_total > 0 else 0.0
                    alloc = min(raw, int(math.floor(share * global_cap)))
                scaled[oid] = max(0, alloc)
                remaining_cap -= scaled[oid]
            idx = 0
            while remaining_cap > 0 and sorted_ids:
                oid = sorted_ids[idx % len(sorted_ids)]
                if scaled[oid] < budgets.get(oid, 0):
                    scaled[oid] += 1
                    remaining_cap -= 1
                idx += 1
                if idx > len(sorted_ids) * 4:
                    break
            budgets = {oid: scaled.get(oid, 0) for oid in active_ids}
            capped_total = sum(budgets.values())
        for oid, meta in per_org_meta.items():
            meta["capped"] = budgets.get(oid, 0)
        meta = {
            "base": base_bs,
            "global_cap": global_cap,
            "per_org": per_org_meta,
            "raw_total": raw_total,
            "capped_total": capped_total,
            "cap_hit": cap_hit,
        }
        return budgets, meta

    def _resolve_per_org_batch(self, organelle_id: str, base: int) -> int:
        cfg = getattr(self.config, "survival", None)
        if cfg is None or not cfg.enabled:
            return max(0, base)
        per_org = max(0, base)
        reserve_state = self._reserve_state.get(organelle_id, {})
        if reserve_state.get("active") and per_org > 0:
            scale = max(0.0, min(1.0, float(cfg.reserve_batch_scale)))
            if scale > 0.0:
                per_org = max(0, int(round(per_org * scale)))
            else:
                per_org = 0
            cap = int(cfg.steps_cap_low_energy)
            if cap > 0:
                per_org = min(per_org, cap)
        return max(0, per_org)

    def _sample_task_for_org(self, organelle_id: str, lp_mix: float) -> GridTask:
        task = (
            self._sample_task_with_policy(lp_mix, organelle_id)
            if lp_mix > 0.0
            else self.environment.sample_task()
        )
        cfg = getattr(self.config, "survival", None)
        if cfg is None or not cfg.enabled:
            return task
        reserve_state = self._reserve_state.get(organelle_id, {})
        hazard_state = self._hazard_state.get(organelle_id, {})
        reserve_active = bool(reserve_state.get("active"))
        hazard_active = bool(hazard_state.get("active"))
        price_bias = bool(getattr(cfg, "price_bias_low_energy", True))
        if not price_bias:
            return task
        if not (reserve_active or hazard_active):
            return task
        try:
            quantile = max(0.0, min(1.0, float(cfg.cheap_cell_quantile)))
            cells = getattr(self.environment.controller, "cells", {})
            if quantile <= 0.0 or not cells:
                return task
            priced: list[tuple[GridKey, float]] = []
            for key, state in cells.items():
                price = getattr(state, "price", None)
                if price is None:
                    continue
                priced.append((key, float(price)))
            if not priced:
                return task
            priced.sort(key=lambda kv: kv[1])
            idx = min(len(priced) - 1, max(0, int(math.floor(quantile * len(priced)))))
            chosen_cell = priced[idx][0]
            return self.environment.sample_task_from_cell(chosen_cell)
        except Exception:
            return task

    def _log_survival_event(self, organelle_id: str, event_type: str, **payload: object) -> None:
        entry: dict[str, object] = {
            "gen": self.generation_index,
            "org": organelle_id,
            "type": event_type,
        }
        entry.update(payload)
        self._survival_events.append(entry)
        if len(self._survival_events) > 200:
            self._survival_events = self._survival_events[-200:]

    def _update_survival_states(self, active_ids: list[str]) -> None:
        cfg = getattr(self.config, "survival", None)
        if cfg is None or not cfg.enabled:
            return
        ticket = float(self.config.energy.m)
        reserve_window = max(1, int(cfg.reserve_cost_window))
        hazard_window = max(2, int(cfg.hazard_window))
        organelle_ids = list(self.host.list_organelle_ids())
        for organelle_id in organelle_ids:
            balance = float(self.host.ledger.energy_balance(organelle_id))
            avg_cost = float(self.population.average_energy(organelle_id, limit=reserve_window))
            reserve_threshold = max(
                ticket * float(cfg.reserve_ratio), float(cfg.reserve_cost_beta) * max(avg_cost, 0.0)
            )
            prev_reserve = self._reserve_state.get(organelle_id, {})
            was_reserve = bool(prev_reserve.get("active"))
            reserve_active = reserve_threshold > 0.0 and balance < reserve_threshold
            self._reserve_state[organelle_id] = {
                "active": reserve_active,
                "threshold": reserve_threshold,
                "balance": balance,
            }
            if reserve_active and not was_reserve:
                self._log_survival_event(
                    organelle_id, "reserve_enter", balance=balance, threshold=reserve_threshold
                )
            elif was_reserve and not reserve_active:
                self._log_survival_event(
                    organelle_id, "reserve_exit", balance=balance, threshold=reserve_threshold
                )

            roi_vals = self.population.roi.get(organelle_id, [])[-hazard_window:]
            z_score = 0.0
            latest_roi = roi_vals[-1] if roi_vals else 0.0
            if len(roi_vals) >= 2:
                mean_roi = sum(roi_vals) / len(roi_vals)
                std_roi = pstdev(roi_vals)
                if std_roi > 1e-6:
                    z_score = (latest_roi - mean_roi) / std_roi
            prev_hazard = self._hazard_state.get(organelle_id, {})
            hazard_active = bool(prev_hazard.get("active"))
            cooldown = max(0, int(self._hazard_cooldown.get(organelle_id, 0)))
            if not hazard_active and cooldown > 0:
                cooldown -= 1
            entered = False
            exited = False
            if hazard_active:
                if z_score >= float(cfg.hazard_exit_threshold):
                    hazard_active = False
                    exited = True
                    cooldown = int(cfg.hazard_cooldown_gens)
            else:
                if cooldown <= 0 and z_score <= float(cfg.hazard_threshold):
                    hazard_active = True
                    entered = True
            self._hazard_cooldown[organelle_id] = cooldown
            if entered:
                rank_shift = False
                down = int(cfg.hazard_rank_downshift)
                if down > 0:
                    try:
                        organelle = self.host.get_organelle(organelle_id)
                        current_rank = (
                            getattr(organelle, "rank", None) if organelle is not None else None
                        )
                        if isinstance(current_rank, int) and current_rank > 1:
                            target_rank = max(1, current_rank - down)
                            if target_rank < current_rank and self.host.resize_organelle_rank(
                                organelle_id, target_rank
                            ):
                                genome = self.population.population.get(organelle_id)
                                if genome is not None:
                                    genome.rank = target_rank
                                rank_shift = True
                    except Exception:
                        rank_shift = False
                self._log_survival_event(
                    organelle_id,
                    "hazard_enter",
                    z=z_score,
                    roi=latest_roi,
                    cooldown=cooldown,
                    rank_shift=rank_shift,
                )
            elif exited:
                self._log_survival_event(
                    organelle_id,
                    "hazard_exit",
                    z=z_score,
                    roi=latest_roi,
                    cooldown=cooldown,
                )
            if hazard_active and float(cfg.hazard_roi_relief_boost) > 0.0:
                boost = float(cfg.hazard_roi_relief_boost)
                current_relief = float(self._roi_relief.get(organelle_id, 0.0))
                max_relief = float(getattr(self.config.assimilation_tuning, "roi_relief_max", 0.5))
                self._roi_relief[organelle_id] = min(max_relief, current_relief + boost)
            self._hazard_state[organelle_id] = {
                "active": hazard_active,
                "z": z_score,
                "cooldown": cooldown,
                "roi": latest_roi,
                "in_active_set": organelle_id in active_ids,
            }
        for stale in [oid for oid in list(self._reserve_state.keys()) if oid not in organelle_ids]:
            self._reserve_state.pop(stale, None)
        for stale in [oid for oid in list(self._hazard_state.keys()) if oid not in organelle_ids]:
            self._hazard_state.pop(stale, None)
            self._hazard_cooldown.pop(stale, None)

    def _should_block_assimilation(self, organelle_id: str) -> dict[str, object] | None:
        cfg = getattr(self.config, "survival", None)
        if cfg is None or not cfg.enabled:
            return None
        info = self._reserve_state.get(organelle_id, {})
        if info.get("active"):
            return {
                "reserve_threshold": float(info.get("threshold", 0.0)),
                "balance": float(info.get("balance", 0.0)),
            }
        return None

    def _snapshot_survival_state(self) -> dict[str, object] | None:
        cfg = getattr(self.config, "survival", None)
        if cfg is None or not cfg.enabled:
            return None
        reserve_active = [oid for oid, info in self._reserve_state.items() if info.get("active")]
        hazard_active = [oid for oid, info in self._hazard_state.items() if info.get("active")]
        if bool(getattr(self.config.survival, "price_bias_low_energy", True)):
            bias_active = sorted({*(reserve_active), *(hazard_active)})
        else:
            bias_active = []
        snapshot: dict[str, object] = {
            "reserve_active_ids": reserve_active,
            "hazard_active_ids": hazard_active,
            "reserve_active_count": len(reserve_active),
            "hazard_active_count": len(hazard_active),
            "price_bias_active_count": len(bias_active),
            "price_bias_active_ids": bias_active,
            "reserve_thresholds": {
                oid: float(info.get("threshold", 0.0)) for oid, info in self._reserve_state.items()
            },
            "hazard_zscores": {
                oid: float(info.get("z", 0.0)) for oid, info in self._hazard_state.items()
            },
        }
        events = [
            event for event in self._survival_events if event.get("gen") == self.generation_index
        ]
        if events:
            snapshot["events"] = events
        return snapshot

    @staticmethod
    def _colony_c2c_debit(meta: dict[str, object], amount: float, counter_key: str) -> bool:
        return colony_c2c_debit(meta=meta, amount=amount, counter_key=counter_key)

    def _colony_expected_cost(self, members: list[str], window: int) -> float:
        total = 0.0
        window = max(1, window)
        for organelle_id in members:
            costs = [
                float(value)
                for value in self.population.energy.get(organelle_id, [])[-window:]
                if math.isfinite(value)
            ]
            if not costs:
                continue
            avg_cost = sum(costs) / len(costs)
            total += avg_cost * window
        return total

    def _colony_roi_series(
        self, members: list[str], window: int
    ) -> tuple[list[float], list[float]]:
        history: list[float] = []
        latest: list[float] = []
        window = max(1, window)
        for organelle_id in members:
            series = [
                float(value)
                for value in self.population.roi.get(organelle_id, [])[-window:]
                if math.isfinite(value)
            ]
            if not series:
                continue
            history.extend(series)
            latest.append(series[-1])
        return history, latest

    def _apply_colony_bankruptcy_guard(self, culled: list[str]) -> None:
        if not self.colonies or not culled:
            return
        tolerance = int(getattr(self.config.assimilation_tuning, "colony_bankrupt_tolerance", 2))
        for cid, meta in list(self.colonies.items()):
            members = [str(x) for x in meta.get("members", [])]
            hits = [oid for oid in culled if oid in members]
            if not hits:
                continue
            count = int(meta.get("bankrupt_violations", 0)) + len(hits)
            meta["bankrupt_violations"] = count
            self._log_colony_event(
                meta,
                self.generation_index,
                "bankrupt",
                members=list(hits),
                total=count,
            )
            if count >= tolerance:
                self._log_colony_event(
                    meta,
                    self.generation_index,
                    "dissolve",
                    reason="bankrupt_guard",
                )
                self.colonies.pop(cid, None)

    def _enter_colony_winter(self, cid: str, meta: dict[str, object], members: list[str]) -> None:
        prev = meta.setdefault("winter_prev_ranks", {})
        if not isinstance(prev, dict):
            prev = {}
            meta["winter_prev_ranks"] = prev
        for organelle_id in members:
            try:
                organelle = self.host.get_organelle(organelle_id)
            except Exception:
                organelle = None
            rank = getattr(organelle, "rank", None) if organelle is not None else None
            if not isinstance(rank, int):
                continue
            prev.setdefault(organelle_id, rank)
            target = max(1, rank - 1)
            if target < rank:
                try:
                    if self.host.resize_organelle_rank(organelle_id, target):
                        genome = self.population.population.get(organelle_id)
                        if genome is not None:
                            genome.rank = target
                except Exception:
                    continue

    def _exit_colony_winter(self, meta: dict[str, object]) -> None:
        prev = meta.pop("winter_prev_ranks", {}) or {}
        if not isinstance(prev, dict):
            return
        for organelle_id, rank in prev.items():
            if not isinstance(rank, int):
                continue
            try:
                organelle = self.host.get_organelle(organelle_id)
            except Exception:
                organelle = None
            current = getattr(organelle, "rank", None) if organelle is not None else None
            if isinstance(current, int) and current >= rank:
                continue
            try:
                if self.host.resize_organelle_rank(organelle_id, rank):
                    genome = self.population.population.get(organelle_id)
                    if genome is not None:
                        genome.rank = rank
            except Exception:
                continue

    def _run_colony_guard(self, cid: str, meta: dict[str, object], members: list[str]) -> None:
        if not members:
            return
        try:
            tasks = self._sample_holdout_tasks()
        except Exception:
            tasks = []
        if not tasks:
            return
        stats = self._team_holdout_stats(members, tasks)
        mean_roi = float(stats.get("mean", 0.0))
        series = stats.get("series") or []
        self._log_colony_event(
            meta,
            self.generation_index,
            "deception_check",
            mean=mean_roi,
            tasks=len(series),
        )

    def _estimate_power(self, organelle_id: str, window: int = 5) -> float:
        try:
            return float(self.population.average_roi(organelle_id, limit=window))
        except Exception:
            return 0.0

    def _population_roi_mean(self, limit: int = 5) -> float:
        try:
            value = float(self.population.aggregate_roi(limit=limit))
        except Exception:
            value = 0.0
        if not math.isfinite(value):
            return 0.0
        return max(-10.0, min(10.0, value))

    def _power_need(self, organelle_id: str) -> float:
        tuning = self.config.assimilation_tuning
        target = max(0.0, min(1.0, float(getattr(tuning, "power_target", 0.75))))
        if target <= 0.0:
            return 0.0
        last_power = None
        history = self.population.assimilation_history
        for (oid, _family, _depth), records in history.items():
            if oid != organelle_id or not records:
                continue
            for record in reversed(records):
                value = record.get("power")
                if isinstance(value, (int, float)) and value >= 0.0:
                    last_power = float(value)
                    break
            if last_power is not None:
                break
        if last_power is None:
            last_power = 0.0
        last_power = max(0.0, min(1.0, float(last_power)))
        need = max(0.0, target - last_power)
        return max(0.0, min(1.0, need / target))

    def _maybe_mint_evidence_token(self, organelle_id: str, power_need: float) -> int:
        tuning = self.config.assimilation_tuning
        threshold = max(0.0, min(1.0, float(getattr(tuning, "evidence_token_threshold", 1.0))))
        if power_need < threshold:
            return 0
        amount = int(getattr(tuning, "evidence_token_mint", 1))
        cap_value = int(getattr(tuning, "evidence_token_cap", 0))
        cap = cap_value if cap_value > 0 else None
        minted = self.population.grant_evidence(organelle_id, amount, cap=cap)
        if minted and self._power_econ_stats is not None:
            self._power_econ_stats["tokens_minted"] = (
                int(self._power_econ_stats.get("tokens_minted", 0)) + minted
            )
        return minted

    def _build_comms_hint(self, organelle_id: str, cell_meta: tuple[str, str] | None) -> str:
        parts: list[str] = []
        stats_map = getattr(self.environment, "organism_stats", None)
        if isinstance(stats_map, dict):
            per_org = stats_map.get(organelle_id)
            if isinstance(per_org, dict) and per_org:
                best_cell: tuple[str, str] | None = None
                best_score: float | None = None
                for cell, value in per_org.items():
                    try:
                        score = float(value)
                    except Exception:
                        continue
                    if best_score is None or score > best_score:
                        best_score = score
                        if isinstance(cell, tuple) and len(cell) == 2:
                            best_cell = (str(cell[0]), str(cell[1]))
                        else:
                            best_cell = None
                if best_score is not None:
                    if best_cell is not None:
                        parts.append(f"best={best_cell[0]}:{best_cell[1]}@{best_score:.2f}")
                    else:
                        parts.append(f"best_score={best_score:.2f}")
        if not parts and cell_meta is not None:
            parts.append(f"focus={cell_meta[0]}:{cell_meta[1]}")
        roi_history = self.population.roi.get(organelle_id, [])
        if roi_history:
            try:
                parts.append(f"roi={float(roi_history[-1]):.2f}")
            except Exception:
                pass
        genome = self.population.population.get(organelle_id)
        if genome is not None:
            explore_rate = getattr(genome, "explore_rate", None)
            if isinstance(explore_rate, (int, float)):
                parts.append(f"explore={float(explore_rate):.2f}")
            hint_weight = getattr(genome, "hint_weight", None)
            if isinstance(hint_weight, (int, float)) and float(hint_weight) > 0.0:
                parts.append(f"hint={float(hint_weight):.2f}")
        if parts:
            return " | ".join(parts)
        if cell_meta is not None:
            return f"focus={cell_meta[0]}:{cell_meta[1]}"
        return "share best cell and ROI once improved"

    def _queue_comms_credit(
        self, poster: str | None, reader: str, baseline: float, credit_amount: float
    ) -> None:
        if not poster or poster == reader or credit_amount <= 0.0:
            return
        window = int(getattr(self.config.comms, "credit_power_window", 6))
        min_delta = float(getattr(self.config.comms, "credit_power_min_delta", 0.05))
        self._comms_credit_queue.append(
            {
                "poster": poster,
                "reader": reader,
                "baseline": baseline,
                "credit": credit_amount,
                "gen": self.generation_index,
                "window": max(1, window),
                "min_delta": max(0.0, min_delta),
            }
        )

    def _log_comms_event(self, event_type: str, **payload: object) -> None:
        entry: dict[str, object] = {"gen": self.generation_index, "type": event_type}
        entry.update(payload)
        self._comms_events_gen.append(entry)

    def _process_comms_credit(self) -> None:
        if not self._comms_credit_queue:
            return
        new_queue: list[dict[str, object]] = []
        rng = getattr(self.environment, "rng", None)
        for event in self._comms_credit_queue:
            max_window = int(event.get("window", 6))
            if self.generation_index - int(event.get("gen", self.generation_index)) > max_window:
                continue
            reader = str(event.get("reader", ""))
            poster = str(event.get("poster", ""))
            baseline = float(event.get("baseline", 0.0))
            min_delta = float(event.get("min_delta", 0.0))
            credit_amount = float(event.get("credit", 0.0))
            current_power = self._estimate_power(reader)
            delta = current_power - baseline
            if delta >= min_delta:
                try:
                    self.host.ledger.credit_energy(poster, credit_amount)
                    self._comms_stats_gen["credits"] = self._comms_stats_gen.get("credits", 0) + 1
                    self._log_comms_event(
                        "credit", poster=poster, reader=reader, delta=delta, amount=credit_amount
                    )
                except Exception:
                    pass
                continue
            # allow small stochastic decay to prevent infinite queue
            if rng is not None and rng.random() < 0.05:
                continue
            new_queue.append(event)
        self._comms_credit_queue = new_queue

    def _apply_prompt_scaffold(self, task: GridTask, prompt_text: str) -> tuple[str, bool]:
        cfg = getattr(self.config, "prompting", None)
        if cfg is None or not getattr(cfg, "few_shot_enabled", False):
            return prompt_text, False
        examples = (getattr(cfg, "few_shot_examples", {}) or {}).get(task.family)
        if not examples:
            return prompt_text, False
        header = getattr(cfg, "few_shot_header", "")
        footer = getattr(cfg, "few_shot_footer", "")
        separator = getattr(cfg, "few_shot_separator", "\n") or "\n"
        lines: list[str] = []
        if header:
            lines.append(header)
        added = 0
        for entry in examples:
            prompt_example = str(entry.get("prompt", "")).strip()
            answer_example = str(entry.get("answer", "")).strip()
            if not prompt_example or not answer_example:
                continue
            lines.append(f"Example Input: {prompt_example}")
            lines.append(f"Example Output: {answer_example}")
            added += 1
        if added == 0:
            return prompt_text, False
        if footer:
            lines.append(footer)
        lines.append(f"Task: {prompt_text}")
        scaffolded = separator.join(lines)
        return scaffolded, True

    def _update_winter_cycle(self) -> None:
        cfg = getattr(self.config, "winter", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            self._winter_active = False
            self._winter_price_factor = 1.0
            self._winter_ticket_multiplier = 1.0
            return

        interval = max(1, int(getattr(cfg, "winter_interval", 40)))
        duration = max(1, int(getattr(cfg, "winter_duration", 4)))
        price_mult = max(0.0, float(getattr(cfg, "price_multiplier", 1.0)))
        ticket_mult = max(0.0, float(getattr(cfg, "ticket_multiplier", 1.0)))

        if self._winter_active:
            self._winter_price_factor = price_mult
            self._winter_ticket_multiplier = ticket_mult
            self._winter_timer -= 1
            if self._winter_timer <= 0:
                self._winter_active = False
                self._winter_timer = 0
                self._winter_price_factor = 1.0
                self._winter_ticket_multiplier = 1.0
                post_roi = self._population_roi_mean(limit=6)
                pre_roi = float(self._winter_pre_roi)
                pre_assim = int(self._winter_pre_assim_attempts)
                post_assim = int(getattr(self, "_last_assim_attempts", 0))
                self._winter_events_gen.append(
                    {
                        "gen": self.generation_index,
                        "type": "winter_end",
                        "pre_roi": pre_roi,
                        "post_roi": post_roi,
                        "delta_roi": post_roi - pre_roi,
                        "pre_assim": pre_assim,
                        "post_assim": post_assim,
                        "delta_assim": post_assim - pre_assim,
                    }
                )
                self._winter_pre_roi = 0.0
                self._winter_pre_assim_attempts = 0
                bonus = float(self.config.energy.m) * max(
                    0.0, float(getattr(cfg, "post_winter_bonus", 0.0))
                )
                if bonus > 0.0:
                    credited = self._apply_post_winter_bonus(bonus)
                    if credited:
                        self._winter_events_gen.append(
                            {
                                "gen": self.generation_index,
                                "type": "winter_bonus",
                                "credited": credited,
                                "amount": bonus,
                            }
                        )
                self._winter_counter = 0
            return

        # not currently in winter
        self._winter_price_factor = 1.0
        self._winter_ticket_multiplier = 1.0
        self._winter_counter += 1
        if self._winter_counter >= interval:
            self._winter_active = True
            self._winter_timer = duration
            self._winter_counter = 0
            self._winter_price_factor = price_mult
            self._winter_ticket_multiplier = ticket_mult
            self._winter_pre_roi = self._population_roi_mean(limit=6)
            self._winter_pre_assim_attempts = int(getattr(self, "_last_assim_attempts", 0))
            self._winter_events_gen.append(
                {
                    "gen": self.generation_index,
                    "type": "winter_start",
                    "duration": duration,
                    "price_multiplier": price_mult,
                    "ticket_multiplier": ticket_mult,
                    "pre_roi": float(self._winter_pre_roi),
                    "pre_assim": int(self._winter_pre_assim_attempts),
                }
            )

    def _apply_post_winter_bonus(self, amount: float) -> dict[str, float]:
        credited: dict[str, float] = {}
        if amount <= 0.0:
            return credited
        for organelle_id in self.population.population.keys():
            try:
                before = self.host.ledger.energy_balance(organelle_id)
                cap = getattr(self.host.ledger, "energy_cap", before + amount)
            except Exception:
                continue
            available = max(0.0, cap - before)
            grant = min(amount, available)
            if grant <= 0.0:
                continue
            try:
                self.host.ledger.credit_energy(organelle_id, grant)
            except Exception:
                continue
            credited[organelle_id] = float(grant)
        return credited

    def _consume_c2c_latents(
        self,
        organelle_id: str,
        member_colony: tuple[str, dict[str, object]] | None,
        read_cost: float,
    ) -> None:
        caches = self.environment.read_caches(max_items=1)
        for cap in caches:
            vec = cap.get("latent") or []
            if not isinstance(vec, list) or not vec:
                continue
            paid = False
            if member_colony is not None:
                _cid, _meta = member_colony
                if self._colony_c2c_debit(_meta, read_cost, "c2c_reads_left"):
                    paid = True
            if not paid:
                try:
                    if self.host.ledger.energy_balance(organelle_id) < read_cost:
                        continue
                    self.host.ledger.consume_energy(organelle_id, read_cost)
                    paid = True
                except Exception:
                    continue
            self._pending_latents.setdefault(organelle_id, []).append(vec)

    def _post_c2c_latent(
        self,
        organelle_id: str,
        latent: object,
        cost: float,
        ttl: int,
        member_colony: tuple[str, dict[str, object]] | None,
    ) -> None:
        if not isinstance(latent, list) or not latent:
            return
        paid = False
        if member_colony is not None:
            _cid, _meta = member_colony
            if self._colony_c2c_debit(_meta, cost, "c2c_posts_left"):
                paid = True
        if not paid:
            try:
                if self.host.ledger.energy_balance(organelle_id) < cost:
                    return
                self.host.ledger.consume_energy(organelle_id, cost)
            except Exception:
                return
        try:
            self.environment.post_cache(organelle_id, latent, ttl=ttl)
        except Exception:
            pass

    def _knowledge_cfg(self):
        return getattr(self.config, "knowledge", None)

    def _knowledge_enabled(self) -> bool:
        cfg = self._knowledge_cfg()
        return bool(cfg and getattr(cfg, "enabled", False))

    def _prune_knowledge_cache(self) -> None:
        if not self._knowledge_enabled() or not self._knowledge_store:
            return
        cfg = self._knowledge_cfg()
        ttl = max(1, int(getattr(cfg, "ttl", 20)))
        limit = max(1, int(getattr(cfg, "max_items", 8)))
        now = self.generation_index
        expired = 0
        for organelle_id, entries in list(self._knowledge_store.items()):
            pruned: list[dict[str, object]] = []
            for entry in entries:
                created = int(entry.get("gen", now))
                if now - created <= ttl:
                    pruned.append(entry)
                else:
                    expired += 1
            if len(pruned) > limit:
                pruned = pruned[:limit]
            if pruned:
                self._knowledge_store[organelle_id] = pruned
            else:
                self._knowledge_store.pop(organelle_id, None)
        if expired:
            self._knowledge_stats_gen["expired"] = (
                self._knowledge_stats_gen.get("expired", 0) + expired
            )

    def _prepare_knowledge_prompt(self, organelle_id: str, task: GridTask) -> str:
        if not self._knowledge_enabled():
            return ""
        if self._reserve_state.get(organelle_id, {}).get("active"):
            return ""
        if self._hazard_state.get(organelle_id, {}).get("active"):
            return ""
        entries = self._knowledge_store.get(organelle_id)
        if not entries:
            return ""
        cfg = self._knowledge_cfg()
        ttl = max(1, int(getattr(cfg, "ttl", 20)))
        now = self.generation_index
        relevant: list[str] = []
        for entry in entries:
            if entry.get("cell") != task.cell:
                continue
            created = int(entry.get("gen", now))
            if now - created > ttl:
                continue
            note = entry.get("note")
            if isinstance(note, str) and note:
                relevant.append(note)
        if not relevant:
            return ""
        read_cost = max(0.0, float(getattr(cfg, "read_cost", 0.0)))
        if read_cost > 0.0 and not self.host.ledger.consume_energy(organelle_id, read_cost):
            self._knowledge_stats_gen["read_denied"] = (
                self._knowledge_stats_gen.get("read_denied", 0) + 1
            )
            return ""
        max_items = max(1, int(getattr(cfg, "max_items", len(relevant))))
        snippet = "\n".join(f"- {relevant[idx]}" for idx in range(min(len(relevant), max_items)))
        self._knowledge_stats_gen["reads"] = self._knowledge_stats_gen.get("reads", 0) + 1
        self._knowledge_stats_gen["hits"] = self._knowledge_stats_gen.get("hits", 0) + 1
        return f"Memory cache for this cell:\n{snippet}"

    def _record_knowledge_entry(
        self, organelle_id: str, task: GridTask, metrics: RouteMetrics
    ) -> None:
        if not self._knowledge_enabled():
            return
        answer = getattr(metrics, "answer", "")
        if not isinstance(answer, str):
            return
        clean_answer = answer.strip()
        if not clean_answer:
            return
        collapsed = textwrap.shorten(clean_answer.replace("\n", " "), width=160, placeholder="…")
        note = f"{task.family}/{task.depth}: {collapsed}"
        existing = self._knowledge_store.get(organelle_id, [])
        if any(entry.get("note") == note and entry.get("cell") == task.cell for entry in existing):
            return
        cfg = self._knowledge_cfg()
        write_cost = max(0.0, float(getattr(cfg, "write_cost", 0.0)))
        if write_cost > 0.0 and not self.host.ledger.consume_energy(organelle_id, write_cost):
            self._knowledge_stats_gen["write_denied"] = (
                self._knowledge_stats_gen.get("write_denied", 0) + 1
            )
            return
        updated = [{"cell": task.cell, "note": note, "gen": self.generation_index}]
        updated.extend(entry for entry in existing if entry.get("note") != note)
        max_items = max(1, int(getattr(cfg, "max_items", len(updated))))
        if len(updated) > max_items:
            updated = updated[:max_items]
        self._knowledge_store[organelle_id] = updated
        self._knowledge_stats_gen["writes"] = self._knowledge_stats_gen.get("writes", 0) + 1

    def _log_colony_event(
        self, meta: dict[str, object], generation: int, event_type: str, **payload: object
    ) -> None:
        events = meta.setdefault("events", [])
        entry: dict[str, object] = {"gen": generation, "type": event_type}
        entry.update(payload)
        if isinstance(events, list):
            events.append(entry)
            if len(events) > 120:
                del events[:-120]
        archive_entry = dict(entry)
        archive_entry["colony"] = meta.get("id")
        self._colony_events_archive.append(archive_entry)
        if len(self._colony_events_archive) > 240:
            self._colony_events_archive = self._colony_events_archive[-240:]

    def _find_member_colony(self, organelle_id: str) -> tuple[str, dict[str, object]] | None:
        for cid, meta in self.colonies.items():
            members = meta.get("members", [])
            if isinstance(members, list) and organelle_id in members:
                meta.setdefault("id", cid)
                return cid, meta
        return None

    def _request_and_apply_policy(self, organelle_id: str) -> None:
        # Count attempt
        try:
            self._policy_attempts_gen += 1
        except Exception:
            self._policy_attempts_gen = 1
        allowed = list(getattr(self.config.policy, "allowed_fields", []))
        key_pairs = []
        for idx, key in enumerate(allowed):
            val = 0.5 if idx == 0 else 0.2
            key_pairs.append(f"{key}={val:.2f}")
        kv_example = " ".join(key_pairs) if key_pairs else ""
        keys_clause = ", ".join(allowed) if allowed else "none"
        example_text = kv_example or "budget_frac=0.55 reserve_ratio=0.30"
        prompt = textwrap.dedent(
            (
                f"You are the budgeting policy module. Ignore any previous question and respond ONLY with "
                f"space-separated key=value pairs covering: {keys_clause}. Example: {example_text}. "
                "Use decimals like 0.55 and `true`/`false` for booleans. Do NOT include prose, Markdown, or JSON. "
                "If you cannot comply, output '{{}}' exactly."
            )
        ).strip()
        try:
            result = self.host.step(
                prompt=prompt,
                intent="choose policy",
                max_routes=1,
                allowed_organelle_ids=[organelle_id],
                recurrent_passes=1,
            )
        except TypeError:
            result = self.host.step(
                prompt=prompt,
                intent="choose policy",
                max_routes=1,
                allowed_organelle_ids=[organelle_id],
            )
        metrics = result.responses.get(organelle_id)
        if metrics is None:
            return
        allowed = list(getattr(self.config.policy, "allowed_fields", []))
        answer_text = str(getattr(metrics, "answer", "")).strip()
        if not answer_text:
            answer_text = str(result.envelope.observation.state.get("answer", ""))
        pol = self._parse_policy_json(answer_text, allowed)
        if not pol:
            fallback = {key: 0.5 for key in allowed}
            if fallback:
                pol = fallback
            else:
                self._penalize_policy_failure(organelle_id, answer_text)
                return
        # Count successful parse
        try:
            self._policy_parsed_gen += 1
        except Exception:
            self._policy_parsed_gen = 1
        # Energy charging for policy request (micro-cost)
        try:
            micro = float(getattr(self.config.policy, "energy_cost", 0.0))
        except Exception:
            micro = 0.0
        if micro > 0.0:
            # Optional scaling by token usage
            try:
                if bool(getattr(self.config.policy, "charge_tokens", False)):
                    cap = max(1, int(getattr(self.config.policy, "token_cap", 64)))
                    scale = min(1.0, float(metrics.tokens) / float(cap))
                    micro = micro * max(0.0, min(1.0, scale))
            except Exception:
                pass
            try:
                if self.host.ledger.energy_balance(organelle_id) >= micro:
                    self.host.ledger.consume_energy(organelle_id, micro)
                    self._policy_cost_total += float(micro)
            except Exception:
                pass
        self._active_policies[organelle_id] = pol
        # Apply immediate effects
        # gate_bias_delta
        try:
            if isinstance(pol.get("gate_bias_delta"), (int, float)):
                genome = self.population.population.get(organelle_id)
                if genome is not None:
                    genome.gate_bias += float(pol["gate_bias_delta"]) * 0.1
        except Exception:
            pass
        # comms preferences
        try:
            genome = self.population.population.get(organelle_id)
            if genome is not None:
                if isinstance(pol.get("read"), bool):
                    genome.read_rate = 1.0 if pol["read"] else 0.0
                if isinstance(pol.get("post"), bool):
                    genome.post_rate = 1.0 if pol["post"] else 0.0
        except Exception:
            pass

        bonus = float(getattr(self.config.policy, "success_bonus", 0.0))
        if bonus > 0.0:
            try:
                self.host.ledger.credit_energy(organelle_id, bonus)
                self._policy_cost_total -= bonus
            except Exception:
                pass

    def _penalize_policy_failure(self, organelle_id: str, answer: str) -> None:
        penalty = float(getattr(self.config.policy, "failure_penalty", 0.0))
        if penalty > 0.0:
            try:
                if self.host.ledger.energy_balance(organelle_id) >= penalty:
                    self.host.ledger.consume_energy(organelle_id, penalty)
                    self._policy_cost_total += penalty
            except Exception:
                pass
        try:
            self._policy_fail_counts += 1
        except Exception:
            self._policy_fail_counts = 1
        snippet = answer.strip().splitlines()
        preview = snippet[0] if snippet else ""
        preview = preview[:160]
        self._policy_failure_samples.append(
            {
                "organelle_id": organelle_id,
                "sample": preview,
            }
        )
        if len(self._policy_failure_samples) > 10:
            self._policy_failure_samples = self._policy_failure_samples[-10:]

    def _attempt_assimilation(self, capped: int | None = None) -> int:
        removable: list[str] = []
        merges = 0
        gating: Dict[str, int] = {
            "canary_failed": 0,
            "low_energy": 0,
            "low_power": 0,
            "no_best_cell": 0,
            "no_activity": 0,
            "cooldown": 0,
            "uplift_below_threshold": 0,
            "cell_merges_exceeded": 0,
            "insufficient_scores": 0,
            "global_probe_failed": 0,
            "holdout_failed": 0,
            "topup_success": 0,
            "topup_roi_blocked": 0,
            "topup_cap_blocked": 0,
            "topup_already_sufficient": 0,
            "topup_disabled": 0,
            "reserve_guard": 0,
            "cautious_skip": 0,
        }
        merges_per_cell: Dict[GridKey, int] = {}
        per_cell_interval = self.config.assimilation_tuning.per_cell_interval
        max_cell_merges = self.config.assimilation_tuning.max_merges_per_cell
        for genome in list(self.population.population.values()):
            # Policy reserve gate: skip assimilation if reserve active
            pol = self._active_policies.get(genome.organelle_id)
            if pol and isinstance(pol.get("reserve_ratio"), (int, float)):
                rr = float(pol["reserve_ratio"]) if pol["reserve_ratio"] is not None else 0.0
                rr = max(
                    getattr(self.config.policy, "reserve_min", 0.0),
                    min(getattr(self.config.policy, "reserve_max", 0.75), rr),
                )
                reserve = rr * 4.0 * self.config.energy.m
                if self.host.ledger.energy_balance(genome.organelle_id) < reserve:
                    self._record_assimilation_gate(
                        reason="reserve_active",
                        organelle_id=genome.organelle_id,
                        details={
                            "reserve": reserve,
                            "balance": self.host.ledger.energy_balance(genome.organelle_id),
                        },
                    )
                    continue
            self._assim_attempt_total += 1

            if self.environment.canary_failed(genome.organelle_id):
                gating["canary_failed"] += 1
                self._record_assimilation_gate(
                    reason="canary_failed",
                    organelle_id=genome.organelle_id,
                    details={"generation": self.generation_index},
                )
                continue

            balance = self.host.ledger.energy_balance(genome.organelle_id)
            balance, topup_info = self._maybe_top_up_energy(genome, balance)
            status = topup_info.get("status", "unknown")
            if status == "credited":
                gating["topup_success"] += 1
            elif status == "skip_low_roi":
                gating["topup_roi_blocked"] += 1
            elif status == "skip_no_capacity":
                gating["topup_cap_blocked"] += 1
            elif status == "already_sufficient":
                gating["topup_already_sufficient"] += 1
            elif status == "disabled":
                gating["topup_disabled"] += 1

            reserve_block = self._should_block_assimilation(genome.organelle_id)
            if reserve_block is not None:
                gating["reserve_guard"] += 1
                self._record_assimilation_gate(
                    reason="reserve_guard",
                    organelle_id=genome.organelle_id,
                    details=reserve_block,
                )
                continue

            hazard_state = self._hazard_state.get(genome.organelle_id, {})
            hazard_active = bool(hazard_state.get("active"))
            cooldown = int(hazard_state.get("cooldown", 0))

            if balance < self.config.energy.m:
                gating["low_energy"] += 1
                self._record_assimilation_gate(
                    reason="low_energy",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "balance": balance,
                        "ticket": self.config.energy.m,
                        "top_up": topup_info,
                    },
                )
                continue

            if hazard_active:
                gating["cautious_skip"] += 1
                self._record_assimilation_gate(
                    reason="cautious_skip",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "z": hazard_state.get("z"),
                        "roi": hazard_state.get("roi"),
                    },
                )
                self._log_survival_event(
                    genome.organelle_id,
                    "cautious_skip",
                    mode="hazard",
                    z=hazard_state.get("z"),
                    roi=hazard_state.get("roi"),
                )
                continue

            min_power_recovery = (
                float(getattr(self.config.survival, "min_power_recovery", 0.0))
                if getattr(self.config, "survival", None)
                else 0.0
            )
            if cooldown > 0 and min_power_recovery > 0.0:
                recent_roi = float(self.population.average_roi(genome.organelle_id, limit=5))
                if recent_roi < min_power_recovery:
                    gating["cautious_skip"] += 1
                    self._record_assimilation_gate(
                        reason="cautious_skip",
                        organelle_id=genome.organelle_id,
                        details={
                            "generation": self.generation_index,
                            "cooldown": cooldown,
                            "roi": recent_roi,
                        },
                    )
                    self._log_survival_event(
                        genome.organelle_id,
                        "cautious_skip",
                        mode="recovery",
                        cooldown=cooldown,
                        roi=recent_roi,
                    )
                    continue

            threshold = self.config.evolution.assimilation_threshold
            hazard_margin = (
                getattr(self.config.survival, "hazard_holdout_margin", 0.0)
                if getattr(self.config, "survival", None)
                else 0.0
            )
            if cooldown > 0 and hazard_margin:
                threshold += float(hazard_margin)

            stats_map = getattr(self.environment, "organism_stats", {})
            if not stats_map.get(genome.organelle_id):
                gating["no_activity"] += 1
                self._record_assimilation_gate(
                    reason="no_activity",
                    organelle_id=genome.organelle_id,
                    details={"generation": self.generation_index},
                )
                continue
            best_cell = self.environment.best_cell_score(genome.organelle_id)
            if best_cell is None:
                gating["no_best_cell"] += 1
                self._record_assimilation_gate(
                    reason="no_best_cell",
                    organelle_id=genome.organelle_id,
                    details={"generation": self.generation_index},
                )
                continue
            cell, ema = best_cell
            family, depth = cell
            key = (genome.organelle_id, cell)
            last_attempt = self.assimilation_cooldown.get(key, -per_cell_interval)
            if self.generation_index - last_attempt < per_cell_interval:
                gating["cooldown"] += 1
                self._record_assimilation_gate(
                    reason="cooldown",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "cooldown_remaining": per_cell_interval
                        - (self.generation_index - last_attempt),
                    },
                )
                continue
            tau = self.config.controller.tau
            if family in {"math", "math.sequence"}:
                tau = min(0.6, tau + 0.08)
            elif family in {"logic.bool", "word.count"}:
                tau = min(0.58, tau + 0.06)
            elif family == "json_repair":
                tau = min(0.57, tau + 0.05)
            relief = float(self._tau_relief.get(cell, 0.0))
            if relief > 0.0:
                tau = max(0.2, tau - relief)
            uplift_gate = ema - tau
            if uplift_gate < threshold:
                self.assimilation_cooldown[key] = self.generation_index
                gating["uplift_below_threshold"] += 1
                self._record_assimilation_gate(
                    reason="uplift_below_threshold",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "ema": ema,
                        "tau": self.config.controller.tau,
                        "uplift_gate": uplift_gate,
                        "threshold": threshold,
                        "relief": relief,
                    },
                )
                self._register_tau_failure(cell)
                continue
            if merges_per_cell.get(cell, 0) >= max_cell_merges:
                gating["cell_merges_exceeded"] += 1
                self._record_assimilation_gate(
                    reason="cell_merges_exceeded",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "cell": {"family": cell[0], "depth": cell[1]},
                        "max_cell_merges": max_cell_merges,
                    },
                )
                continue
            score_records = self.population.recent_score_records(genome.organelle_id, limit=16)
            if not score_records:
                continue
            scores = [float(record.get("score", 0.0) or 0.0) for record in score_records]
            tuning = self.config.assimilation_tuning
            min_samples_required = self._min_samples_required(tuning)
            base_min = self._min_window_requirement(tuning)
            available = len(scores) - (len(scores) % 2)
            token_spent = False
            min_window = base_min if available < base_min else min(base_min, available)
            step = max(2, int(getattr(tuning, "window_step", 2)))
            if available < min_window:
                adjusted_window, tokens_used = self._apply_evidence_tokens(
                    genome.organelle_id,
                    available - (available % 2),
                    min_window,
                    min_samples_required,
                )
                if tokens_used:
                    token_spent = True
                    min_window = adjusted_window
                if not token_spent:
                    self.assimilation_cooldown[key] = self.generation_index
                    gating["insufficient_scores"] += 1
                    self._record_assimilation_gate(
                        reason="insufficient_scores",
                        organelle_id=genome.organelle_id,
                        details={
                            "generation": self.generation_index,
                            "scores_available": len(scores),
                            "min_window": min_window,
                        },
                    )
                    continue
            window_len = available
            while window_len > min_window and window_len - step >= min_window:
                window_len_candidate = window_len - step
                if window_len_candidate % 2 != 0:
                    window_len_candidate -= 1
                if window_len_candidate < min_window:
                    break
                window_len = window_len_candidate
                break
            start_idx = len(scores) - window_len
            window = scores[start_idx:]
            window_records = score_records[start_idx:]
            split = window_len // 2
            control = window[:split]
            treatment = window[split:]
            control_records = window_records[:split]
            treatment_records = window_records[split:]
            if len(control) < min_samples_required or len(treatment) < min_samples_required:
                self.assimilation_cooldown[key] = self.generation_index
                gating["insufficient_scores"] += 1
                self._record_assimilation_gate(
                    reason="insufficient_scores_window",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "window_len": window_len,
                    },
                )
                continue
            avg_energy = self.population.average_energy(genome.organelle_id)
            result = self.assimilation.evaluate(
                organelle_id=genome.organelle_id,
                control_scores=control,
                treatment_scores=treatment,
                safety_hits=0,
                energy_cost=avg_energy * len(treatment),
                energy_balance=balance,
                energy_top_up=topup_info,
                control_meta=[dict(record) for record in control_records],
                treatment_meta=[dict(record) for record in treatment_records],
            )
            self.assimilation_cooldown[key] = self.generation_index
            # If statistical power is too low, defer and expand evidence window next time
            try:
                min_power = float(getattr(self.config.assimilation_tuning, "trial_min_power", 0.1))
            except Exception:
                min_power = 0.1
            try:
                dr_min_power = float(
                    getattr(self.config.assimilation_tuning, "dr_min_power", min_power)
                )
                min_power = max(min_power, dr_min_power)
            except Exception:
                pass
            if result.event.power is not None and result.event.power < min_power:
                gating["low_power"] += 1
                self._record_assimilation_gate(
                    reason="low_power",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "power": float(result.event.power),
                        "min_power": float(min_power),
                        "sample_size": int(result.event.sample_size or 0),
                    },
                )
                try:
                    self.population.grant_evidence(genome.organelle_id, 1)
                except Exception:
                    pass
                continue
            # DR small-n refinement: if DR used but effective sample is too small, defer
            try:
                if bool(result.event.dr_used):
                    dr_sizes = result.event.dr_sample_sizes or {}
                    min_stratum = int(
                        getattr(self.config.assimilation_tuning, "dr_min_stratum_size", 2)
                    )
                    contributing = 0
                    for _k, s in dr_sizes.items():
                        try:
                            if int(s.get("paired", 0)) >= min_stratum:
                                contributing += 1
                        except Exception:
                            continue
                    sample_n = int(result.event.sample_size or 0)
                    if contributing < 1 or sample_n < max(2 * min_stratum, 6):
                        gating["low_power"] += 1
                        self._record_assimilation_gate(
                            reason="low_power_dr",
                            organelle_id=genome.organelle_id,
                            details={
                                "generation": self.generation_index,
                                "contributing_strata": int(contributing),
                                "min_stratum": int(min_stratum),
                                "sample_size": sample_n,
                            },
                        )
                        try:
                            self.population.grant_evidence(genome.organelle_id, 1)
                        except Exception:
                            pass
                        continue
            except Exception:
                pass
            probe_records: list[dict[str, object]] = []
            soup_records: list[dict[str, float]] = []
            holdout_info: dict[str, object] | None = None
            audit_info: dict[str, object] | None = None
            decision_final = False
            attempt_detail: dict[str, object] = {
                "generation": self.generation_index,
                "organelle_id": genome.organelle_id,
                "cell": {"family": cell[0], "depth": cell[1]},
                "uplift": float(result.event.uplift),
                "p_value": float(result.event.p_value),
                "sample_size": result.event.sample_size,
                "threshold": self.config.evolution.assimilation_threshold,
                "uplift_threshold": result.event.uplift_threshold,
                "p_value_threshold": result.event.p_value_threshold,
                "control_mean": result.event.control_mean,
                "treatment_mean": result.event.treatment_mean,
                "control_std": result.event.control_std,
                "treatment_std": result.event.treatment_std,
                "energy_balance": balance,
                "top_up": dict(topup_info),
                "tau_relief": float(relief),
                "roi_relief": float(self._roi_relief.get(genome.organelle_id, 0.0)),
                "method": result.event.method,
                "dr_used": bool(result.event.dr_used),
                "strata": dict(result.event.dr_sample_sizes),
                "passes_stat_test": bool(result.decision),
            }
            if not result.decision:
                self.assim_fail_streak += 1
                self._maybe_decay_assimilation_thresholds()
                self._register_tau_failure(cell)
            if result.decision and (capped is None or merges < capped):
                soup_ids, stats_map = self._select_soup_members(cell, genome.organelle_id)
                holdout_ok, holdout_info = self._holdout_accepts(
                    genome.organelle_id,
                    [oid for oid in soup_ids if oid != genome.organelle_id],
                )
                attempt_detail["holdout"] = holdout_info or {}
                attempt_detail["holdout_passed"] = bool(holdout_ok)
                if not holdout_ok:
                    gating["holdout_failed"] += 1
                    self.assim_fail_streak += 1
                    self._maybe_decay_assimilation_thresholds()
                    self._record_assimilation_gate(
                        reason="holdout_failed",
                        organelle_id=genome.organelle_id,
                        details={
                            "generation": self.generation_index,
                            "holdout": holdout_info or {},
                        },
                    )
                    self._register_tau_failure(cell)
                elif self._global_probe(genome.organelle_id, cell, gating):
                    probe_records = self._run_hf_probes(genome.organelle_id)
                    soup_records = self._apply_lora_soup_merge(
                        cell,
                        genome.organelle_id,
                        soup_ids,
                        stats_map,
                        probe_records,
                    )
                    audit_info = None
                    if getattr(self.config.assimilation_tuning, "merge_audit_enabled", False):
                        try:
                            tasks = self._sample_holdout_tasks()
                            post_roi = self._evaluate_holdout_roi(genome.organelle_id, tasks)
                            pre_roi = None
                            if holdout_info:
                                pre_roi = holdout_info.get("candidate_roi")
                            audit_info = {
                                "post_roi": float(post_roi),
                                "pre_roi": float(pre_roi) if pre_roi is not None else None,
                                "tasks": len(tasks),
                            }
                        except Exception:
                            audit_info = {"post_roi": None, "pre_roi": None, "tasks": 0}
                    if audit_info is not None:
                        attempt_detail["audit"] = audit_info
                        pre_roi_val = audit_info.get("pre_roi")
                        post_roi_val = audit_info.get("post_roi")
                        delta_val: float | None = None
                        if isinstance(pre_roi_val, (int, float)) and isinstance(
                            post_roi_val, (int, float)
                        ):
                            delta_val = float(post_roi_val) - float(pre_roi_val)
                        audit_info["delta"] = delta_val
                        audit_entry = {
                            "generation": self.generation_index,
                            "organelle_id": genome.organelle_id,
                            "cell": {"family": cell[0], "depth": cell[1]},
                            "pre_roi": audit_info.get("pre_roi"),
                            "post_roi": audit_info.get("post_roi"),
                            "delta": delta_val,
                            "tasks": int(audit_info.get("tasks", 0) or 0),
                        }
                        self._merge_audits_gen.append(self._sanitize_telemetry(audit_entry))
                    self._spawn_replacement_from(genome)
                    removable.append(genome.organelle_id)
                    merges += 1
                    merges_per_cell[cell] = merges_per_cell.get(cell, 0) + 1
                    decision_final = True
                    self.assim_fail_streak = 0
                    self._register_tau_success(cell)
                    attempt_detail["global_probe_passed"] = True
                else:
                    attempt_detail["global_probe_passed"] = False
                    self._register_tau_failure(cell)
            else:
                attempt_detail["holdout_passed"] = False
                attempt_detail["global_probe_passed"] = False
                # Offspring merge path: allow energy-eligible trial child
                mode = getattr(self.config.assimilation_tuning, "merge_mode", "strict")
                enable_trials = bool(
                    getattr(self.config.assimilation_tuning, "trial_offspring_enabled", False)
                )
                cap = int(getattr(self.config.assimilation_tuning, "trial_per_gen_cap", 0))
                if (
                    enable_trials
                    and mode in {"offspring", "hybrid"}
                    and self.trial_creations_this_gen < cap
                ):
                    soup_ids, stats_map = self._select_soup_members(cell, genome.organelle_id)
                    child_id = self._create_trial_offspring(
                        cell, genome.organelle_id, soup_ids, stats_map
                    )
                    if child_id:
                        self.trial_creations_this_gen += 1
                        attempt_detail["trial_offspring_created"] = child_id
                # Router-team acceptance gate: try a routed pair as a colony candidate
                try:
                    soup_ids, _stats_map = self._select_soup_members(cell, genome.organelle_id)
                    partner = None
                    for oid in soup_ids:
                        if oid != genome.organelle_id:
                            partner = oid
                            break
                    if partner is not None:
                        tasks = self._sample_holdout_tasks()
                        if tasks:
                            # Collect per-task ROI series for team (best-of-two) and bases
                            energy_cfg = self.config.energy
                            team_series: list[float] = []
                            a_series: list[float] = []
                            b_series: list[float] = []
                            for index, task in enumerate(tasks, start=1):
                                grid_task = task.to_grid_task(
                                    self.environment, task_id=f"team_{index:04d}"
                                )
                                best_roi = 0.0
                                for oid in (genome.organelle_id, partner):
                                    result_i = self.host.step(
                                        prompt=grid_task.prompt,
                                        intent="team holdout",
                                        max_routes=1,
                                        allowed_organelle_ids=[oid],
                                    )
                                    metrics_i = result_i.responses.get(oid)
                                    if metrics_i is None:
                                        continue
                                    success_i, reward_i = grid_task.evaluate(metrics_i.answer)
                                    revenue_i = grid_task.price * reward_i.total
                                    cost_i = (
                                        energy_cfg.alpha * metrics_i.flops_estimate
                                        + energy_cfg.beta * metrics_i.memory_gb
                                        + energy_cfg.gamma * metrics_i.latency_ms
                                        + energy_cfg.lambda_p * metrics_i.trainable_params
                                    )
                                    roi_i = (
                                        (float("inf") if revenue_i > 0 else 0.0)
                                        if cost_i <= 0.0
                                        else (revenue_i / cost_i)
                                    )
                                    roi_i = (
                                        0.0
                                        if not math.isfinite(roi_i)
                                        else float(max(0.0, min(roi_i, 10.0)))
                                    )
                                    if oid == genome.organelle_id:
                                        a_series.append(roi_i)
                                    else:
                                        b_series.append(roi_i)
                                    best_roi = max(best_roi, roi_i)
                                team_series.append(best_roi)
                            # Compute acceptance using CI on team mean vs baseline mean
                            base_a = sum(a_series) / max(len(a_series), 1)
                            base_b = sum(b_series) / max(len(b_series), 1)
                            baseline = max(base_a, base_b)
                            ci_low, ci_high, team_mu, team_se = self._compute_mean_ci(team_series)
                            margin = float(self.config.assimilation_tuning.holdout_margin)
                            min_power_tasks = 8
                            accept = len(team_series) >= min_power_tasks and (
                                ci_low > baseline + margin
                            )
                            if accept:
                                cid = f"col_{genome.organelle_id[:4]}_{partner[:4]}"
                                meta = self.colonies.setdefault(
                                    cid,
                                    {
                                        "members": [genome.organelle_id, partner],
                                        "pot": 0.0,
                                        "reserve_ratio": 0.25,
                                        "created_gen": self.generation_index,
                                    },
                                )
                                self._log_colony_event(
                                    meta,
                                    self.generation_index,
                                    "create",
                                    members=list(meta.get("members", [])),
                                )
                                self.promotions_this_gen += 1
                                attempt_detail["team_promoted_colony"] = cid
                            # log team acceptance stats
                            attempt_detail["team_holdout"] = {
                                "team_roi": float(team_mu),
                                "base_a": float(base_a),
                                "base_b": float(base_b),
                                "delta": float(team_mu - baseline),
                                "ci_low": float(ci_low),
                                "ci_high": float(ci_high),
                                "tasks": int(len(team_series)),
                                "min_tasks": int(min_power_tasks),
                            }
                except Exception:
                    pass
            self._record_assimilation_attempt(attempt_detail)
            history_record = {
                "generation": self.generation_index,
                "uplift": float(result.event.uplift),
                "p_value": float(result.event.p_value),
                "passed": bool(decision_final),
                "ema": float(
                    self.environment.organism_stats.get(genome.organelle_id, {}).get(cell, 0.0)
                ),
                "roi": float(self.population.average_roi(genome.organelle_id)),
                "probes": probe_records,
                "holdout": holdout_info,
                "sample_size": result.event.sample_size,
                "control_mean": result.event.control_mean,
                "control_std": result.event.control_std,
                "treatment_mean": result.event.treatment_mean,
                "treatment_std": result.event.treatment_std,
                "energy_balance": result.event.energy_balance,
                "energy_top_up": (
                    result.event.energy_top_up.model_dump(mode="json")
                    if result.event.energy_top_up is not None
                    else None
                ),
            }
            if audit_info is not None:
                history_record["audit"] = audit_info
            if self.sink:
                event = result.event.model_copy(
                    update={
                        "cell": {"family": cell[0], "depth": cell[1]},
                        "probes": probe_records,
                        "soup": soup_records,
                        "holdout": holdout_info,
                        "audit": audit_info,
                    }
                )
                self.sink.log_assimilation(event, decision_final)
            self.population.record_assimilation(
                genome.organelle_id,
                cell,
                history_record,
            )
            if token_spent:
                attempt_detail["evidence_token_used"] = True
        for organelle_id in removable:
            self.host.retire_organelle(organelle_id)
            self.population.remove(organelle_id)
        # expose gating counts for diagnostics in generation summary
        try:
            self.assim_gating_counts = gating  # type: ignore[attr-defined]
            self.assim_gating_samples_snapshot = list(self.assim_gating_samples[-24:])  # type: ignore[attr-defined]
            self.assim_attempt_samples_snapshot = list(self.assim_attempt_samples[-24:])  # type: ignore[attr-defined]
        except Exception:
            pass
        return merges

    def _create_trial_offspring(
        self,
        cell: GridKey,
        candidate_id: str,
        soup_ids: list[str],
        stats_map: dict[str, dict[str, float]],
    ) -> str | None:
        try:
            organelle = self.host.get_organelle(candidate_id)
            target_rank = (
                int(getattr(organelle, "rank", self.config.host.max_lora_rank))
                if organelle is not None
                else self.config.host.max_lora_rank
            )
            target_rank = max(1, min(target_rank, self.config.host.max_lora_rank))
            # weights based on roi*ema, same as merge
            weights: list[float] = []
            for oid in soup_ids:
                stats = stats_map.get(oid, {"roi": 0.0, "ema": 0.0})
                w = (max(stats.get("roi", 0.0), 0.0) + 1e-3) * (stats.get("ema", 0.0) + 1e-3)
                weights.append(w)
            total = sum(weights) or 1.0
            soup = {oid: w / total for oid, w in zip(soup_ids, weights)}
            soup_state, _alpha = self.host.build_lora_soup_state(soup, target_rank)
            child_id = self.host.spawn_organelle(rank=target_rank)
            child = self.host.get_organelle(child_id)
            if child is None:
                return None
            child.import_adapter_state(soup_state, alpha=1.0)
            # stipend energy
            stipend = float(getattr(self.config.assimilation_tuning, "trial_stipend", 0.5))
            self.host.ledger.set_energy(child_id, stipend)
            # register in population
            donors = [
                self.population.population.get(oid)
                for oid in soup_ids
                if oid in self.population.population
            ]

            def _avg_trait(attr: str) -> float:
                values = [float(getattr(gen, attr, 0.0)) for gen in donors if gen is not None]
                if not values:
                    return 0.0
                return max(0.0, min(1.0, sum(values) / len(values)))

            self.population.register(
                Genome(
                    organelle_id=child_id,
                    drive_weights={"novelty": 0.1},
                    gate_bias=0.0,
                    rank=target_rank,
                    explore_rate=_avg_trait("explore_rate"),
                    post_rate=_avg_trait("post_rate"),
                    read_rate=_avg_trait("read_rate"),
                    hint_weight=_avg_trait("hint_weight"),
                )
            )
            # track probation
            self.trial_offspring[child_id] = {
                "parents": list(soup_ids),
                "cell": {"family": cell[0], "depth": cell[1]},
                "created_gen": self.generation_index,
                "probation_left": int(
                    getattr(self.config.assimilation_tuning, "trial_probation_gens", 5)
                ),
                "promoted": False,
            }
            return child_id
        except Exception:
            return None

    def _review_trial_offspring(self) -> None:
        if not self.trial_offspring:
            return
        margin = float(getattr(self.config.assimilation_tuning, "trial_promote_margin", 0.02))
        min_power = float(getattr(self.config.assimilation_tuning, "trial_min_power", 0.1))
        to_remove: list[str] = []
        for child_id, meta in list(self.trial_offspring.items()):
            if bool(meta.get("promoted", False)):
                to_remove.append(child_id)
                continue
            # sample holdout and compare to baseline mates
            tasks = self._sample_holdout_tasks()
            child_roi = self._evaluate_holdout_roi(str(child_id), tasks)
            parents: list[str] = [str(x) for x in meta.get("parents", [])]
            baselines = [self._evaluate_holdout_roi(pid, tasks) for pid in parents]
            baselines = [b for b in baselines if math.isfinite(b)]
            best_base = max(baselines, default=float("-inf"))
            target = (best_base if math.isfinite(best_base) else 0.0) + margin
            # accumulate evidence across generations
            ev = meta.setdefault("evidence", {"deltas": []})
            try:
                deltas: list[float] = list(ev.get("deltas", []))
            except Exception:
                deltas = []
            uplift = float(child_roi - (best_base if math.isfinite(best_base) else 0.0))
            deltas.append(uplift)
            if len(deltas) > 24:
                deltas = deltas[-24:]
            ev["deltas"] = deltas
            # simple power proxy for mean uplift against margin
            n = len(deltas)
            if n >= 2:
                mean_u = float(sum(deltas) / n)
                var_u = float(sum((x - mean_u) ** 2 for x in deltas) / max(n - 1, 1))
                se = math.sqrt(max(var_u, 1e-12) / n)
                delta = mean_u - margin
                z = delta / max(se, 1e-12)
                z_alpha = 1.64
                from math import erf, sqrt

                def phi(x: float) -> float:
                    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

                power = max(0.0, min(1.0, 1.0 - float(phi(z_alpha - z))))
            else:
                mean_u = uplift
                power = 0.0
            if child_roi >= target or (power >= min_power and mean_u >= 0.0):
                # promote
                self.promotions_this_gen += 1
                meta["promoted"] = True
                to_remove.append(child_id)
                # record as assimilation success in history/logs for transparency
                if self.sink:
                    event = AssimilationEvent(
                        organelle_id=str(child_id),
                        uplift=child_roi - (best_base if math.isfinite(best_base) else 0.0),
                        p_value=1.0,
                        passed=True,
                        energy_cost=0.0,
                        safety_hits=0,
                        sample_size=len(tasks),
                        control_mean=best_base if math.isfinite(best_base) else 0.0,
                        treatment_mean=child_roi,
                        control_std=0.0,
                        treatment_std=0.0,
                        ci_low=None,
                        ci_high=None,
                        power=power,
                        uplift_threshold=self.config.evolution.assimilation_threshold,
                        p_value_threshold=self.config.evolution.assimilation_p_value,
                        energy_balance=self.host.ledger.energy_balance(str(child_id)),
                        energy_top_up=None,
                        cell=meta.get("cell"),
                    )
                    self.sink.log_assimilation(event, True)
                continue
            # probation countdown
            meta["probation_left"] = int(meta.get("probation_left", 0)) - 1
            if int(meta["probation_left"]) <= 0:
                # cull child
                to_remove.append(child_id)
        for cid in to_remove:
            # if still present, retire
            self.host.retire_organelle(cid)
            self.population.remove(cid)
            self.trial_offspring.pop(cid, None)

    @staticmethod
    def _sanitize_telemetry(value: object) -> object:
        return sanitize_telemetry(value)

    def _record_assimilation_gate(
        self, reason: str, organelle_id: str, details: dict[str, object]
    ) -> None:
        limit = max(1, int(getattr(self.config.assimilation_tuning, "gating_snapshot_limit", 48)))
        sample = {
            "generation": self.generation_index,
            "organelle_id": organelle_id,
            "reason": reason,
            "details": self._sanitize_telemetry(details),
        }
        self.assim_gating_samples.append(sample)
        if len(self.assim_gating_samples) > limit:
            self.assim_gating_samples = self.assim_gating_samples[-limit:]

    def _record_assimilation_attempt(self, details: dict[str, object]) -> None:
        limit = max(1, int(getattr(self.config.assimilation_tuning, "gating_snapshot_limit", 48)))
        sample = self._sanitize_telemetry(details)
        self.assim_attempt_samples.append(sample)
        if len(self.assim_attempt_samples) > limit:
            self.assim_attempt_samples = self.assim_attempt_samples[-limit:]

    def _register_tau_failure(self, cell: GridKey) -> None:
        window = int(getattr(self.config.assimilation_tuning, "tau_relief_window", 12))
        step = float(getattr(self.config.assimilation_tuning, "tau_relief_step", 0.01))
        max_relief = float(getattr(self.config.assimilation_tuning, "tau_relief_max", 0.15))
        if window <= 0 or step <= 0.0 or max_relief <= 0.0:
            return
        count = self._tau_fail_counts.get(cell, 0) + 1
        self._tau_fail_counts[cell] = count
        if count >= window:
            relief = min(max_relief, self._tau_relief.get(cell, 0.0) + step)
            self._tau_relief[cell] = relief
            self._tau_fail_counts[cell] = 0

    def _register_tau_success(self, cell: GridKey) -> None:
        if cell not in self._tau_relief:
            self._tau_fail_counts.pop(cell, None)
            return
        step = float(getattr(self.config.assimilation_tuning, "tau_relief_step", 0.01))
        relief = max(0.0, self._tau_relief.get(cell, 0.0) - step)
        if relief <= 1e-6:
            self._tau_relief.pop(cell, None)
        else:
            self._tau_relief[cell] = relief
        self._tau_fail_counts.pop(cell, None)

    def _register_roi_skip(self, organelle_id: str) -> None:
        window = int(getattr(self.config.assimilation_tuning, "roi_relief_window", 8))
        step = float(getattr(self.config.assimilation_tuning, "roi_relief_step", 0.05))
        max_relief = float(getattr(self.config.assimilation_tuning, "roi_relief_max", 0.5))
        if window <= 0 or step <= 0.0 or max_relief <= 0.0:
            return
        count = self._topup_fail_counts.get(organelle_id, 0) + 1
        self._topup_fail_counts[organelle_id] = count
        if count >= window:
            relief = min(max_relief, self._roi_relief.get(organelle_id, 0.0) + step)
            self._roi_relief[organelle_id] = relief
            self._topup_fail_counts[organelle_id] = 0

    def _register_roi_success(self, organelle_id: str) -> None:
        if organelle_id in self._roi_relief:
            step = float(getattr(self.config.assimilation_tuning, "roi_relief_step", 0.05))
            relief = max(0.0, self._roi_relief.get(organelle_id, 0.0) - step)
            if relief <= 1e-6:
                self._roi_relief.pop(organelle_id, None)
            else:
                self._roi_relief[organelle_id] = relief
        self._topup_fail_counts.pop(organelle_id, None)

    def _update_qd_archive(self) -> int:
        if not getattr(self.config.qd, "enabled", False):
            return len(self._qd_archive)
        # cost bins as in _select_soup_members
        energies = {
            oid: float(self.population.average_energy(oid))
            for oid in self.population.population.keys()
        }
        vals = sorted(v for v in energies.values() if math.isfinite(v))
        bins: list[float] = []
        if vals:
            try:
                qs = quantiles(
                    vals, n=max(2, getattr(self.config.qd, "cost_bins", 3)), method="inclusive"
                )
                bins = [float(x) for x in qs]
            except Exception:
                bins = [vals[len(vals) // 2]]

        def cost_bin(val: float) -> int:
            if not bins:
                return 0
            for i, edge in enumerate(bins):
                if val <= edge:
                    return i
            return len(bins)

        for oid in self.population.population.keys():
            best = self.environment.best_cell_score(oid)
            if not best:
                continue
            (family, depth), ema = best
            key = (str(family), str(depth), cost_bin(energies.get(oid, 0.0)))
            roi = float(self.population.average_roi(oid, limit=5))
            novelty = float(
                self.population.cell_novelty(
                    oid,
                    (str(family), str(depth)),
                    scale=float(getattr(self.config.qd, "novelty_weight", 0.3)),
                    floor=float(getattr(self.config.qd, "novelty_min", 0.05)),
                )
            )
            prev = self._qd_archive.get(key)
            if prev is None or roi > float(prev.get("roi", 0.0)):
                self._qd_archive[key] = {
                    "organelle_id": oid,
                    "roi": roi,
                    "ema": float(ema),
                    "energy": energies.get(oid, 0.0),
                    "novelty": novelty,
                }
        cap = max(1, int(getattr(self.config.qd, "archive_cap", 256)))
        if len(self._qd_archive) > cap:
            excess = len(self._qd_archive) - cap
            worst_keys = sorted(
                self._qd_archive.items(),
                key=lambda kv: (float(kv[1].get("roi", 0.0)), float(kv[1].get("novelty", 0.0))),
            )[:excess]
            for key, _ in worst_keys:
                self._qd_archive.pop(key, None)
        return len(self._qd_archive)

    def _sample_team_synergy(self) -> None:
        # Sample up to 2 top ROI pairs and log simple synergy stats on matched holdout
        ids = list(self.population.population.keys())
        if len(ids) < 2:
            return
        ranked = sorted(
            ids, key=lambda oid: self.population.average_roi(oid, limit=5), reverse=True
        )
        pairs = [(ranked[i], ranked[i + 1]) for i in range(0, min(len(ranked) - 1, 3), 2)]
        tasks = self._sample_holdout_tasks()
        if not tasks:
            return
        tune = self.config.assimilation_tuning
        delta_cfg = float(getattr(tune, "team_probe_synergy_delta", 0.12))
        nu_cfg = float(getattr(tune, "team_probe_variance_nu", 0.25))
        sustain_required = max(1, int(getattr(tune, "team_probe_sustain", 3)))
        for a, b in pairs:
            solo_a_stats = self._team_holdout_stats([a], tasks)
            solo_b_stats = self._team_holdout_stats([b], tasks)
            team_stats = self._team_holdout_stats([a, b], tasks)
            solo_means = [float(solo_a_stats["mean"]), float(solo_b_stats["mean"])]
            solo_vars = [float(solo_a_stats["variance"]), float(solo_b_stats["variance"])]
            team_mean = float(team_stats["mean"])
            solo_sum = sum(solo_means)
            synergy_sum = team_mean - solo_sum
            delta = team_mean - max(solo_means)
            min_solo_var = min((v for v in solo_vars if v > 1e-6), default=0.0)
            if min_solo_var > 0.0:
                variance_ratio = float(team_stats["variance"]) / min_solo_var
                variance_delta = float(team_stats["variance"]) - min_solo_var
            else:
                variance_ratio = 0.0 if float(team_stats["variance"]) <= 1e-6 else 1.0
                variance_delta = float(team_stats["variance"])
            synergy = delta
            pair = (min(a, b), max(a, b))
            meets_synergy = solo_sum > 0.0 and synergy_sum >= delta_cfg * solo_sum
            if min_solo_var > 1e-6:
                meets_variance = variance_delta <= -nu_cfg * min_solo_var
            else:
                meets_variance = float(team_stats["variance"]) <= nu_cfg
            pass_condition = meets_synergy and meets_variance
            sustain_count = 0
            if pass_condition:
                sustain_count = self._synergy_sustain.get(pair, 0) + 1
            else:
                sustain_count = 0
            self._synergy_sustain[pair] = sustain_count
            sample = {
                "generation": self.generation_index,
                "a": a,
                "b": b,
                "solo_a": float(solo_means[0]),
                "solo_b": float(solo_means[1]),
                "team": team_mean,
                "synergy": float(synergy),
                "synergy_sum": float(synergy_sum),
                "solo_sum": float(solo_sum),
                "tasks": int(len(tasks)),
                "team_var": float(team_stats["variance"]),
                "solo_var_min": float(min_solo_var),
                "variance_ratio": float(variance_ratio),
                "variance_delta": float(variance_delta),
                "passes_threshold": bool(pass_condition),
                "sustain": sustain_count,
            }
            self._synergy_window.append(self._sanitize_telemetry(sample))
            if len(self._synergy_window) > 24:
                self._synergy_window = self._synergy_window[-24:]
            if pass_condition and sustain_count >= sustain_required:
                candidate = {
                    "pair": pair,
                    "generation": self.generation_index,
                    "synergy_sum": float(synergy_sum),
                    "solo_sum": float(solo_sum),
                    "variance_delta": float(variance_delta),
                    "variance_ratio": float(variance_ratio),
                    "sustain": sustain_count,
                }
                self._team_probe_candidates_gen.append(candidate)

    def _maybe_promote_colonies(self) -> None:
        if not bool(getattr(self.config.assimilation_tuning, "colonies_enabled", False)):
            return
        windows = int(getattr(self.config.assimilation_tuning, "colony_windows", 3))
        delta = float(getattr(self.config.assimilation_tuning, "colony_synergy_delta", 0.1))
        recent = self._synergy_window[-(windows * 4) :]
        by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
        for rec in recent:
            a = str(rec.get("a"))
            b = str(rec.get("b"))
            pair = (min(a, b), max(a, b))
            by_pair.setdefault(pair, []).append(rec)
        variance_improve = float(
            getattr(self.config.assimilation_tuning, "colony_variance_improve", 0.2)
        )
        margin = float(getattr(self.config.assimilation_tuning, "holdout_margin", 0.03))
        review_interval = int(
            getattr(self.config.assimilation_tuning, "colony_review_interval", max(3, windows))
        )
        required_passes = int(getattr(self.config.assimilation_tuning, "colony_required_passes", 2))
        expand_delta = float(
            getattr(self.config.assimilation_tuning, "colony_expand_delta", margin)
        )
        for (a, b), records in by_pair.items():
            if len(records) < windows:
                continue
            window_records = records[-windows:]
            if not (a in self.population.population and b in self.population.population):
                continue
            if not all(isinstance(rec.get("synergy"), (int, float)) for rec in window_records):
                continue
            if not all(float(rec.get("synergy", 0.0)) >= delta for rec in window_records):
                continue
            if not all(float(rec.get("synergy", 0.0)) >= expand_delta for rec in window_records):
                continue
            variance_pass = True
            for rec in window_records:
                ratio = float(rec.get("variance_ratio", 1.0))
                if ratio > 1.0 - variance_improve + 1e-6:
                    variance_pass = False
                    break
            if not variance_pass:
                continue
            mean_s = sum(float(rec.get("synergy", 0.0)) for rec in window_records) / windows
            cid = f"col_{a[:4]}_{b[:4]}"
            if cid in self.colonies:
                continue
            colony_meta = self._create_colony_meta(cid, [a, b])
            colony_meta.update(
                {
                    "last_delta": mean_s,
                    "margin": margin,
                    "last_variance_ratio": float(window_records[-1].get("variance_ratio", 1.0)),
                }
            )
            colony_meta["review_interval"] = review_interval
            colony_meta["required_passes"] = required_passes
            colony_meta["holdout_passes"] = 0
            colony_meta["holdout_failures"] = 0
            colony_meta["last_review"] = self.generation_index
            colony_meta.setdefault("expand_history", [])
            self.colonies[cid] = colony_meta
            self._log_colony_event(
                colony_meta,
                self.generation_index,
                "create",
                members=list(colony_meta.get("members", [])),
                reason="synergy",
            )

    def _colony_refresh_roles(self, meta: dict[str, object]) -> None:
        members = [str(x) for x in meta.get("members", [])]
        meta["members"] = members
        roles: dict[str, int] = {}
        for idx, member in enumerate(members):
            roles[member] = idx
        meta["roles"] = roles

    def _create_colony_meta(
        self,
        cid: str,
        members: list[str],
        *,
        template: dict[str, object] | None = None,
        pot: float = 0.0,
    ) -> dict[str, object]:
        cfg = self.config.assimilation_tuning
        meta: dict[str, object] = {
            "id": cid,
            "members": list(members),
            "pot": float(pot),
            "reserve_ratio": float(getattr(cfg, "colony_reserve_ratio", 0.25)),
            "created_gen": self.generation_index,
            "last_review": self.generation_index,
            "holdout_passes": 0,
            "holdout_failures": 0,
            "review_interval": int(getattr(cfg, "colony_review_interval", 6)),
            "required_passes": int(getattr(cfg, "colony_required_passes", 2)),
            "expand_history": [],
        }
        tier_count = int(getattr(cfg, "colony_tier_count", 1))
        base_tier = 0
        if template is not None:
            base_tier = int(template.get("tier", 0))
        base_tier = max(0, min(tier_count - 1, base_tier))
        meta["tier"] = base_tier
        meta["tier_cooldown"] = 0
        base_tax = float(getattr(cfg, "colony_tax_rate", 0.1))
        base_subsidy = float(getattr(cfg, "colony_subsidy_fraction", 0.25))
        mutation_scale = float(getattr(cfg, "colony_trait_mutation_scale", 0.05))
        if template is not None:
            tax = float(template.get("tax_rate", base_tax))
            subsidy = float(template.get("subsidy_frac", base_subsidy))
            cohesion = float(template.get("cohesion_weight", 0.5))
            tax += random.gauss(0.0, mutation_scale)
            subsidy += random.gauss(0.0, mutation_scale)
            cohesion += random.gauss(0.0, mutation_scale)
            meta["comms_bonus"] = float(template.get("comms_bonus", 0.0))
        else:
            tax = base_tax
            subsidy = base_subsidy
            cohesion = 0.5
            meta["comms_bonus"] = 0.0
        meta["tax_rate"] = max(0.0, min(0.5, tax))
        meta["subsidy_frac"] = max(0.0, min(1.0, subsidy))
        meta["cohesion_weight"] = max(0.0, min(1.0, cohesion))
        meta["roi_mean"] = 0.0
        meta["fitness"] = 0.0
        meta["fitness_components"] = {"roi_mean": 0.0, "pot_ratio": 0.0, "bandwidth": 0.0}
        self._colony_refresh_roles(meta)
        return meta

    def _dissolve_colony(
        self, cid: str, *, reason: str, reassign_pot: bool = False
    ) -> tuple[list[str], float]:
        meta = self.colonies.get(cid)
        if meta is None:
            return [], 0.0
        members = [str(x) for x in meta.get("members", [])]
        pot = float(meta.get("pot", 0.0))
        share = pot / max(len(members), 1) if members else 0.0
        inherited_pot = pot if reassign_pot else 0.0
        if share > 0.0 and not reassign_pot:
            for member in members:
                try:
                    self.host.ledger.credit_energy(member, share)
                except Exception:
                    pass
        meta["pot"] = 0.0
        self._log_colony_event(
            meta,
            self.generation_index,
            "dissolve",
            reason=reason,
            members=list(members),
            pot=float(pot),
        )
        self.colonies.pop(cid, None)
        return members, float(inherited_pot)

    def _replicate_colony(
        self, parent_cid: str, candidates: list[str], *, inherited_pot: float = 0.0
    ) -> dict[str, object] | None:
        parent = self.colonies.get(parent_cid)
        if parent is None:
            return None
        cfg = self.config.assimilation_tuning
        min_size = int(getattr(cfg, "colony_min_size", 2))
        pool = [oid for oid in candidates if oid in self.population.population]
        if len(pool) < min_size:
            return None
        pool.sort(key=lambda oid: self.population.average_roi(oid, limit=5), reverse=True)
        members = pool[:min_size]
        for member in members:
            if member in candidates:
                candidates.remove(member)
        parent_pot = max(0.0, float(parent.get("pot", 0.0)))
        max_share = parent_pot * 0.25
        pot_share = min(max_share, parent_pot)
        parent["pot"] = parent_pot - pot_share
        inherited_pot = max(0.0, float(inherited_pot))
        child_cid = f"{parent_cid}_c{self.generation_index}"
        child_meta = self._create_colony_meta(
            child_cid, members, template=parent, pot=pot_share + inherited_pot
        )
        self.colonies[child_cid] = child_meta
        self._log_colony_event(
            parent,
            self.generation_index,
            "selection_win",
            child=child_cid,
            fitness=float(parent.get("fitness", 0.0)),
        )
        self._log_colony_event(
            child_meta,
            self.generation_index,
            "create",
            members=list(members),
            parent=parent_cid,
            inherited_pot=float(inherited_pot),
        )
        return child_meta

    def _colony_selection_step(self) -> None:
        cfg = self.config.assimilation_tuning
        if not getattr(cfg, "colony_selection_enabled", False):
            return
        interval = max(1, int(getattr(cfg, "colony_selection_interval", 20)))
        if self.generation_index % interval != 0:
            return
        if len(self.colonies) < 2:
            return
        min_size = int(getattr(cfg, "colony_min_size", 2))
        colonies: list[tuple[str, float, dict[str, object]]] = []
        for cid, meta in self.colonies.items():
            members = meta.get("members", [])
            if not isinstance(members, list) or len(members) < min_size:
                continue
            fitness = float(meta.get("fitness", float("-inf")))
            colonies.append((cid, fitness, meta))
        if len(colonies) < 2:
            return
        colonies.sort(key=lambda item: item[1])
        worst_cid, worst_fit, worst_meta = colonies[0]
        best_cid, best_fit, best_meta = colonies[-1]
        margin = float(getattr(cfg, "colony_selection_margin", 0.05))
        if best_fit - worst_fit < margin:
            return
        freed_members, freed_pot = self._dissolve_colony(
            worst_cid, reason="selection", reassign_pot=True
        )
        if not freed_members and freed_pot <= 0.0:
            return
        pool = self._colony_selection_pool
        pool_members: list[str] = pool.setdefault("members", [])
        pool_events: list[dict[str, object]] = pool.setdefault("events", [])
        existing = set(pool_members)
        for member in freed_members:
            if member not in existing:
                pool_members.append(member)
                existing.add(member)
        reward_frac = float(getattr(cfg, "colony_selection_reward_frac", 0.25))
        reward_frac = max(0.0, min(1.0, reward_frac))
        reward_bonus = freed_pot * reward_frac
        pool_pot = float(pool.get("pot", 0.0)) + freed_pot
        if reward_bonus > 0.0 and best_meta is not None:
            pool_pot -= reward_bonus
            best_meta["pot"] = float(best_meta.get("pot", 0.0)) + reward_bonus
            self._log_colony_event(
                best_meta,
                self.generation_index,
                "selection_bonus",
                amount=float(reward_bonus),
                source=worst_cid,
            )
        pool["pot"] = max(0.0, pool_pot)
        event_payload = {
            "gen": self.generation_index,
            "type": "pool_add",
            "colony": worst_cid,
            "count": len(freed_members),
            "pot": float(freed_pot),
        }
        pool_events.append(event_payload)
        if len(pool_events) > 200:
            del pool_events[:-200]
        self._colony_selection_stats["dissolved"] = (
            int(self._colony_selection_stats.get("dissolved", 0)) + 1
        )
        min_pool = float(getattr(cfg, "colony_selection_min_pool", 0.0))
        created = 0
        min_pool = max(0.0, min_pool)
        while len(pool_members) >= min_size:
            if pool["pot"] < min_pool:
                break
            reserve_floor = (
                float(best_meta.get("reserve_floor", self.config.energy.m))
                if best_meta
                else self.config.energy.m
            )
            inherit_target = max(min_pool, reserve_floor * 0.5)
            inherit_amount = min(float(pool["pot"]), inherit_target)
            if inherit_amount <= 0.0:
                break
            pool["pot"] = max(0.0, float(pool["pot"]) - inherit_amount)
            child_meta = self._replicate_colony(
                best_cid, pool_members, inherited_pot=inherit_amount
            )
            if child_meta is None:
                pool["pot"] = float(pool.get("pot", 0.0)) + inherit_amount
                break
            created += 1
            pool_events.append(
                {
                    "gen": self.generation_index,
                    "type": "replicate",
                    "from": best_cid,
                    "child": child_meta.get("id"),
                    "inherit_pot": float(inherit_amount),
                }
            )
            if pool["pot"] < min_pool:
                break
        if created:
            self._colony_selection_stats["replicated"] = (
                int(self._colony_selection_stats.get("replicated", 0)) + created
            )
        self._colony_selection_stats["pool_members"] = len(pool_members)
        self._colony_selection_stats["pool_pot"] = round(float(pool.get("pot", 0.0)), 4)
        events = self._colony_selection_stats.setdefault("events", [])
        events.append(
            {
                "gen": self.generation_index,
                "type": "selection",
                "dissolved": worst_cid,
                "best": best_cid,
                "created": created,
                "pool_pot": float(pool.get("pot", 0.0)),
            }
        )
        if len(events) > 200:
            del events[:-200]

    def _colony_tier_migration(self) -> None:
        cfg = self.config.assimilation_tuning
        tier_count = int(getattr(cfg, "colony_tier_count", 1))
        if tier_count <= 1 or not self.colonies:
            return
        promote_passes = int(getattr(cfg, "colony_tier_promote_passes", 3))
        promote_delta = float(getattr(cfg, "colony_tier_promote_delta", 0.1))
        demote_failures = int(getattr(cfg, "colony_tier_demote_failures", 3))
        demote_delta = float(getattr(cfg, "colony_tier_demote_delta", -0.05))
        hazard_floor = float(getattr(cfg, "colony_tier_hazard_floor", -2.0))
        cooldown_default = int(getattr(cfg, "colony_tier_cooldown", 3))
        for _cid, meta in self.colonies.items():
            tier = int(meta.get("tier", 0))
            cooldown = int(meta.get("tier_cooldown", 0))
            if cooldown > 0:
                meta["tier_cooldown"] = cooldown - 1
                continue
            passes = int(meta.get("holdout_passes", 0))
            failures = int(meta.get("holdout_failures", 0))
            delta = float(meta.get("last_delta", 0.0))
            hazard = float(meta.get("hazard_z", 0.0))
            new_tier = tier
            reason = None
            if tier < tier_count - 1 and passes >= promote_passes and delta >= promote_delta:
                new_tier = tier + 1
                reason = "tier_promote"
            elif tier > 0 and (
                failures >= demote_failures or delta <= demote_delta or hazard <= hazard_floor
            ):
                new_tier = tier - 1
                reason = "tier_demote"
            if new_tier != tier:
                meta["tier"] = new_tier
                meta["tier_cooldown"] = max(0, cooldown_default)
                if reason == "tier_promote":
                    meta["holdout_passes"] = 0
                else:
                    meta["holdout_failures"] = 0
                self._log_colony_event(
                    meta,
                    self.generation_index,
                    reason if reason else "tier_change",
                    tier=int(tier),
                    new_tier=int(new_tier),
                    delta=float(delta),
                    hazard=float(hazard),
                )

    def _tick_colonies(self) -> None:
        if not self.colonies:
            return
        cfg = self.config.assimilation_tuning
        max_failures = int(getattr(cfg, "colony_max_failures", 2))
        for cid, meta in list(self.colonies.items()):
            meta.setdefault("id", cid)
            meta.setdefault("tax_rate", float(getattr(cfg, "colony_tax_rate", 0.1)))
            meta.setdefault("subsidy_frac", float(getattr(cfg, "colony_subsidy_fraction", 0.25)))
            meta.setdefault("cohesion_weight", 0.5)
            meta.setdefault("comms_bonus", 0.0)
            members: list[str] = [str(x) for x in meta.get("members", [])]
            meta["members"] = members
            self._colony_refresh_roles(meta)
            pot = float(meta.get("pot", 0.0))
            earn = float(meta.pop("_pot_earn_gen", 0.0))
            if earn:
                self._log_colony_event(
                    meta, self.generation_index, "pot_update", pot=float(pot), earn=float(earn)
                )
            reserve_ticket_mult = float(getattr(cfg, "colony_reserve_ticket_multiplier", 3.0))
            reserve_ratio_cfg = float(getattr(cfg, "colony_reserve_ratio", 0.25))
            reserve_cost_window = int(getattr(cfg, "colony_reserve_cost_window", 6))
            expected_cost = self._colony_expected_cost(members, reserve_cost_window)
            reserve_floor = max(
                self.config.energy.m * reserve_ticket_mult, reserve_ratio_cfg * expected_cost
            )
            meta["reserve_floor"] = float(reserve_floor)
            reserve_prev = bool(meta.get("reserve_active"))
            reserve_active = pot < reserve_floor
            meta["reserve_active"] = reserve_active
            if reserve_active:
                meta["freeze_reproduction"] = True
                if not reserve_prev:
                    self._log_colony_event(
                        meta,
                        self.generation_index,
                        "reserve_enter",
                        pot=float(pot),
                        floor=float(reserve_floor),
                    )
            else:
                meta.pop("freeze_reproduction", None)
                if reserve_prev:
                    self._log_colony_event(
                        meta,
                        self.generation_index,
                        "reserve_exit",
                        pot=float(pot),
                        floor=float(reserve_floor),
                    )
            hazard_window = int(getattr(cfg, "colony_winter_window", 6))
            roi_history, roi_latest = self._colony_roi_series(members, hazard_window)
            hazard_z = 0.0
            if len(roi_history) >= max(3, hazard_window):
                try:
                    hist_mean = sum(roi_history) / len(roi_history)
                    hist_std = pstdev(roi_history) if len(roi_history) > 1 else 0.0
                    latest_mean = sum(roi_latest) / len(roi_latest) if roi_latest else hist_mean
                    if hist_std > 1e-6:
                        hazard_z = (latest_mean - hist_mean) / hist_std
                except Exception:
                    hazard_z = 0.0
            meta["hazard_z"] = float(hazard_z)
            subsidy_threshold = float(getattr(cfg, "colony_subsidy_threshold", 1.0))
            subsidy_frac = float(
                meta.get("subsidy_frac", getattr(cfg, "colony_subsidy_fraction", 0.25))
            )
            subsidies: list[tuple[str, float]] = []
            if subsidy_frac > 0.0 and subsidy_threshold > 0.0:
                threshold_balance = subsidy_threshold * self.config.energy.m
                for member in members:
                    try:
                        bal = float(self.host.ledger.energy_balance(member))
                    except Exception:
                        bal = 0.0
                    if bal >= threshold_balance or pot <= 0.0:
                        continue
                    deficit = threshold_balance - bal
                    grant = min(pot * subsidy_frac, deficit)
                    if grant <= 0.0:
                        continue
                    try:
                        self.host.ledger.credit_energy(member, grant)
                    except Exception:
                        continue
                    pot -= grant
                    subsidies.append((member, grant))
            for member, amount in subsidies:
                self._log_colony_event(
                    meta, self.generation_index, "subsidy", member=member, amount=float(amount)
                )
            meta["pot"] = pot
            roi_vals = [
                float(self.population.average_roi(m, limit=5))
                for m in members
                if m in self.population.population
            ]
            roi_vals = [v for v in roi_vals if math.isfinite(v)]
            roi_mean = sum(roi_vals) / len(roi_vals) if roi_vals else 0.0
            meta["roi_mean"] = roi_mean
            reserve_floor = float(meta.get("reserve_floor", max(self.config.energy.m, 1.0)))
            pot_ratio = pot / max(reserve_floor, 1e-6) if reserve_floor > 0 else 0.0
            pot_ratio = max(0.0, min(2.0, pot_ratio))
            alpha = float(getattr(cfg, "colony_selection_alpha", 1.0))
            beta_weight = float(getattr(cfg, "colony_selection_beta", 0.2))
            gamma_weight = float(getattr(cfg, "colony_selection_gamma", 0.0))
            bandwidth_val = float(meta.get("bandwidth_budget", meta.get("bandwidth_left", 0.0)))
            fitness = alpha * roi_mean + beta_weight * pot_ratio + gamma_weight * bandwidth_val
            meta["fitness"] = fitness
            meta["fitness_components"] = {
                "roi_mean": roi_mean,
                "pot_ratio": pot_ratio,
                "bandwidth": bandwidth_val,
            }
            winter_prev = bool(meta.get("winter_mode"))
            winter_threshold = -abs(float(getattr(cfg, "colony_winter_z_kappa", 1.0)))
            winter_mode = hazard_z <= winter_threshold
            meta["winter_mode"] = winter_mode
            if winter_mode and not winter_prev:
                self._log_colony_event(
                    meta,
                    self.generation_index,
                    "winter_enter",
                    z=float(hazard_z),
                )
                self._enter_colony_winter(cid, meta, members)
            elif not winter_mode and winter_prev:
                self._log_colony_event(
                    meta,
                    self.generation_index,
                    "winter_exit",
                    z=float(hazard_z),
                )
                self._exit_colony_winter(meta)
            review_interval = int(meta.get("review_interval", 5))
            last_review = int(
                meta.get("last_review", meta.get("created_gen", self.generation_index))
            )
            required_passes = int(meta.get("required_passes", 2))
            margin = float(meta.get("margin", getattr(cfg, "holdout_margin", 0.03)))
            variance_improve = float(getattr(cfg, "colony_variance_improve", 0.2))
            expand_delta = float(getattr(cfg, "colony_expand_delta", margin))
            expand_windows = int(getattr(cfg, "colony_expand_windows", max(2, review_interval)))
            shrink_delta = float(getattr(cfg, "colony_shrink_delta", -0.02))
            if len(members) >= 2 and self.generation_index - last_review >= review_interval:
                tasks = self._sample_holdout_tasks()
                delta = float("-inf")
                variance_ratio = 1.0
                pass_gate = False
                if tasks:
                    try:
                        team_stats = self._team_holdout_stats(members, tasks)
                        solo_stats = [self._team_holdout_stats([mem], tasks) for mem in members]
                        solo_means = [float(stat["mean"]) for stat in solo_stats]
                        solo_vars = [float(stat["variance"]) for stat in solo_stats]
                        delta = (
                            float(team_stats["mean"]) - max(solo_means)
                            if solo_means
                            else float(team_stats["mean"])
                        )
                        min_var = min((v for v in solo_vars if v > 1e-6), default=0.0)
                        team_var = float(team_stats["variance"])
                        if min_var > 0.0:
                            variance_ratio = team_var / min_var
                        else:
                            variance_ratio = 0.0 if team_var <= 1e-6 else 1.0
                        pass_gate = delta >= margin and variance_ratio <= (
                            1.0 - variance_improve + 1e-6
                        )
                    except Exception:
                        delta = float("-inf")
                        variance_ratio = 1.0
                        pass_gate = False
                meta.setdefault("delta_history", []).append(float(delta))
                meta["delta_history"] = meta["delta_history"][-24:]
                meta.setdefault("variance_history", []).append(float(variance_ratio))
                meta["variance_history"] = meta["variance_history"][-24:]
                meta["last_review"] = self.generation_index
                meta["last_delta"] = float(delta)
                meta["last_variance_ratio"] = float(variance_ratio)
                variance_leash = float(getattr(cfg, "colony_variance_leash", 1.5))
                variance_guard = variance_ratio >= variance_leash
                meta["variance_guard"] = bool(variance_guard)
                if variance_guard:
                    if not meta.get("_variance_logged"):
                        self._log_colony_event(
                            meta,
                            self.generation_index,
                            "variance_guard",
                            variance_ratio=float(variance_ratio),
                        )
                        meta["_variance_logged"] = True
                else:
                    meta.pop("_variance_logged", None)
                if pass_gate:
                    meta["holdout_passes"] = int(meta.get("holdout_passes", 0)) + 1
                    meta["holdout_failures"] = max(0, int(meta.get("holdout_failures", 0)) - 1)
                    self._log_colony_event(
                        meta,
                        self.generation_index,
                        "holdout_pass",
                        delta=float(delta),
                        variance_ratio=float(variance_ratio),
                    )
                else:
                    meta["holdout_failures"] = int(meta.get("holdout_failures", 0)) + 1
                    self._log_colony_event(
                        meta,
                        self.generation_index,
                        "holdout_fail",
                        delta=float(delta),
                        variance_ratio=float(variance_ratio),
                    )
                # Shrink instead of dissolve when feasible
                try:  # pragma: no cover - exercised in long runs
                    min_size = int(getattr(cfg, "colony_min_size", 2))
                except Exception:  # pragma: no cover
                    min_size = 2
                shrink_trigger = (
                    float(meta.get("last_delta", 0.0)) <= shrink_delta
                    or variance_ratio > (1.0 - variance_improve + 1e-6)
                    or variance_guard
                )
                if (
                    len(members) > min_size
                    and shrink_trigger
                    and int(meta.get("holdout_failures", 0)) >= 1
                ):
                    try:  # pragma: no cover
                        worst = min(
                            members, key=lambda oid: self.population.average_roi(oid, limit=5)
                        )
                        members.remove(worst)
                        meta["members"] = members
                        self._colony_refresh_roles(meta)
                        self._log_colony_event(meta, self.generation_index, "shrink", removed=worst)
                        meta["holdout_failures"] = max(0, int(meta.get("holdout_failures", 0)) - 1)
                    except Exception:  # pragma: no cover
                        pass
                if int(meta.get("holdout_failures", 0)) >= max_failures:
                    self._log_colony_event(
                        meta, self.generation_index, "dissolve", reason="failures"
                    )
                    self.colonies.pop(cid, None)
                    continue
                expand_ok = delta >= expand_delta and variance_ratio <= (
                    1.0 - variance_improve + 1e-6
                )
                history = meta.setdefault("expand_history", [])
                history.append(bool(expand_ok))
                meta["expand_history"] = history[-max(1, expand_windows) :]
            reserve_ratio = float(meta.get("reserve_ratio", 0.25))
            privileges_unlocked = int(meta.get("holdout_passes", 0)) >= required_passes
            if (
                privileges_unlocked
                and not meta.get("freeze_reproduction")
                and not meta.get("winter_mode")
            ):
                for m in members:
                    bal = self.host.ledger.energy_balance(m)
                    ticket = self.config.energy.m
                    reserve = reserve_ratio * 4.0 * ticket
                    if pot > reserve and bal < max(
                        ticket, self.config.assimilation_tuning.energy_floor or ticket
                    ):
                        amount = min(pot - reserve, max(ticket - bal, 0.0))
                        if amount > 0:
                            self.host.ledger.credit_energy(m, amount)
                            pot -= amount
                            self._log_colony_event(
                                meta, self.generation_index, "topup", member=m, amount=float(amount)
                            )
            meta["pot"] = pot
            # Attempt cautious expansion to a 3rd member if synergy persists
            try:
                max_size = int(getattr(cfg, "colony_max_size", 3))
                can_expand = (
                    len(members) < max_size
                    and int(meta.get("holdout_passes", 0)) >= required_passes
                    and len(meta.get("expand_history", [])) >= expand_windows
                    and all(meta.get("expand_history", [])[-expand_windows:])
                    and not meta.get("freeze_reproduction")
                    and not meta.get("winter_mode")
                )
                if can_expand:
                    # Pick a candidate outside the colony with highest recent ROI
                    candidates = [
                        oid for oid in self.population.population.keys() if oid not in members
                    ]
                    if candidates:
                        best_cand = max(
                            candidates, key=lambda oid: self.population.average_roi(oid, limit=5)
                        )
                        tasks = self._sample_holdout_tasks()
                        if tasks:
                            current_stats = self._team_holdout_stats(members, tasks)
                            expanded_stats = self._team_holdout_stats(members + [best_cand], tasks)
                            delta_expand = float(expanded_stats["mean"]) - float(
                                current_stats["mean"]
                            )
                            current_var = float(current_stats["variance"])
                            expanded_var = float(expanded_stats["variance"])
                            if current_var > 1e-6:
                                variance_gain = (
                                    expanded_var <= (1.0 - variance_improve + 1e-6) * current_var
                                )
                            else:
                                variance_gain = expanded_var <= 1e-6
                            if delta_expand >= expand_delta and variance_gain:
                                members.append(best_cand)
                                meta["members"] = members
                                self._colony_refresh_roles(meta)
                                meta["last_review"] = self.generation_index
                                meta["last_delta"] = float(delta_expand)
                                meta["last_variance_ratio"] = (
                                    expanded_var / max(current_var, 1e-6)
                                    if current_var > 1e-6
                                    else 0.0
                                )
                                meta["expand_history"] = []
                                self._log_colony_event(
                                    meta,
                                    self.generation_index,
                                    "expand",
                                    added=best_cand,
                                    delta=float(delta_expand),
                                    variance=float(expanded_var),
                                )
            except Exception:
                pass

    def _auto_tune_assimilation_energy(self, summary: dict[str, object]) -> None:
        tuning = self.config.assimilation_tuning
        window = 12
        costs: list[float] = []
        rois: list[float] = []
        deltas: list[float] = []
        for organelle_id in self.population.population.keys():
            costs.extend(
                value
                for value in self.population.energy.get(organelle_id, [])[-window:]
                if math.isfinite(value)
            )
            rois.extend(
                value
                for value in self.population.roi.get(organelle_id, [])[-window:]
                if math.isfinite(value)
            )
            deltas.extend(
                value
                for value in self.population.recent_energy_deltas(organelle_id, limit=window)
                if math.isfinite(value)
            )
        if not costs or not rois:
            return
        avg_cost = mean(costs)
        n_episodes = max(1, self.config.environment.synthetic_batch_size)
        ticket = max(self.config.energy.m, 1e-6)
        roi_required = 1.0 + ticket / max(n_episodes * avg_cost, 1e-6)
        roi_required = max(1.0, roi_required)
        roi_cap = max(roi_required * 3.0, roi_required + 1.5, 5.0)
        roi_samples = [max(0.0, value) for value in rois if value >= 0.0]
        if not roi_samples:
            percentile = roi_required
        elif len(roi_samples) >= 4:
            try:
                percentile = quantiles(roi_samples, n=4, method="inclusive")[2]
            except Exception:
                percentile = max(roi_samples)
        else:
            percentile = max(roi_samples)
        percentile = max(roi_required, min(percentile, roi_cap))
        std_delta = pstdev(deltas) if len(deltas) >= 2 else 0.0
        target_floor = max(ticket, ticket + std_delta)
        target_floor = min(self.host.ledger.energy_cap, target_floor)

        if not hasattr(self, "_energy_guard_bandit"):
            self._energy_guard_bandit = EnergyGuardBandit(weights=[-1.0, -0.5, 0.0, 0.5])
            if tuning.energy_floor <= 0.0:
                tuning.energy_floor = max(ticket, tuning.energy_floor_base)
            if tuning.energy_floor_roi <= 0.0:
                tuning.energy_floor_roi = max(1.0, tuning.energy_floor_roi_base)

        bandit: EnergyGuardBandit = self._energy_guard_bandit
        bandit.record_reward(self._energy_guard_reward(summary))

        base_floor = max(ticket, tuning.energy_floor_base)
        base_threshold = max(1.0, tuning.energy_floor_roi_base)
        stat_floor = target_floor
        stat_threshold = percentile

        for arm in bandit.arms:
            weight = arm.weight
            floor_candidate = base_floor + weight * (stat_floor - base_floor)
            threshold_candidate = base_threshold + weight * (stat_threshold - base_threshold)
            floor_candidate = max(ticket, min(self.host.ledger.energy_cap, floor_candidate))
            threshold_candidate = max(1.0, min(roi_cap, threshold_candidate))
            arm.floor = floor_candidate
            arm.roi = threshold_candidate

        choice = bandit.select()
        selected = bandit.arms[choice]
        smoothing = 0.12
        current_floor = tuning.energy_floor if tuning.energy_floor > 0.0 else base_floor
        current_threshold = (
            tuning.energy_floor_roi if tuning.energy_floor_roi > 0.0 else base_threshold
        )
        new_floor = (1.0 - smoothing) * current_floor + smoothing * selected.floor
        new_threshold = (1.0 - smoothing) * current_threshold + smoothing * selected.roi
        new_floor = max(ticket, min(self.host.ledger.energy_cap, new_floor))
        new_threshold = max(1.0, min(roi_cap, new_threshold, base_threshold + 0.4))
        decay_applied = False
        stall_generations = getattr(self, "no_merge_counter", 0)
        stall_window = max(4, self.config.assimilation_tuning.per_cell_interval * 2)
        if stall_generations >= stall_window:
            factor = 0.7 ** max(1, stall_generations // stall_window)
            new_threshold = max(1.0, new_threshold * factor)
            new_floor = max(ticket, new_floor * 0.85)
            decay_applied = True
        tuning.energy_floor = new_floor
        tuning.energy_floor_roi = new_threshold
        summary["assimilation_energy_tuning"] = {
            "avg_cost": avg_cost,
            "roi_required": roi_required,
            "roi_percentile": stat_threshold,
            "roi_cap": roi_cap,
            "energy_floor": new_floor,
            "energy_floor_roi": new_threshold,
            "delta_std": std_delta,
            "bandit_choice": choice,
            "bandit_weight": selected.weight,
            "bandit_reward": getattr(bandit, "last_reward", 0.0),
            "base_floor": base_floor,
            "base_roi": base_threshold,
            "stall_generations": stall_generations,
            "decay_applied": decay_applied,
        }

    def _maybe_top_up_energy(
        self, genome: Genome, balance: float
    ) -> tuple[float, dict[str, float | str]]:
        tuning = self.config.assimilation_tuning
        floor = getattr(tuning, "energy_floor", 0.0)
        roi_threshold = getattr(tuning, "energy_floor_roi", 0.0)
        roi_bonus = getattr(tuning, "energy_topup_roi_bonus", 0.0)
        relief = float(self._roi_relief.get(genome.organelle_id, 0.0))
        # dynamic easing: variance and fail streak make top-ups slightly easier to build evidence
        try:
            recent = self.population.roi.get(genome.organelle_id, [])[-8:]
            roi_std = pstdev([r for r in recent if math.isfinite(r)]) if len(recent) >= 2 else 0.0
        except Exception:
            roi_std = 0.0
        streak = max(0, int(getattr(self, "assim_fail_streak", 0)))
        dynamic_bonus = min(1.5, float(roi_bonus) + 0.15 * float(roi_std) + 0.02 * float(streak))
        effective_threshold = max(0.0, float(roi_threshold) - dynamic_bonus - relief)
        ledger = self.host.ledger
        tokens_available = self.population.evidence_tokens(genome.organelle_id)
        info: dict[str, float | str] = {
            "status": "disabled" if floor <= 0.0 else "pending",
            "before": float(balance),
            "after": float(balance),
            "floor": float(floor),
            "roi_threshold": float(roi_threshold),
            "roi_threshold_effective": float(effective_threshold),
            "credited": 0.0,
            "roi_std": float(roi_std),
            "fail_streak": float(streak),
            "relief": relief,
            "tokens_available": int(tokens_available),
        }
        survival_cfg = getattr(self.config, "survival", None)
        if survival_cfg is not None and getattr(survival_cfg, "enabled", False):
            reserve_state = self._reserve_state.get(genome.organelle_id, {})
            hazard_state = self._hazard_state.get(genome.organelle_id, {})
            bonus = 0.0
            if reserve_state.get("active"):
                bonus += float(getattr(survival_cfg, "hazard_topup_bonus", 0.0)) * 0.5
            if hazard_state.get("active"):
                bonus += float(getattr(survival_cfg, "hazard_topup_bonus", 0.0))
            if bonus > 0.0:
                effective_threshold = max(0.0, effective_threshold - bonus)
                info["survival_bonus"] = bonus
                info["roi_threshold_effective"] = float(effective_threshold)
            info["reserve_active"] = bool(reserve_state.get("active"))
            info["hazard_active"] = bool(hazard_state.get("active"))
        if floor <= 0.0:
            return balance, info
        if balance >= floor:
            info["status"] = "already_sufficient"
            info["after"] = float(balance)
            return balance, info
        roi = self.population.average_roi(genome.organelle_id, limit=5)
        info["roi"] = float(roi)
        if roi < effective_threshold:
            # attempt evidence-bypass top-up (only if tokens present)
            if tokens_available > 0 and self.population.consume_evidence(genome.organelle_id, 1):
                tokens_available -= 1
                info["tokens_available"] = int(tokens_available)
                available = max(0.0, min(floor - balance, ledger.energy_cap - balance))
                if available > 0.0:
                    ledger.credit_energy(genome.organelle_id, available)
                    new_balance = ledger.energy_balance(genome.organelle_id)
                    info["status"] = "credited"
                    info["credited"] = float(new_balance - balance)
                    info["after"] = float(new_balance)
                    info["floor"] = float(floor)
                    info["roi_threshold"] = float(roi_threshold)
                    info["evidence_bypass"] = True
                    self._register_roi_success(genome.organelle_id)
                    if self._power_econ_stats is not None:
                        self._power_econ_stats["tokens_used"] = (
                            int(self._power_econ_stats.get("tokens_used", 0)) + 1
                        )
                    return new_balance, info
            info["status"] = "skip_low_roi"
            # info-aware top-up when close to required window
            info_gap = int(getattr(tuning, "info_topup_gap", 0))
            info_slack = float(getattr(tuning, "info_topup_roi_slack", 0.0))
            if info_gap > 0:
                scores_available = len(self.population.recent_scores(genome.organelle_id, limit=16))
                base_min = self._min_window_requirement(tuning)
                info_need = max(0, base_min - scores_available)
                if 0 < info_need <= info_gap and roi >= (effective_threshold - info_slack):
                    available = max(0.0, min(floor - balance, ledger.energy_cap - balance))
                    if available > 0.0:
                        ledger.credit_energy(genome.organelle_id, available)
                        new_balance = ledger.energy_balance(genome.organelle_id)
                        info["status"] = "credited"
                        info["credited"] = float(new_balance - balance)
                        info["after"] = float(new_balance)
                        info["floor"] = float(floor)
                        info["roi_threshold"] = float(roi_threshold)
                        info["info_topup"] = int(info_need)
                        self._register_roi_success(genome.organelle_id)
                        if self._power_econ_stats is not None:
                            self._power_econ_stats["info_topups"] = (
                                int(self._power_econ_stats.get("info_topups", 0)) + 1
                            )
                        return new_balance, info
            self._register_roi_skip(genome.organelle_id)
            return balance, info
        available = max(0.0, min(floor - balance, ledger.energy_cap - balance))
        if available <= 0.0:
            info["status"] = "skip_no_capacity"
            return balance, info
        ledger.credit_energy(genome.organelle_id, available)
        new_balance = ledger.energy_balance(genome.organelle_id)
        info["status"] = "credited"
        info["credited"] = float(new_balance - balance)
        info["after"] = float(new_balance)
        info["floor"] = float(floor)
        info["roi_threshold"] = float(roi_threshold)
        self._register_roi_success(genome.organelle_id)
        return new_balance, info

    def _load_holdout_tasks(self) -> list[EvaluationTask]:
        if self._holdout_tasks_cache is not None:
            return self._holdout_tasks_cache
        cfg = self.config.assimilation_tuning
        path = cfg.holdout_tasks_path or self.config.evaluation.tasks_path
        if path is None:
            self._holdout_tasks_cache = []
            return self._holdout_tasks_cache
        try:
            runtime = EvaluationManager.from_file(
                enabled=True,
                cadence=1,
                tasks_path=path,
                sample_size=None,
                reward_weight=self.config.evaluation.reward_weight,
            )
            self._holdout_tasks_cache = list(runtime.tasks)
        except Exception:
            self._holdout_tasks_cache = []
        return self._holdout_tasks_cache

    def _sample_holdout_tasks(self, override_size: int | None = None) -> list[EvaluationTask]:
        tasks = self._load_holdout_tasks()
        if not tasks:
            return []
        if override_size is not None:
            target = int(max(1, override_size))
        else:
            target = int(max(1, getattr(self.config.assimilation_tuning, "holdout_sample_size", 2)))
        sample_size = max(1, min(len(tasks), target))
        if len(tasks) <= sample_size:
            return list(tasks)
        return self.holdout_rng.sample(tasks, sample_size)

    def _evaluate_holdout_roi(self, organelle_id: str, tasks: list[EvaluationTask]) -> float:
        if not tasks:
            return 0.0
        energy_cfg = self.config.energy
        rois: list[float] = []
        for index, task in enumerate(tasks, start=1):
            grid_task = task.to_grid_task(self.environment, task_id=f"holdout_{index:04d}")
            result = self.host.step(
                prompt=grid_task.prompt,
                intent="assimilation holdout",
                max_routes=1,
                allowed_organelle_ids=[organelle_id],
            )
            metrics = result.responses.get(organelle_id)
            if metrics is None:
                continue
            success, reward = grid_task.evaluate(metrics.answer)
            revenue = grid_task.price * reward.total
            cost = (
                energy_cfg.alpha * metrics.flops_estimate
                + energy_cfg.beta * metrics.memory_gb
                + energy_cfg.gamma * metrics.latency_ms
                + energy_cfg.lambda_p * metrics.trainable_params
            )
            if cost <= 0.0:
                roi_value = float("inf") if revenue > 0 else 0.0
            else:
                roi_value = revenue / cost
            if not math.isfinite(roi_value):
                roi_value = max(0.0, min(roi_value, 10.0)) if revenue > 0 else 0.0
            if not math.isfinite(roi_value):
                continue
            rois.append(roi_value)
        if not rois:
            return float("-inf")
        return sum(rois) / len(rois)

    def _team_holdout_stats(
        self, member_ids: list[str], tasks: list[EvaluationTask]
    ) -> dict[str, object]:
        if not tasks or not member_ids:
            return {"mean": 0.0, "variance": 0.0, "series": []}
        energy_cfg = self.config.energy
        rois: list[float] = []
        for index, task in enumerate(tasks, start=1):
            grid_task = task.to_grid_task(self.environment, task_id=f"team_{index:04d}")
            best_roi = 0.0
            for oid in member_ids:
                result = self.host.step(
                    prompt=grid_task.prompt,
                    intent="team holdout",
                    max_routes=1,
                    allowed_organelle_ids=[oid],
                )
                metrics = result.responses.get(oid)
                if metrics is None:
                    continue
                success, reward = grid_task.evaluate(metrics.answer)
                revenue = grid_task.price * reward.total
                cost = (
                    energy_cfg.alpha * metrics.flops_estimate
                    + energy_cfg.beta * metrics.memory_gb
                    + energy_cfg.gamma * metrics.latency_ms
                    + energy_cfg.lambda_p * metrics.trainable_params
                )
                if cost <= 0.0:
                    roi_value = float("inf") if revenue > 0 else 0.0
                else:
                    roi_value = revenue / cost
                if not math.isfinite(roi_value):
                    roi_value = max(0.0, min(roi_value, 10.0)) if revenue > 0 else 0.0
                best_roi = max(best_roi, float(roi_value))
            rois.append(best_roi)
        if not rois:
            return {"mean": 0.0, "variance": 0.0, "series": []}
        mean_val = float(sum(rois) / len(rois))
        variance = float(pstdev(rois)) if len(rois) >= 2 else 0.0
        return {"mean": mean_val, "variance": variance, "series": rois}

    def _evaluate_team_holdout_roi(
        self, a_id: str, b_id: str, tasks: list[EvaluationTask]
    ) -> float:
        stats = self._team_holdout_stats([a_id, b_id], tasks)
        return float(stats.get("mean", 0.0))

    def _evaluate_multi_team_holdout_roi(
        self, member_ids: list[str], tasks: list[EvaluationTask]
    ) -> float:
        stats = self._team_holdout_stats(member_ids, tasks)
        return float(stats.get("mean", 0.0))

    def _min_samples_required(self, tuning: object | None = None) -> int:
        cfg = tuning or self.config.assimilation_tuning
        try:
            value = int(getattr(cfg, "min_uplift_samples", 2))
        except Exception:
            value = 2
        return max(1, value)

    def _normalize_min_window_candidate(self, candidate: int, tuning: object | None = None) -> int:
        cfg = tuning or self.config.assimilation_tuning
        min_samples = self._min_samples_required(cfg)
        floor = max(2 * min_samples, 2)
        try:
            floor_override = int(getattr(cfg, "min_window_min", floor))
            if floor_override > 0:
                floor = max(floor, floor_override)
        except Exception:
            pass
        try:
            ceiling = int(getattr(cfg, "min_window_max", max(floor, candidate)))
        except Exception:
            ceiling = max(floor, candidate, 2 * min_samples)
        if ceiling < floor:
            ceiling = floor
        normalized = max(floor, min(candidate, ceiling))
        if normalized % 2 != 0:
            if normalized + 1 <= ceiling:
                normalized += 1
            else:
                normalized -= 1
                if normalized < floor:
                    normalized = floor
        return normalized

    def _min_window_requirement(self, tuning: object | None = None) -> int:
        cfg = tuning or self.config.assimilation_tuning
        min_samples = self._min_samples_required(cfg)
        base = max(2 * min_samples, 2)
        try:
            configured = int(getattr(cfg, "min_window", base))
        except Exception:
            configured = base
        value = max(base, configured)
        if value % 2 != 0:
            value += 1
        return value

    def _apply_evidence_tokens(
        self,
        organelle_id: str,
        available_even: int,
        min_window: int,
        min_samples_required: int,
    ) -> tuple[int, bool]:
        tuning = self.config.assimilation_tuning
        token_window = int(getattr(tuning, "evidence_token_window", 0))
        if token_window <= 0 or available_even <= 0:
            return min_window, False
        tokens_used = False
        tokens_available = self.population.evidence_tokens(organelle_id)
        target_min = min_window
        base_floor = max(2, 2 * max(1, min_samples_required))
        try:
            floor_override = int(getattr(tuning, "min_window_min", base_floor))
        except Exception:
            floor_override = base_floor
        if floor_override <= 0:
            min_floor = base_floor
        else:
            min_floor = max(2, floor_override)
        while tokens_available > 0 and available_even < target_min:
            reduced = max(min_floor, target_min - token_window)
            if not self.population.consume_evidence(organelle_id, 1):
                break
            tokens_available -= 1
            target_min = reduced
            tokens_used = True
            if self._power_econ_stats is not None:
                stats = self._power_econ_stats
                stats["tokens_used"] = int(stats.get("tokens_used", 0)) + 1
            if target_min <= available_even:
                break
        return target_min, tokens_used

    def _set_min_window(self, tuning: object, candidate: int) -> int:
        normalized = self._normalize_min_window_candidate(candidate, tuning)
        if getattr(tuning, "min_window", None) != normalized:
            tuning.min_window = normalized
        return normalized

    @staticmethod
    def _compute_mean_ci(
        series: list[float], alpha: float = 0.05
    ) -> tuple[float, float, float, float]:
        return compute_mean_ci(series=series, alpha=alpha)

    @staticmethod
    def _power_proxy(
        mu: float, baseline: float, margin: float, se: float, alpha: float = 0.05
    ) -> float:
        return power_proxy(mu=mu, baseline=baseline, margin=margin, se=se, alpha=alpha)

    @staticmethod
    def _team_accept(ci_low: float, baseline: float, margin: float, n: int, min_tasks: int) -> bool:
        return team_accept(
            ci_low=ci_low, baseline=baseline, margin=margin, n=n, min_tasks=min_tasks
        )

    def _maybe_team_probes(self) -> int:
        """Probe a few high-ROI pairs per generation and promote colonies when CI gate passes.

        Returns number of promotions.
        """
        per_gen = int(getattr(self.config.assimilation_tuning, "team_probe_per_gen", 0))
        if per_gen <= 0 or len(self.population.population) < 2:
            return 0
        gate_counts: dict[str, int] = {}
        self._team_gate_counts_gen = gate_counts
        sample_override = int(
            getattr(self.config.assimilation_tuning, "team_holdout_sample_size", 0) or 0
        )
        handoff_enabled = bool(
            getattr(self.config.assimilation_tuning, "team_handoff_enabled", False)
        )
        handoff_prompt = str(
            getattr(
                self.config.assimilation_tuning,
                "team_handoff_prompt",
                "Partner answer:\n{answer}\nProvide a critique or improved answer.",
            )
        )
        handoff_cap = int(
            getattr(self.config.assimilation_tuning, "team_handoff_cap_per_gen", 0) or 0
        )

        def _record_team_gate(reason: str, data: dict[str, object]) -> None:
            payload = self._sanitize_telemetry(data)
            payload["reason"] = reason
            payload["generation"] = self.generation_index
            self._team_gate_samples.append(payload)
            if len(self._team_gate_samples) > 120:
                self._team_gate_samples = self._team_gate_samples[-120:]
            gate_counts[reason] = gate_counts.get(reason, 0) + 1

        # Pick top-K candidates by recent ROI
        scored = [
            (oid, float(self.population.average_roi(oid, limit=5)))
            for oid in self.population.population.keys()
        ]
        if not scored:
            return 0
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [oid for oid, _ in scored[: min(6, len(scored))]]
        # Build candidate pairs; prefer co-routing synergy if available
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        # 1) From co-routing counts
        if hasattr(self, "_co_routing_counts") and self._co_routing_counts:
            pairs_sorted = sorted(
                self._co_routing_counts.items(), key=lambda kv: kv[1], reverse=True
            )
            for (a, b), _c in pairs_sorted:
                if a in top and b in top and (a, b) not in seen:
                    pairs.append((a, b))
                    seen.add((a, b))
        # 2) Fill with top-ROI unique pairs
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                key = (top[i], top[j])
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)
        promotions = 0
        rng = self.environment.rng
        rng.shuffle(pairs)
        for a_id, b_id in pairs[:per_gen]:
            override = sample_override if sample_override > 0 else None
            sampler = self._sample_holdout_tasks
            if override is None:
                tasks = sampler()
            else:
                try:
                    tasks = sampler(override_size=override)
                except TypeError:
                    tasks = sampler()
            if not tasks:
                continue
            # Compute best-of-two ROI series and baseline means
            energy_cfg = self.config.energy
            team_series: list[float] = []
            a_series: list[float] = []
            b_series: list[float] = []
            identical_answers = 0
            answer_samples: list[tuple[str, str]] = []
            for index, task in enumerate(tasks, start=1):
                grid_task = task.to_grid_task(self.environment, task_id=f"team_probe_{index:04d}")
                best_roi = 0.0
                partner_answer = ""
                # First member
                result_a = self.host.step(
                    prompt=grid_task.prompt,
                    intent="team probe",
                    max_routes=1,
                    allowed_organelle_ids=[a_id],
                )
                metrics_a = result_a.responses.get(a_id)
                if metrics_a is not None:
                    success_a, reward_a = grid_task.evaluate(metrics_a.answer)
                    revenue_a = grid_task.price * reward_a.total
                    cost_a = (
                        energy_cfg.alpha * metrics_a.flops_estimate
                        + energy_cfg.beta * metrics_a.memory_gb
                        + energy_cfg.gamma * metrics_a.latency_ms
                        + energy_cfg.lambda_p * metrics_a.trainable_params
                    )
                    roi_a = (
                        (float("inf") if revenue_a > 0 else 0.0)
                        if cost_a <= 0.0
                        else (revenue_a / cost_a)
                    )
                    roi_a = 0.0 if not math.isfinite(roi_a) else float(max(0.0, min(roi_a, 10.0)))
                    a_series.append(roi_a)
                    best_roi = max(best_roi, roi_a)
                    partner_answer = str(metrics_a.answer)
                # Second member with optional handoff
                prompt_b = grid_task.prompt
                if handoff_enabled and partner_answer:
                    use_handoff = True
                    if handoff_cap > 0 and getattr(self, "_team_handoff_used", 0) >= handoff_cap:
                        use_handoff = False
                    if use_handoff:
                        prompt_b = f"{grid_task.prompt}\n\n{handoff_prompt.format(answer=partner_answer)}".strip()
                        self._team_handoff_used = int(getattr(self, "_team_handoff_used", 0)) + 1
                result_b = self.host.step(
                    prompt=prompt_b,
                    intent="team probe",
                    max_routes=1,
                    allowed_organelle_ids=[b_id],
                )
                metrics_b = result_b.responses.get(b_id)
                if metrics_b is not None:
                    success_b, reward_b = grid_task.evaluate(metrics_b.answer)
                    revenue_b = grid_task.price * reward_b.total
                    cost_b = (
                        energy_cfg.alpha * metrics_b.flops_estimate
                        + energy_cfg.beta * metrics_b.memory_gb
                        + energy_cfg.gamma * metrics_b.latency_ms
                        + energy_cfg.lambda_p * metrics_b.trainable_params
                    )
                    roi_b = (
                        (float("inf") if revenue_b > 0 else 0.0)
                        if cost_b <= 0.0
                        else (revenue_b / cost_b)
                    )
                    roi_b = 0.0 if not math.isfinite(roi_b) else float(max(0.0, min(roi_b, 10.0)))
                    b_series.append(roi_b)
                    best_roi = max(best_roi, roi_b)
                    answer_b = str(metrics_b.answer)
                    if (
                        partner_answer.strip()
                        and answer_b.strip()
                        and partner_answer.strip() == answer_b.strip()
                    ):
                        identical_answers += 1
                    if len(answer_samples) < 2:
                        answer_samples.append((partner_answer.strip(), answer_b.strip()))
                team_series.append(best_roi)
            attempt_info: dict[str, object] = {
                "pair": [a_id, b_id],
                "tasks": len(team_series),
            }
            if team_series and (identical_answers * 2) >= len(team_series):
                attempt_info["same_answers"] = int(identical_answers)
                if answer_samples:
                    attempt_info["answer_samples"] = [
                        {
                            "first": textwrap.shorten(pair[0], width=120, placeholder="…"),
                            "second": textwrap.shorten(pair[1], width=120, placeholder="…"),
                        }
                        for pair in answer_samples
                    ]
                _record_team_gate("identical_answers", attempt_info)
                continue
            base_a = sum(a_series) / max(len(a_series), 1)
            base_b = sum(b_series) / max(len(b_series), 1)
            baseline = max(base_a, base_b)
            ci_low, ci_high, team_mu, team_se = self._compute_mean_ci(team_series)
            min_tasks = int(getattr(self.config.assimilation_tuning, "team_min_tasks", 8))
            # team_holdout_margin can be present but None; fall back to generic holdout_margin
            margin_val = getattr(self.config.assimilation_tuning, "team_holdout_margin", None)
            if margin_val is None:
                margin_val = getattr(self.config.assimilation_tuning, "holdout_margin", 0.02)
            try:
                margin = float(margin_val)
            except Exception:
                margin = 0.02
            # Power proxy gate
            min_power = float(getattr(self.config.assimilation_tuning, "team_min_power", 0.2))
            power = self._power_proxy(team_mu, baseline, margin, team_se)
            has_tasks = len(team_series) >= min_tasks
            delta_mu = team_mu - baseline
            ci_gate = team_mu >= (baseline + margin)
            attempt_info.update(
                {
                    "team_mu": float(team_mu),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "baseline": float(baseline),
                    "margin": float(margin),
                    "delta_mu": float(delta_mu),
                    "power": float(power),
                    "min_power": float(min_power),
                    "min_tasks": int(min_tasks),
                    "team_se": float(team_se),
                    "same_answers": int(identical_answers),
                }
            )
            if answer_samples:
                samples_fmt = []
                for first, second in answer_samples:
                    samples_fmt.append(
                        {
                            "first": textwrap.shorten(first or "", width=120, placeholder="…"),
                            "second": textwrap.shorten(second or "", width=120, placeholder="…"),
                        }
                    )
                attempt_info["answer_samples"] = samples_fmt
            if power >= min_power and has_tasks and ci_gate:
                cid = f"col_{a_id[:4]}_{b_id[:4]}"
                meta = self.colonies.setdefault(
                    cid,
                    {
                        "members": [a_id, b_id],
                        "pot": 0.0,
                        "reserve_ratio": 0.25,
                        "created_gen": self.generation_index,
                    },
                )
                self._log_colony_event(
                    meta, self.generation_index, "create", members=list(meta.get("members", []))
                )
                promotions += 1
                attempt_info["colony_id"] = cid
                _record_team_gate("accepted", attempt_info)
            else:
                if not has_tasks:
                    reason = "insufficient_tasks"
                elif power < min_power:
                    reason = "low_power"
                elif not ci_gate:
                    reason = "ci_low"
                else:
                    reason = "unknown"
                _record_team_gate(reason, attempt_info)
        if promotions > 0:
            self.promotions_this_gen += promotions
        return promotions

    def _holdout_accepts(
        self, candidate_id: str, mate_ids: list[str]
    ) -> tuple[bool, dict[str, object] | None]:
        cfg = self.config.assimilation_tuning
        retries = 0
        margin = cfg.holdout_margin
        info: dict[str, object] | None = None
        while retries <= cfg.holdout_max_retries:
            tasks = self._sample_holdout_tasks()
            if not tasks:
                return True, info
            candidate_roi = self._evaluate_holdout_roi(candidate_id, tasks)
            baseline_rois = [self._evaluate_holdout_roi(mid, tasks) for mid in mate_ids]
            baseline_rois = [roi for roi in baseline_rois if math.isfinite(roi)]
            best_baseline = max(baseline_rois, default=float("-inf"))
            target = (best_baseline if math.isfinite(best_baseline) else 0.0) + margin
            accepted = candidate_roi >= target
            info = {
                "candidate_roi": candidate_roi,
                "baseline_roi": best_baseline if math.isfinite(best_baseline) else None,
                "margin": margin,
                "tasks": len(tasks),
                "retries": retries,
            }
            if accepted:
                return True, info
            retries += 1
            margin = max(0.0, margin - cfg.holdout_margin_step)
        return False, info

    def _species_partition(self) -> dict[tuple[tuple[str, int], ...], list[str]]:
        mapping: dict[tuple[tuple[str, int], ...], list[str]] = {}
        for organelle_id in self.host.list_organelle_ids():
            organelle = self.host.get_organelle(organelle_id)
            if organelle is None:
                continue
            adapters = self.host._active_adapters(organelle)
            key = tuple(
                sorted(
                    (module, int(count))
                    for module, count in adapters.items()
                    if module not in {"rank", "total"}
                )
            )
            mapping.setdefault(key, []).append(organelle_id)
        return mapping

    @staticmethod
    def _compute_diversity_metrics(
        balances: dict[str, float],
        species_map: dict[tuple[tuple[str, int], ...], list[str]],
    ) -> dict[str, float]:
        values = [max(0.0, val) for val in balances.values()]
        total = sum(values)
        metrics = {
            "energy_gini": 0.0,
            "effective_population": 0.0,
            "species_count": float(len(species_map) if species_map else 0),
            "max_species_share": 0.0,
        }
        if not values or total <= 0.0:
            return metrics
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumulative = sum((i + 1) * val for i, val in enumerate(sorted_vals))
        gini = (2.0 * cumulative) / (n * total) - (n + 1) / n
        shares = [val / total for val in values if val > 0.0]
        effective = 1.0 / sum(share * share for share in shares) if shares else 0.0
        species_shares = [
            sum(balances[oid] for oid in ids) / total for ids in species_map.values() if ids
        ]
        metrics.update(
            {
                "energy_gini": float(max(0.0, min(gini, 1.0))),
                "effective_population": float(effective),
                "max_species_share": float(max(species_shares) if species_shares else 0.0),
            }
        )
        return metrics

    def _enforce_diversity(self) -> None:
        cfg = getattr(self.config, "diversity", None)
        if cfg is None or not cfg.enabled:
            self._diversity_snapshot = None
            return
        organelle_ids = self.host.list_organelle_ids()
        balances = {
            org_id: max(0.0, self.host.ledger.energy_balance(org_id)) for org_id in organelle_ids
        }
        species_map = self._species_partition()
        metrics_before = self._compute_diversity_metrics(balances, species_map)
        total_energy = sum(balances.values())
        enforced = False
        if (
            total_energy > 0
            and cfg.energy_gini_cap < 1.0
            and metrics_before["energy_gini"] > cfg.energy_gini_cap
        ):
            mean_energy = total_energy / max(len(organelle_ids), 1)
            blended = {
                org_id: 0.5 * balances[org_id] + 0.5 * mean_energy for org_id in organelle_ids
            }
            scale = total_energy / (sum(blended.values()) or 1.0)
            for org_id, value in blended.items():
                new_energy = value * scale
                self.host.ledger.set_energy(org_id, new_energy)
                balances[org_id] = new_energy
            total_energy = sum(balances.values())
            enforced = True
        if total_energy > 0 and cfg.max_species_energy_share < 1.0:
            cap = max(0.0, min(cfg.max_species_energy_share, 1.0))
            species_map = self._species_partition()
            for ids in species_map.values():
                species_energy = sum(balances.get(org_id, 0.0) for org_id in ids)
                if species_energy <= 0.0:
                    continue
                share = species_energy / total_energy
                if share > cap:
                    enforced = True
                    scale = (cap * total_energy) / species_energy
                    for org_id in ids:
                        new_energy = balances.get(org_id, 0.0) * scale
                        self.host.ledger.set_energy(org_id, new_energy)
                        balances[org_id] = new_energy
            total_energy = sum(balances.values())
        species_map = self._species_partition()
        metrics_after = self._compute_diversity_metrics(balances, species_map)
        metrics_after["enforced"] = float(enforced)
        self._diversity_snapshot = metrics_after

    def _energy_guard_reward(self, summary: dict[str, object]) -> float:
        merges = float(summary.get("merges", 0) or 0)
        avg_roi = float(summary.get("avg_roi", 0.0) or 0.0)
        gating = summary.get("assimilation_gating") or {}
        low_energy = float(gating.get("low_energy", 0) or 0)
        cooldown = float(gating.get("cooldown", 0) or 0)
        insufficient = float(gating.get("insufficient_scores", 0) or 0)
        reward = merges * 2.0 + max(0.0, avg_roi)
        reward -= 0.002 * low_energy
        reward -= 0.01 * cooldown
        reward -= 0.005 * insufficient
        return reward

    def _maybe_decay_assimilation_thresholds(self) -> None:
        cfg = self.config.assimilation_tuning
        if self.assim_fail_streak <= 0:
            return
        if self.assim_fail_streak % 10 != 0:
            return
        decay = max(0.5, min(cfg.adaptive_decay, 1.0))
        new_threshold = max(
            cfg.adaptive_floor, self.config.evolution.assimilation_threshold * decay
        )
        self.config.evolution.assimilation_threshold = new_threshold
        self.assimilation.update_thresholds(uplift_threshold=new_threshold)
        # relax p-value marginally when decay triggers
        self.config.evolution.assimilation_p_value = min(
            0.5, self.config.evolution.assimilation_p_value * (1.0 + (1.0 - decay))
        )

    def _auto_nudge_evidence(self, summary: dict[str, object]) -> None:
        """Adapt assimilation evidence knobs in‑run when progress stalls.

        Nudges are incremental and bounded, and revert softly after a success.
        """
        tuning = self.config.assimilation_tuning
        gating = summary.get("assimilation_gating") or {}
        if not isinstance(gating, dict):
            gating = {}
        low_power = int(gating.get("low_power", 0) or 0)
        topup_blocked = int(gating.get("topup_roi_blocked", 0) or 0)
        insufficient = int(gating.get("insufficient_scores", 0) or 0)
        promotions = int(summary.get("promotions", 0) or 0)
        merges = int(summary.get("merges", 0) or 0)
        # Initialize baselines once
        if not hasattr(self, "_nudge_baseline"):
            self._nudge_baseline = {
                "min_window": self._min_window_requirement(tuning),
                "holdout": int(getattr(tuning, "holdout_sample_size", 4)),
                "cap": int(getattr(tuning, "trial_per_gen_cap", 2)),
                "prob": int(getattr(tuning, "trial_probation_gens", 5)),
                "stipend": float(getattr(tuning, "trial_stipend", 0.5)),
                "bonus": float(getattr(tuning, "energy_topup_roi_bonus", 0.0)),
                "tau": float(self.config.controller.tau),
            }
        base = self._nudge_baseline  # type: ignore[attr-defined]
        # Decide whether to nudge up evidence or relax back
        stall = (
            (self.assim_fail_streak >= 8)
            or (low_power >= 2 and promotions == 0 and merges == 0)
            or (topup_blocked >= 5)
        )
        changed: dict[str, float] = {}
        # Optional: auto-tune min_window downward when insufficient_scores dominates
        if bool(getattr(tuning, "window_autotune", False)) and insufficient >= 50:
            mw = self._min_window_requirement(tuning)
            min_samples_floor = self._min_samples_required(tuning)
            lower = max(
                2 * min_samples_floor, int(getattr(tuning, "min_window_min", 2 * min_samples_floor))
            )
            if lower % 2 != 0:
                lower += 1
            if mw > lower:
                new_target = mw - 2
                normalized = self._set_min_window(tuning, new_target)
                if normalized != mw:
                    changed["min_window"] = normalized
        if stall:
            # Increase evidence and budget within bounds
            mw = self._min_window_requirement(tuning)
            ho = int(getattr(tuning, "holdout_sample_size", 4))
            cap = int(getattr(tuning, "trial_per_gen_cap", 2))
            prob = int(getattr(tuning, "trial_probation_gens", 5))
            stipend = float(getattr(tuning, "trial_stipend", 0.5))
            bonus = float(getattr(tuning, "energy_topup_roi_bonus", 0.0))
            tau = float(self.config.controller.tau)
            new_mw = min(12, mw + 2)
            new_ho = min(24, ho + 2)
            new_cap = min(4, cap + 1)
            new_prob = min(12, prob + 2)
            new_stipend = min(1.2, stipend + 0.1)
            new_bonus = min(1.5, bonus + 0.1)
            new_tau = max(0.35, tau - 0.01)
            normalized_mw = self._set_min_window(tuning, new_mw)
            if normalized_mw != mw:
                changed["min_window"] = normalized_mw
            if new_ho != ho:
                tuning.holdout_sample_size = new_ho
                changed["holdout_sample_size"] = new_ho
            if new_cap != cap:
                tuning.trial_per_gen_cap = new_cap
                changed["trial_per_gen_cap"] = new_cap
            if new_prob != prob:
                tuning.trial_probation_gens = new_prob
                changed["trial_probation_gens"] = new_prob
            if new_stipend != stipend:
                tuning.trial_stipend = new_stipend
                changed["trial_stipend"] = new_stipend
            if new_bonus != bonus:
                tuning.energy_topup_roi_bonus = new_bonus
                changed["energy_topup_roi_bonus"] = new_bonus
            if new_tau != tau:
                self.config.controller.tau = new_tau
                changed["tau"] = new_tau
        elif promotions > 0 or merges > 0:
            # Softly revert towards baselines after successes
            mw = self._min_window_requirement(tuning)
            ho = int(getattr(tuning, "holdout_sample_size", 4))
            cap = int(getattr(tuning, "trial_per_gen_cap", 2))
            prob = int(getattr(tuning, "trial_probation_gens", 5))
            stipend = float(getattr(tuning, "trial_stipend", 0.5))
            bonus = float(getattr(tuning, "energy_topup_roi_bonus", 0.0))
            tau = float(self.config.controller.tau)

            def step_towards(cur: float, tgt: float, step: float) -> float:
                if cur > tgt:
                    return max(tgt, cur - step)
                if cur < tgt:
                    return min(tgt, cur + step)
                return cur

            new_mw = int(step_towards(mw, base["min_window"], 2.0))
            new_ho = int(step_towards(ho, base["holdout"], 2.0))
            new_cap = int(step_towards(cap, base["cap"], 1.0))
            new_prob = int(step_towards(prob, base["prob"], 2.0))
            new_stipend = step_towards(stipend, base["stipend"], 0.1)
            new_bonus = step_towards(bonus, base["bonus"], 0.1)
            new_tau = step_towards(tau, base["tau"], 0.01)
            normalized_mw = self._set_min_window(tuning, new_mw)
            if normalized_mw != mw:
                changed["min_window"] = normalized_mw
            if new_ho != ho:
                tuning.holdout_sample_size = new_ho
                changed["holdout_sample_size"] = new_ho
            if new_cap != cap:
                tuning.trial_per_gen_cap = new_cap
                changed["trial_per_gen_cap"] = new_cap
            if new_prob != prob:
                tuning.trial_probation_gens = new_prob
                changed["trial_probation_gens"] = new_prob
            if new_stipend != stipend:
                tuning.trial_stipend = float(new_stipend)
                changed["trial_stipend"] = float(new_stipend)
            if new_bonus != bonus:
                tuning.energy_topup_roi_bonus = float(new_bonus)
                changed["energy_topup_roi_bonus"] = float(new_bonus)
            if new_tau != tau:
                self.config.controller.tau = float(new_tau)
                changed["tau"] = float(new_tau)
        if changed:
            summary["adaptive_nudges"] = {k: float(v) for k, v in changed.items()}

    def _collect_human_feedback(
        self, prompt: str, responses: dict[str, tuple[str, float]]
    ) -> Optional[dict[str, HumanFeedbackResult]]:
        if self.human_bandit is None or not self.config.human_bandit.enabled:
            return None
        freq = getattr(self.human_bandit, "frequency", self.config.human_bandit.frequency)
        freq = max(0.0, min(1.0, freq))
        if freq <= 0.0:
            return None
        interval = 1 if freq >= 1.0 else max(int(round(1.0 / freq)), 1)
        if self.bandit_counter % interval != 0:
            self.bandit_counter += 1
            return None
        self.bandit_counter += 1
        feedback: dict[str, HumanFeedbackResult] = {}
        for organelle_id, (answer, _energy) in responses.items():
            feedback[organelle_id] = self.human_bandit.solicit(prompt, answer)
        return feedback

    def _blend_rewards(
        self,
        base_rewards: dict[str, RewardBreakdown],
        human_feedback: dict[str, HumanFeedbackResult],
    ) -> dict[str, RewardBreakdown]:
        combined: dict[str, RewardBreakdown] = {}
        for organelle_id, reward in base_rewards.items():
            human = human_feedback.get(organelle_id)
            if human is None:
                combined[organelle_id] = reward
                continue
            hf = human.reward
            combined[organelle_id] = RewardBreakdown(
                task_reward=reward.task_reward + hf.task_reward,
                novelty_bonus=reward.novelty_bonus + hf.novelty_bonus,
                competence_bonus=reward.competence_bonus + hf.competence_bonus,
                helper_bonus=reward.helper_bonus + hf.helper_bonus,
                risk_penalty=reward.risk_penalty + hf.risk_penalty,
                cost_penalty=reward.cost_penalty + hf.cost_penalty,
            )
        return combined

    def _spawn_replacement_from(self, genome: Genome) -> None:
        mutant_template = self.population.mutate(genome)
        self._record_mutation_stats(mutant_template)
        new_id = self.host.spawn_organelle(
            rank=mutant_template.rank,
            hebbian_config=self.config.hebbian,
            activation_bias=mutant_template.gate_bias,
        )
        child_genome = Genome(
            organelle_id=new_id,
            drive_weights=mutant_template.drive_weights,
            gate_bias=mutant_template.gate_bias,
            rank=mutant_template.rank,
            explore_rate=mutant_template.explore_rate,
            post_rate=mutant_template.post_rate,
            read_rate=mutant_template.read_rate,
            hint_weight=mutant_template.hint_weight,
            beta_exploit=mutant_template.beta_exploit,
            q_decay=mutant_template.q_decay,
            ucb_bonus=mutant_template.ucb_bonus,
            budget_aggressiveness=mutant_template.budget_aggressiveness,
            rank_noise=dict(mutant_template.rank_noise),
            adapter_dropout=set(mutant_template.adapter_dropout),
            duplication_factors=dict(mutant_template.duplication_factors),
        )
        self.population.register(child_genome)

    def _record_mutation_stats(self, mutant: Genome) -> None:
        stats = self._mutation_stats_gen
        if not isinstance(stats, dict):
            stats = {"rank_noise": 0, "dropout": 0, "duplication": 0}
            self._mutation_stats_gen = stats
        if getattr(mutant, "rank_noise", None):
            stats["rank_noise"] = stats.get("rank_noise", 0) + 1
        if getattr(mutant, "adapter_dropout", None):
            if len(mutant.adapter_dropout) > 0:
                stats["dropout"] = stats.get("dropout", 0) + 1
        if getattr(mutant, "duplication_factors", None):
            if len(mutant.duplication_factors) > 0:
                stats["duplication"] = stats.get("duplication", 0) + 1

    def _compute_viability_map(self) -> Dict[str, bool]:
        threshold = self.config.energy.m
        viability: Dict[str, bool] = {}
        canary_failed = getattr(self.environment, "canary_failed", None)
        for organelle_id in self.host.list_organelle_ids():
            has_energy = self.host.ledger.energy_balance(organelle_id) >= threshold
            canary_ok = True
            if callable(canary_failed):
                try:
                    canary_ok = not canary_failed(organelle_id)
                except Exception:
                    canary_ok = True
            viability[organelle_id] = has_energy and canary_ok
        return viability

    def _mu_lambda_selection(self, viability: Dict[str, bool]) -> list[Genome]:
        strategy = self.config.population_strategy
        ranked = self.population.rank_for_selection(viability)
        if not ranked:
            return []
        mu = max(1, min(strategy.mu, len(ranked)))
        survivors = ranked[:mu]
        survivor_ids = {genome.organelle_id for genome in survivors}
        for genome in list(self.population.population.values()):
            if genome.organelle_id not in survivor_ids:
                self.host.retire_organelle(genome.organelle_id)
                self.population.remove(genome.organelle_id)
        return survivors

    def _spawn_offspring(self, survivors: list[Genome]) -> None:
        strategy = self.config.population_strategy
        if not survivors:
            return
        current_population = len(self.population.population)
        lambda_quota = max(0, min(strategy.lambda_, strategy.max_population - current_population))
        if lambda_quota <= 0:
            return
        for idx in range(lambda_quota):
            parent = survivors[idx % len(survivors)]
            self._spawn_replacement_from(parent)

    def _maybe_refresh_population(
        self, survivors: list[Genome]
    ) -> None:  # pragma: no cover - relies on stochastic long runs
        strategy = getattr(self.config, "population_strategy", None)
        if strategy is None:
            return
        interval = int(getattr(strategy, "refresh_interval", 0) or 0)
        if interval <= 0 or self.no_merge_counter < interval:
            return
        refresh_count = int(getattr(strategy, "refresh_count", 1) or 1)
        genomes = list(self.population.population.values())
        if not genomes:
            return
        survivor_ids = {g.organelle_id for g in survivors}
        genomes.sort(key=lambda g: self.population.average_roi(g.organelle_id, limit=5))
        candidates = [g for g in genomes if g.organelle_id not in survivor_ids]
        if len(candidates) < refresh_count:
            candidates = genomes
        retired = 0
        for genome in candidates[:refresh_count]:
            retired += 1
            self.host.retire_organelle(genome.organelle_id)
            self.population.remove(genome.organelle_id)
        parents = survivors or genomes[:1]
        if not parents:
            self._population_refresh_gen = {"count": 0, "reason": "no_parents"}
            return
        for idx in range(retired):
            parent = parents[idx % len(parents)]
            self._spawn_replacement_from(parent)
        self._population_refresh_gen = {
            "count": retired,
            "reason": "no_merges",
            "no_merge_counter": int(self.no_merge_counter),
        }
        self.no_merge_counter = 0

    def _apply_morphogenesis(self, survivors: list[Genome]) -> None:
        if not survivors or self.morphogenesis is None:
            return
        self.morphogenesis.apply(survivors, self.population)

    def _global_probe(
        self,
        organelle_id: str,
        cell: GridKey,
        gating: Dict[str, int],
        per_cell: int = 1,
    ) -> bool:
        passes = 0
        total_required: int

        # Always probe the focus cell first
        focus_tasks = [
            self.environment.sample_task_from_cell(cell, canary=False)
            for _ in range(max(1, per_cell))
        ]
        for task in focus_tasks:
            if not self._run_probe_task(organelle_id, task):
                gating["global_probe_failed"] += 1
                return False
        passes += len(focus_tasks)

        other_cells = [other for other in self.environment.iter_cells() if other != cell]
        max_others = self.config.assimilation_tuning.probe_max_other_cells
        if max_others is not None:
            max_others = max(0, max_others)
            other_cells = other_cells[:max_others]

        other_tasks: list[GridTask] = []
        for other in other_cells:
            other_tasks.append(self.environment.sample_task_from_cell(other, canary=False))

        required_passes = self.config.assimilation_tuning.probe_required_passes
        total_checks = len(focus_tasks) + len(other_tasks)
        if required_passes is None:
            total_required = total_checks
        else:
            total_required = min(max(1, required_passes), total_checks)

        for task in other_tasks:
            if self._run_probe_task(organelle_id, task):
                passes += 1
            # Early exit when meeting requirement
            if passes >= total_required:
                return True

        if passes >= total_required:
            return True
        gating["global_probe_failed"] += 1
        return False

    def _run_probe_task(self, organelle_id: str, task: GridTask) -> bool:
        result = self.host.step(
            prompt=task.prompt,
            intent="probe assimilation",
            max_routes=1,
            allowed_organelle_ids=[organelle_id],
        )
        metrics = result.responses.get(organelle_id)
        if metrics is None:
            return False
        success, _reward = task.evaluate(metrics.answer)
        return success

    def _select_soup_members(
        self, cell: GridKey, candidate_id: str
    ) -> tuple[list[str], dict[str, dict[str, float]]]:
        soup_size = max(2, self.config.assimilation_tuning.soup_size)
        stats_map: dict[str, dict[str, float]] = {}
        candidate_roi = max(self.population.average_roi(candidate_id, limit=5), 0.0)
        candidate_ema = float(self.environment.organism_stats.get(candidate_id, {}).get(cell, 0.0))
        # Compute simple cost bins (QD) from recent average energy to prefer similar-cost merges
        energies: dict[str, float] = {}
        for oid in self.population.population.keys():
            energies[oid] = float(self.population.average_energy(oid))
        energy_values = sorted(v for v in energies.values() if math.isfinite(v))
        bins: list[float] = []
        if energy_values:
            try:
                qs = quantiles(
                    energy_values,
                    n=max(2, getattr(self.config.qd, "cost_bins", 3)),
                    method="inclusive",
                )
                bins = [float(x) for x in qs]
            except Exception:
                bins = [energy_values[len(energy_values) // 2]]
        candidates: list[tuple[str, float, float, int, float]] = []

        def cost_bin(val: float) -> int:
            if not bins:
                return 0
            for i, edge in enumerate(bins):
                if val <= edge:
                    return i
            return len(bins)

        cand_bin = cost_bin(energies.get(candidate_id, 0.0))

        novelty_weight = float(getattr(self.config.qd, "novelty_weight", 0.3))
        novelty_floor = float(getattr(self.config.qd, "novelty_min", 0.05))
        candidate_novelty = self.population.cell_novelty(
            candidate_id, cell, scale=novelty_weight, floor=novelty_floor
        )
        stats_map[candidate_id] = {
            "roi": float(candidate_roi),
            "ema": candidate_ema,
            "novelty": float(candidate_novelty),
            "cost_bin": cand_bin,
        }

        for organelle_id, per_cell in self.environment.organism_stats.items():
            if organelle_id == candidate_id:
                continue
            ema = per_cell.get(cell)
            if ema is None:
                continue
            roi = self.population.average_roi(organelle_id, limit=5)
            mbin = cost_bin(energies.get(organelle_id, 0.0))
            novelty = self.population.cell_novelty(
                organelle_id, cell, scale=novelty_weight, floor=novelty_floor
            )
            candidates.append((organelle_id, float(ema), float(roi), mbin, float(novelty)))
        # Prefer same-bin mates; fallback to global best if insufficient
        candidates.sort(
            key=lambda item: (
                item[3] == cand_bin,
                item[4],
                item[1],
                item[2],
            ),
            reverse=True,
        )
        # ensure unique energy bins represented when possible
        seen_bins: set[int] = set()
        selected: list[tuple[str, float, float, int, float]] = []
        for entry in candidates:
            if len(selected) >= max(0, soup_size - 1):
                break
            bin_id = entry[3]
            if bin_id not in seen_bins or len(seen_bins) < soup_size - 1:
                selected.append(entry)
                seen_bins.add(bin_id)
        if len(selected) < max(0, soup_size - 1):
            for entry in candidates:
                if entry in selected:
                    continue
                selected.append(entry)
                if len(selected) >= max(0, soup_size - 1):
                    break
        for organelle_id, ema, roi, _mbin, novelty in selected:
            stats_map[organelle_id] = {
                "roi": max(roi, 0.0),
                "ema": float(ema),
                "novelty": float(novelty),
                "cost_bin": _mbin,
            }
        soup_ids = [candidate_id] + [entry[0] for entry in selected]
        return soup_ids, stats_map

    def _apply_lora_soup_merge(
        self,
        cell: GridKey,
        candidate_id: str,
        soup_ids: list[str],
        stats_map: dict[str, dict[str, float]],
        probe_records: list[dict[str, object]],
    ) -> list[dict[str, float]]:
        summary: list[dict[str, float]] = []
        weights: list[float] = []
        probe_boost = 1.0
        if probe_records:
            scores = [float(record.get("reward", 0.0)) for record in probe_records]
            positives = [score for score in scores if score > 0.0]
            if positives:
                probe_boost += sum(positives) / max(len(positives), 1)
        method = str(getattr(self.config.assimilation_tuning, "merge_method", "naive")).lower()
        novelty_weight = float(getattr(self.config.qd, "novelty_weight", 0.3))
        # optional fisher-style importance weighting
        importances: dict[str, float] = {}
        if method == "fisher_svd":

            def _fallback_importance(org: object | None) -> float:
                if org is None:
                    return 0.0
                try:
                    state = org.export_adapter_state()  # type: ignore[attr-defined]
                except Exception:
                    state = {}
                total = 0.0
                if isinstance(state, dict):
                    for tensor in state.values():
                        if isinstance(tensor, torch.Tensor):
                            try:
                                total += float(tensor.float().pow(2).sum().item())
                            except Exception:
                                continue
                return total

            for oid in soup_ids:
                org = self.host.get_organelle(oid)
                imp = 0.0
                if org is not None and hasattr(org, "fisher_importance"):
                    try:
                        imp = float(org.fisher_importance())  # type: ignore[attr-defined]
                    except Exception:
                        imp = 0.0
                if not imp or not math.isfinite(imp):
                    imp = _fallback_importance(org)
                importances[oid] = max(imp, 1e-6)
            max_importance = max(importances.values()) if importances else 1.0
            if max_importance <= 0.0 or not math.isfinite(max_importance):
                max_importance = 1.0
            importances = {k: (v / max_importance) for k, v in importances.items()}
        organelle = self.host.get_organelle(candidate_id)
        target_rank = (
            int(getattr(organelle, "rank", self.config.host.max_lora_rank))
            if organelle is not None
            else self.config.host.max_lora_rank
        )
        target_rank = max(1, min(target_rank, self.config.host.max_lora_rank))
        block_roles: dict[str, int] | None = None
        block_mode = False
        block_rank = target_rank
        if getattr(self.config.assimilation_tuning, "team_block_diagonal_merges", False):
            colony = self._find_member_colony(candidate_id)
            if colony is not None:
                _cid, meta = colony
                roles_meta = meta.get("roles")
                if isinstance(roles_meta, dict):
                    membership = set(meta.get("members", []))
                    if all(oid in membership for oid in soup_ids):
                        role_map: dict[str, int] = {}
                        valid = True
                        for oid in soup_ids:
                            role_val = roles_meta.get(oid)
                            if not isinstance(role_val, int):
                                valid = False
                                break
                            role_map[oid] = role_val
                        if valid and len(role_map) == len(soup_ids):
                            block_roles = role_map
                            rank_cap = int(
                                getattr(
                                    self.config.assimilation_tuning,
                                    "team_block_rank_cap",
                                    self.config.host.max_lora_rank,
                                )
                            )
                            summed_rank = 0
                            for oid in soup_ids:
                                genome = self.population.population.get(oid)
                                if genome is not None:
                                    summed_rank += max(1, int(getattr(genome, "rank", 1)))
                                else:
                                    organelle = self.host.get_organelle(oid)
                                    summed_rank += max(
                                        1,
                                        (
                                            int(getattr(organelle, "rank", target_rank))
                                            if organelle is not None
                                            else target_rank
                                        ),
                                    )
                            block_rank = max(
                                1, min(self.config.host.max_lora_rank, rank_cap, summed_rank)
                            )
                            block_mode = True

        mutation_meta: dict[str, dict[str, object]] = {}
        for oid in soup_ids:
            stats = stats_map.get(oid, {"roi": 0.0, "ema": 0.0})
            roi = max(stats.get("roi", 0.0), 0.0)
            ema = stats.get("ema", 0.0)
            weight = (roi + 1e-3) * (ema + 1e-3)
            if method == "fisher_svd":
                weight *= importances.get(oid, 1.0)
            if "novelty" in stats:
                weight *= 1.0 + max(0.0, float(stats["novelty"]))
            if oid == candidate_id:
                weight *= probe_boost
            if "novelty" in stats:
                weight *= 1.0 + novelty_weight * max(0.0, float(stats["novelty"]))
            record = {
                "organelle_id": oid,
                "weight": float(weight),
                "roi": float(roi),
                "ema": float(ema),
            }
            if importances:
                record["importance"] = float(importances.get(oid, 0.0))
            if "novelty" in stats:
                record["novelty"] = float(stats["novelty"])
            if "cost_bin" in stats:
                record["cost_bin"] = int(stats["cost_bin"])
            if block_roles and oid in block_roles:
                record["role"] = int(block_roles[oid])
            genome_meta = self.population.population.get(oid)
            if genome_meta is not None:
                if getattr(genome_meta, "rank_noise", None):
                    record["rank_noise"] = {
                        key: round(float(value), 3) for key, value in genome_meta.rank_noise.items()
                    }
                if getattr(genome_meta, "adapter_dropout", None):
                    if genome_meta.adapter_dropout:
                        record["dropout"] = sorted(genome_meta.adapter_dropout)
                if getattr(genome_meta, "duplication_factors", None):
                    if genome_meta.duplication_factors:
                        record["duplication"] = {
                            key: round(float(value), 3)
                            for key, value in genome_meta.duplication_factors.items()
                        }
                mutation_meta[oid] = {
                    "rank_noise": dict(getattr(genome_meta, "rank_noise", {})),
                    "dropout": sorted(getattr(genome_meta, "adapter_dropout", [])),
                    "duplication": dict(getattr(genome_meta, "duplication_factors", {})),
                }
            summary.append(record)
            weights.append(weight)
        weight_sum = sum(weights) or 1.0
        soup_map = {oid: (weight / weight_sum) for oid, weight in zip(soup_ids, weights)}
        if block_mode and block_roles:
            self.host.merge_lora_soup(
                soup_map,
                block_rank,
                roles=block_roles,
                mode="block",
                mutation_meta=mutation_meta or None,
            )
        else:
            self.host.merge_lora_soup(
                soup_map,
                target_rank,
                mutation_meta=mutation_meta or None,
            )
        return summary

    def _run_hf_probes(self, organelle_id: str) -> list[dict[str, object]]:
        prompts = list(self.config.assimilation_tuning.hf_prompts)
        if not prompts or self.human_bandit is None:
            return []
        records: list[dict[str, object]] = []
        for prompt in prompts:
            result = self.host.step(
                prompt=prompt,
                intent="human probe",
                max_routes=1,
                allowed_organelle_ids=[organelle_id],
            )
            metrics = result.responses.get(organelle_id)
            if metrics is None:
                records.append(
                    {
                        "prompt": prompt,
                        "passed": False,
                        "reward": 0.0,
                        "notes": "no-metrics",
                    }
                )
                continue
            answer = metrics.answer
            feedback = self.human_bandit.solicit(prompt, answer)
            reward_total = feedback.reward.total
            records.append(
                {
                    "prompt": prompt,
                    "passed": reward_total > 0.0,
                    "reward": reward_total,
                    "notes": feedback.notes,
                }
            )
        return records


__all__ = ["EcologyLoop"]
