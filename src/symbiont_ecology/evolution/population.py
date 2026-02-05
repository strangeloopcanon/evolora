"""Population-based evolution primitives."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

from symbiont_ecology.config import EvolutionConfig, ForagingConfig


@dataclass
class Genome:
    organelle_id: str
    drive_weights: dict[str, float]
    gate_bias: float
    rank: int
    # evolved behavioural traits (defaults are neutral)
    explore_rate: float = 0.5
    post_rate: float = 0.0
    read_rate: float = 0.0
    hint_weight: float = 0.0
    beta_exploit: float = 1.5
    q_decay: float = 0.3
    ucb_bonus: float = 0.2
    budget_aggressiveness: float = 0.5
    rank_noise: dict[str, float] = field(default_factory=dict)
    adapter_dropout: set[str] = field(default_factory=set)
    duplication_factors: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.explore_rate = max(0.0, min(1.0, float(self.explore_rate)))
        self.budget_aggressiveness = max(0.0, min(1.0, float(self.budget_aggressiveness)))
        if abs(self.budget_aggressiveness - 0.5) < 1e-6 and abs(self.explore_rate - 0.5) > 1e-6:
            self.budget_aggressiveness = self.explore_rate
        self.explore_rate = self.budget_aggressiveness
        self.post_rate = max(0.0, min(1.0, float(self.post_rate)))
        self.read_rate = max(0.0, min(1.0, float(self.read_rate)))
        self.hint_weight = max(0.0, min(1.0, float(self.hint_weight)))
        self.beta_exploit = max(0.0, float(self.beta_exploit))
        if not math.isfinite(self.beta_exploit):
            self.beta_exploit = 1.5
        self.q_decay = max(0.01, min(0.99, float(self.q_decay)))
        self.ucb_bonus = max(0.0, float(self.ucb_bonus))
        if not math.isfinite(self.ucb_bonus):
            self.ucb_bonus = 0.2
        clean_noise: dict[str, float] = {}
        for key, value in (self.rank_noise or {}).items():
            if not isinstance(key, str) or not key:
                continue
            try:
                clean_noise[key] = max(-3.0, min(3.0, float(value)))
            except Exception:
                continue
        self.rank_noise = clean_noise
        dropout_set: set[str] = set()
        for item in self.adapter_dropout or set():
            if isinstance(item, str) and item:
                dropout_set.add(item)
        self.adapter_dropout = dropout_set
        clean_duplication: dict[str, float] = {}
        for key, value in (self.duplication_factors or {}).items():
            if not isinstance(key, str) or not key:
                continue
            try:
                factor = max(0.0, min(3.0, float(value)))
            except Exception:
                continue
            if factor > 0.0:
                clean_duplication[key] = factor
        self.duplication_factors = clean_duplication


@dataclass
class PopulationManager:
    config: EvolutionConfig
    foraging: ForagingConfig | None = None
    population: dict[str, Genome] = field(default_factory=dict)
    history: dict[str, List[float]] = field(default_factory=dict)
    score_meta: dict[str, List[dict[str, object]]] = field(default_factory=dict)
    ages: dict[str, int] = field(default_factory=dict)
    energy: dict[str, List[float]] = field(default_factory=dict)
    roi: dict[str, List[float]] = field(default_factory=dict)
    adapter_usage: dict[str, dict[str, List[float]]] = field(default_factory=dict)
    energy_delta: dict[str, List[float]] = field(default_factory=dict)
    assimilation_history: dict[tuple[str, str, str], list[dict[str, object]]] = field(
        default_factory=dict
    )
    evidence_credit: dict[str, int] = field(default_factory=dict)
    cell_values: dict[str, dict[tuple[str, str], float]] = field(default_factory=dict)
    cell_counts: dict[str, dict[tuple[str, str], int]] = field(default_factory=dict)
    global_cell_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    assimilation_history_limit: int | None = None

    def register(self, genome: Genome) -> None:
        self.population[genome.organelle_id] = genome
        self.history.setdefault(genome.organelle_id, [])
        self.score_meta.setdefault(genome.organelle_id, [])
        self.energy.setdefault(genome.organelle_id, [])
        self.roi.setdefault(genome.organelle_id, [])
        self.adapter_usage.setdefault(genome.organelle_id, {})
        self.energy_delta.setdefault(genome.organelle_id, [])
        self.ages[genome.organelle_id] = 0
        self.cell_values.setdefault(genome.organelle_id, {})
        self.cell_counts.setdefault(genome.organelle_id, {})

    def select_parents(self, k: int = 2) -> list[Genome]:
        return random.sample(  # nosec B311
            list(self.population.values()), min(k, len(self.population))
        )

    def mutate(self, genome: Genome) -> Genome:
        mutated_weights: dict[str, float] = {}
        for key, weight in genome.drive_weights.items():
            mutated_weights[key] = weight + random.gauss(0, self.config.mutation_rate)
        if random.random() < 0.3:  # nosec B311
            mutated_weights["word_count_focus"] = mutated_weights.get(
                "word_count_focus", 0.0
            ) + random.gauss(0, self.config.mutation_rate * 0.5)
        if random.random() < 0.2:  # nosec B311
            mutated_weights["logic_focus"] = mutated_weights.get("logic_focus", 0.0) + random.gauss(
                0, self.config.mutation_rate * 0.4
            )
        rank_delta = random.choice([-1, 0, 0, 1])  # nosec B311
        mutated_rank = max(1, genome.rank + rank_delta)
        rank_noise = dict(getattr(genome, "rank_noise", {}))
        adapter_dropout = set(getattr(genome, "adapter_dropout", set()))
        duplication = dict(getattr(genome, "duplication_factors", {}))
        layer_tags = list(getattr(self.config, "mutation_layer_tags", [])) or [
            "attn",
            "mlp",
            "proj",
        ]
        dropout_decay = max(
            0.0, min(1.0, float(getattr(self.config, "mutation_dropout_decay", 0.0)))
        )
        if adapter_dropout and random.random() < dropout_decay:  # nosec B311
            adapter_dropout_list = list(adapter_dropout)
            adapter_dropout.discard(random.choice(adapter_dropout_list))  # nosec B311
        dropout_prob = max(0.0, min(1.0, float(getattr(self.config, "mutation_dropout_prob", 0.0))))
        if layer_tags and random.random() < dropout_prob:  # nosec B311
            adapter_dropout.add(random.choice(layer_tags))  # nosec B311
        duplication_prob = max(
            0.0, min(1.0, float(getattr(self.config, "mutation_duplication_prob", 0.0)))
        )
        duplication_scale = max(0.0, float(getattr(self.config, "mutation_duplication_scale", 0.5)))
        if layer_tags and random.random() < duplication_prob:  # nosec B311
            tag = random.choice(layer_tags)  # nosec B311
            delta = abs(random.gauss(0.0, duplication_scale))
            duplication[tag] = max(0.0, min(3.0, duplication.get(tag, 0.0) + delta))
        elif duplication and random.random() < 0.1:  # allow occasional decay; nosec B311
            tag = random.choice(list(duplication.keys()))  # nosec B311
            duplication[tag] = max(0.0, duplication[tag] * 0.5)
            if duplication[tag] < 0.05:
                duplication.pop(tag, None)
        rank_noise_prob = max(
            0.0, min(1.0, float(getattr(self.config, "mutation_rank_noise_prob", 0.0)))
        )
        rank_noise_scale = max(0.0, float(getattr(self.config, "mutation_rank_noise_scale", 1.0)))
        if layer_tags and random.random() < rank_noise_prob:  # nosec B311
            tag = random.choice(layer_tags)  # nosec B311
            noise = random.gauss(0.0, rank_noise_scale)
            rank_noise[tag] = max(-3.0, min(3.0, rank_noise.get(tag, 0.0) + noise))

        forage_cfg = self.foraging
        base_sigma = max(1e-3, float(self.config.mutation_rate))
        beta_sigma = base_sigma * (forage_cfg.mutation_beta_scale if forage_cfg else 0.3)
        decay_sigma = base_sigma * (forage_cfg.mutation_decay_scale if forage_cfg else 0.2)
        ucb_sigma = base_sigma * (forage_cfg.mutation_ucb_scale if forage_cfg else 0.25)
        budget_sigma = base_sigma * (forage_cfg.mutation_budget_scale if forage_cfg else 0.25)
        beta_exploit = max(0.05, min(6.0, genome.beta_exploit + random.gauss(0, beta_sigma)))
        q_decay = max(0.05, min(0.95, genome.q_decay + random.gauss(0, decay_sigma)))
        ucb_bonus = max(0.0, min(5.0, genome.ucb_bonus + random.gauss(0, ucb_sigma)))
        budget_aggressiveness = max(
            0.0, min(1.0, genome.budget_aggressiveness + random.gauss(0, budget_sigma))
        )
        post_rate = max(0.0, min(1.0, genome.post_rate + random.gauss(0, 0.05)))
        read_rate = max(0.0, min(1.0, genome.read_rate + random.gauss(0, 0.05)))
        hint_weight = max(0.0, min(1.0, genome.hint_weight + random.gauss(0, 0.05)))
        return Genome(
            organelle_id=genome.organelle_id,
            drive_weights=mutated_weights,
            gate_bias=genome.gate_bias + random.gauss(0, 0.1),
            rank=mutated_rank,
            explore_rate=budget_aggressiveness,
            post_rate=post_rate,
            read_rate=read_rate,
            hint_weight=hint_weight,
            beta_exploit=beta_exploit,
            q_decay=q_decay,
            ucb_bonus=ucb_bonus,
            budget_aggressiveness=budget_aggressiveness,
            rank_noise=rank_noise,
            adapter_dropout=adapter_dropout,
            duplication_factors=duplication,
        )

    def niche(self, genome: Genome) -> int:
        novelty = abs(genome.drive_weights.get("novelty", 0.0))
        bins: int = int(self.config.niching_bins)
        if bins <= 0:
            return 0
        return int(novelty * bins) % bins

    def record_score(
        self, organelle_id: str, score: float, *, meta: dict[str, object] | None = None
    ) -> None:
        self.history.setdefault(organelle_id, []).append(score)
        payload: dict[str, object]
        if meta is None:
            payload = {"score": score}
        else:
            payload = dict(meta)
            payload.setdefault("score", score)
        self.score_meta.setdefault(organelle_id, []).append(payload)

    def record_energy(self, organelle_id: str, value: float) -> None:
        self.energy.setdefault(organelle_id, []).append(value)

    def record_energy_delta(self, organelle_id: str, delta: float) -> None:
        self.energy_delta.setdefault(organelle_id, []).append(delta)
        # accumulate evidence credits on positive deltas as a weak proxy
        if delta > 0:
            self.evidence_credit[organelle_id] = self.evidence_credit.get(organelle_id, 0) + 1

    def record_roi(self, organelle_id: str, value: float) -> None:
        if not math.isfinite(value):
            return
        self.roi.setdefault(organelle_id, []).append(value)

    def record_adapter_usage(self, organelle_id: str, modules: dict[str, int], tokens: int) -> None:
        if tokens <= 0:
            return
        per_module = self.adapter_usage.setdefault(organelle_id, {})
        weight = float(tokens)
        for module, count in modules.items():
            if module in {"rank", "total"}:
                continue
            per_module.setdefault(module, []).append(weight * max(count, 1))

    def update_cell_value(
        self,
        organelle_id: str,
        cell: tuple[str, str],
        roi: float,
        *,
        decay: float,
        q_init: float = 0.0,
    ) -> None:
        stats = self.cell_values.setdefault(organelle_id, {})
        counts = self.cell_counts.setdefault(organelle_id, {})
        prev = stats.get(cell, q_init)
        coeff = max(0.01, min(0.99, decay))
        stats[cell] = (1.0 - coeff) * prev + coeff * roi
        counts[cell] = counts.get(cell, 0) + 1
        self.global_cell_counts[cell] = self.global_cell_counts.get(cell, 0) + 1

    def top_cells(self, organelle_id: str, limit: int = 3) -> list[tuple[str, str, float]]:
        stats = self.cell_values.get(organelle_id)
        if not stats:
            return []
        top = sorted(stats.items(), key=lambda item: item[1], reverse=True)[: max(1, limit)]
        return [(cell[0], cell[1], float(value)) for cell, value in top]

    def recent_scores(self, organelle_id: str, limit: int = 10) -> list[float]:
        return self.history.get(organelle_id, [])[-limit:]

    def recent_score_records(self, organelle_id: str, limit: int = 10) -> list[dict[str, object]]:
        records = self.score_meta.get(organelle_id, [])[-limit:]
        if records:
            return [dict(record) for record in records]
        scores = self.history.get(organelle_id, [])[-limit:]
        return [{"score": score} for score in scores]

    def average_score(self, organelle_id: str, limit: int = 10) -> float:
        scores = self.recent_scores(organelle_id, limit)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def average_task_reward(self, organelle_id: str, limit: int = 10) -> float:
        """Calculate average raw task reward (competence) from metadata."""
        records = self.recent_score_records(organelle_id, limit)
        if not records:
            return 0.0
        # If task_reward is missing (old records), fall back to the recorded score.
        values: list[float] = []
        for record in records:
            task_reward = record.get("task_reward")
            if isinstance(task_reward, (int, float)):
                values.append(float(task_reward))
                continue
            score = record.get("score")
            if isinstance(score, (int, float)):
                values.append(float(score))
                continue
            if isinstance(score, str):
                try:
                    values.append(float(score))
                except Exception:
                    continue
        if not values:
            return 0.0
        return sum(values) / len(values)

    def average_energy(self, organelle_id: str, limit: int = 10) -> float:
        values = self.energy.get(organelle_id, [])[-limit:]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def average_roi(self, organelle_id: str, limit: int = 10) -> float:
        values = self.roi.get(organelle_id, [])[-limit:]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def aggregate_roi(self, limit: int = 10) -> float:
        samples: list[float] = []
        for organelle_id in self.population.keys():
            values = self.roi.get(organelle_id, [])[-limit:]
            samples.extend(values)
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def aggregate_energy(self, limit: int = 10) -> float:
        samples: list[float] = []
        for organelle_id in self.population.keys():
            values = self.energy.get(organelle_id, [])[-limit:]
            samples.extend(values)
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def cell_novelty(
        self,
        organelle_id: str,
        cell: tuple[str, str],
        *,
        scale: float = 0.3,
        floor: float = 0.05,
    ) -> float:
        """Return a simple novelty score for a (organelle, cell) pair.

        Novelty is inversely proportional to the global visitation count for the cell
        and the per-organism visitation count. Higher novelty => rarer behaviour.
        """
        global_hits = max(0, self.global_cell_counts.get(cell, 0))
        per_org_hits = max(0, self.cell_counts.get(organelle_id, {}).get(cell, 0))
        novelty = 0.0
        novelty += 1.0 / math.sqrt(1.0 + global_hits)
        novelty += 0.5 / math.sqrt(1.0 + per_org_hits)
        novelty = max(floor, novelty * max(0.0, scale))
        return float(novelty)

    def recent_energy_deltas(self, organelle_id: str, limit: int = 10) -> list[float]:
        return self.energy_delta.get(organelle_id, [])[-limit:]

    def average_adapter_usage(self, organelle_id: str, module: str, limit: int = 10) -> float:
        history = self.adapter_usage.get(organelle_id, {}).get(module, [])
        if not history:
            return 0.0
        samples = history[-limit:]
        return sum(samples) / len(samples)

    def module_utilisation(self, limit: int = 10) -> dict[str, list[tuple[str, float]]]:
        summary: dict[str, list[tuple[str, float]]] = {}
        for organelle_id, per_module in self.adapter_usage.items():
            for module, samples in per_module.items():
                if not samples:
                    continue
                utilisation = sum(samples[-limit:]) / min(len(samples), limit)
                summary.setdefault(module, []).append((organelle_id, utilisation))
        return summary

    def remove(self, organelle_id: str) -> None:
        self.population.pop(organelle_id, None)
        self.history.pop(organelle_id, None)
        self.score_meta.pop(organelle_id, None)
        self.energy.pop(organelle_id, None)
        self.ages.pop(organelle_id, None)
        self.roi.pop(organelle_id, None)
        self.adapter_usage.pop(organelle_id, None)
        self.energy_delta.pop(organelle_id, None)
        self.evidence_credit.pop(organelle_id, None)

    # Evidence credit helpers (for endogenous power economics)
    def grant_evidence(self, organelle_id: str, amount: int = 1, *, cap: int | None = None) -> int:
        minted = max(0, int(amount))
        if minted <= 0:
            return 0
        current = int(self.evidence_credit.get(organelle_id, 0) or 0)
        if cap is not None:
            cap = max(0, int(cap))
            if current >= cap:
                return 0
            minted = min(minted, max(0, cap - current))
        if minted <= 0:
            return 0
        self.evidence_credit[organelle_id] = current + minted
        return minted

    def consume_evidence(self, organelle_id: str, amount: int = 1) -> bool:
        cur = self.evidence_credit.get(organelle_id, 0)
        if cur >= amount > 0:
            self.evidence_credit[organelle_id] = cur - amount
            return True
        return False

    def evidence_tokens(self, organelle_id: str) -> int:
        return int(self.evidence_credit.get(organelle_id, 0) or 0)

    def top_performers(self, limit: int = 1) -> list[Genome]:
        ranked = sorted(
            self.population.values(),
            key=lambda genome: self.average_score(genome.organelle_id),
            reverse=True,
        )
        return ranked[:limit]

    def increment_ages(self) -> None:
        for k in list(self.population.keys()):
            self.ages[k] = self.ages.get(k, 0) + 1

    def prune_excess(self) -> list[str]:
        removed: list[str] = []
        max_pop = self.config.max_population
        if len(self.population) <= max_pop:
            return removed
        ranked = sorted(
            self.population.values(),
            key=lambda g: (self.average_score(g.organelle_id), -self.ages.get(g.organelle_id, 0)),
        )
        to_remove = len(self.population) - max_pop
        for genome in ranked[:to_remove]:
            removed.append(genome.organelle_id)
        return removed

    def rank_for_selection(
        self, viability: dict[str, bool], magnitudes: dict[str, float] | None = None
    ) -> list[Genome]:
        def key(genome: Genome) -> tuple[float, float, float, float]:
            viable = 1.0 if viability.get(genome.organelle_id, False) else 0.0
            roi = self.average_roi(genome.organelle_id)
            competence = self.average_task_reward(genome.organelle_id)
            # Tie-break: prefer adapters with non-zero magnitude (avoid no-ops).
            mag = magnitudes.get(genome.organelle_id, 0.0) if magnitudes else 0.0
            return (viable, competence, mag, roi)

        return sorted(
            self.population.values(),
            key=key,
            reverse=True,
        )

    def record_assimilation(
        self,
        organelle_id: str,
        cell: tuple[str, str],
        record: dict[str, object],
    ) -> None:
        key = (organelle_id, cell[0], cell[1])
        history = self.assimilation_history.setdefault(key, [])
        history.append(record)
        limit = getattr(self, "assimilation_history_limit", None)
        if isinstance(limit, int) and limit > 0 and len(history) > limit:
            self.assimilation_history[key] = history[-limit:]

    def assimilation_records(
        self, organelle_id: str, cell: tuple[str, str], limit: int = 10
    ) -> list[dict[str, object]]:
        key = (organelle_id, cell[0], cell[1])
        records = self.assimilation_history.get(key, [])
        if not records:
            return []
        return records[-limit:]


__all__ = ["Genome", "PopulationManager"]
