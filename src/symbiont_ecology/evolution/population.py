"""Population-based evolution primitives."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

from symbiont_ecology.config import EvolutionConfig


@dataclass
class Genome:
    organelle_id: str
    drive_weights: dict[str, float]
    gate_bias: float
    rank: int


@dataclass
class PopulationManager:
    config: EvolutionConfig
    population: dict[str, Genome] = field(default_factory=dict)
    history: dict[str, List[float]] = field(default_factory=dict)
    ages: dict[str, int] = field(default_factory=dict)
    energy: dict[str, List[float]] = field(default_factory=dict)
    roi: dict[str, List[float]] = field(default_factory=dict)
    adapter_usage: dict[str, dict[str, List[float]]] = field(default_factory=dict)
    energy_delta: dict[str, List[float]] = field(default_factory=dict)
    assimilation_history: dict[tuple[str, str, str], list[dict[str, object]]] = field(default_factory=dict)

    def register(self, genome: Genome) -> None:
        self.population[genome.organelle_id] = genome
        self.history.setdefault(genome.organelle_id, [])
        self.energy.setdefault(genome.organelle_id, [])
        self.roi.setdefault(genome.organelle_id, [])
        self.adapter_usage.setdefault(genome.organelle_id, {})
        self.energy_delta.setdefault(genome.organelle_id, [])
        self.ages[genome.organelle_id] = 0

    def select_parents(self, k: int = 2) -> list[Genome]:
        return random.sample(  # nosec B311
            list(self.population.values()), min(k, len(self.population))
        )

    def mutate(self, genome: Genome) -> Genome:
        candidate_keys = set(genome.drive_weights.keys()) | {"novelty", "word_count_focus", "logic_focus"}
        mutated_weights: dict[str, float] = {}
        for key in candidate_keys:
            base = genome.drive_weights.get(key, 0.0)
            sigma = self.config.mutation_rate * (1.5 if key != "novelty" else 1.0)
            mutated_weights[key] = base + random.gauss(0, sigma)
        rank_delta = random.choice([-2, -1, 0, 0, 1, 2])  # nosec B311
        mutated_rank = max(1, genome.rank + rank_delta)
        return Genome(
            organelle_id=genome.organelle_id,
            drive_weights=mutated_weights,
            gate_bias=genome.gate_bias + random.gauss(0, 0.15),
            rank=mutated_rank,
        )

    def niche(self, genome: Genome) -> int:
        novelty = abs(genome.drive_weights.get("novelty", 0.0))
        bins: int = int(self.config.niching_bins)
        if bins <= 0:
            return 0
        return int(novelty * bins) % bins

    def record_score(self, organelle_id: str, score: float) -> None:
        self.history.setdefault(organelle_id, []).append(score)

    def record_energy(self, organelle_id: str, value: float) -> None:
        self.energy.setdefault(organelle_id, []).append(value)

    def record_energy_delta(self, organelle_id: str, delta: float) -> None:
        self.energy_delta.setdefault(organelle_id, []).append(delta)

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

    def recent_scores(self, organelle_id: str, limit: int = 10) -> list[float]:
        return self.history.get(organelle_id, [])[-limit:]

    def average_score(self, organelle_id: str, limit: int = 10) -> float:
        scores = self.recent_scores(organelle_id, limit)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

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
        self.energy.pop(organelle_id, None)
        self.ages.pop(organelle_id, None)
        self.roi.pop(organelle_id, None)
        self.adapter_usage.pop(organelle_id, None)
        self.energy_delta.pop(organelle_id, None)

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

    def rank_for_selection(self, viability: dict[str, bool]) -> list[Genome]:
        def key(genome: Genome) -> tuple[float, float, float]:
            viable = 1.0 if viability.get(genome.organelle_id, False) else 0.0
            roi = self.average_roi(genome.organelle_id)
            score = self.average_score(genome.organelle_id)
            return (viable, roi, score)

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

    def assimilation_records(self, organelle_id: str, cell: tuple[str, str], limit: int = 10) -> list[dict[str, object]]:
        key = (organelle_id, cell[0], cell[1])
        records = self.assimilation_history.get(key, [])
        if not records:
            return []
        return records[-limit:]


__all__ = ["Genome", "PopulationManager"]
