"""Morphogenesis controller for LoRA growth/shrink mutations."""

from __future__ import annotations

from dataclasses import dataclass

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.evolution.population import Genome, PopulationManager
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.organelles.base import Organelle


@dataclass
class MorphogenesisController:
    config: EcologyConfig
    host: HostKernel

    def apply(self, survivors: list[Genome], population: PopulationManager) -> None:
        if not survivors:
            return
        limits = self.config.limits
        backbone_params = max(self.host.total_backbone_params(), 1)
        base_budget = backbone_params * limits.lora_budget_frac
        adapter_cap = max(0, limits.max_active_adapters_per_layer)
        roi_ranked: list[tuple[Genome, float, Organelle]] = []
        for genome in survivors:
            organelle = self.host.get_organelle(genome.organelle_id)
            if organelle is None:
                continue
            roi_history = population.roi.get(genome.organelle_id, [])
            if not roi_history:
                # Avoid rank collapse for organs that didn't get meaningful interaction this window.
                # Still nudge gating bias based on a neutral ROI prior.
                self._tweak_gate_bias(genome, roi=population.aggregate_roi(limit=5))
                continue
            roi = population.average_roi(genome.organelle_id, limit=5)
            roi_ranked.append((genome, roi, organelle))

        if not roi_ranked:
            return
        roi_ranked.sort(key=lambda item: item[1])
        n = len(roi_ranked)
        # Small populations: preserve the original absolute ROI thresholds.
        # (Used by unit tests and keeps behaviour intuitive when mu < 4.)
        if n < 4:
            for genome, roi, organelle in roi_ranked:
                target_rank = genome.rank
                if roi >= 1.2:
                    target_rank = min(genome.rank + 1, self.config.host.max_lora_rank)
                elif roi <= 0.6 and genome.rank > 1:
                    target_rank = max(1, genome.rank - 1)
                else:
                    self._tweak_gate_bias(genome, roi)
                    continue

                if target_rank == genome.rank:
                    continue

                current_total = self.host.total_trainable_parameters()
                current_org = self.host._trainable_params(organelle)
                estimated = self.host.estimate_trainable(organelle, target_rank)
                projected_total = current_total - current_org + estimated
                budget = max(base_budget, projected_total)

                if target_rank > genome.rank and projected_total > budget:
                    continue

                if self.host.resize_organelle_rank(genome.organelle_id, target_rank):
                    genome.rank = target_rank
            if adapter_cap:
                self._enforce_layer_caps(survivors, population, adapter_cap)
            return

        # Scale-free: adjust ranks by relative ROI within the current survivors set.
        # With mu=4 this becomes 1 shrink + 1 grow per gen (when ranks allow).
        low_count = int(n // 4)
        high_start = n - low_count

        for idx, (genome, roi, organelle) in enumerate(roi_ranked):
            target_rank = genome.rank
            if low_count > 0 and idx < low_count and genome.rank > 1:
                target_rank = max(1, genome.rank - 1)
            elif low_count > 0 and idx >= high_start:
                target_rank = min(genome.rank + 1, self.config.host.max_lora_rank)
            else:
                self._tweak_gate_bias(genome, roi)
                continue

            if target_rank == genome.rank:
                continue

            current_total = self.host.total_trainable_parameters()
            current_org = self.host._trainable_params(organelle)
            estimated = self.host.estimate_trainable(organelle, target_rank)
            projected_total = current_total - current_org + estimated
            budget = max(base_budget, projected_total)

            if target_rank > genome.rank and projected_total > budget:
                continue

            if self.host.resize_organelle_rank(genome.organelle_id, target_rank):
                genome.rank = target_rank
        if adapter_cap:
            self._enforce_layer_caps(survivors, population, adapter_cap)

    @staticmethod
    def _tweak_gate_bias(genome: Genome, roi: float) -> None:
        delta = 0.05 if roi >= 1.0 else -0.05
        genome.gate_bias += delta

    def _enforce_layer_caps(
        self,
        survivors: list[Genome],
        population: PopulationManager,
        cap: int,
    ) -> None:
        module_summary: dict[str, list[tuple[Genome, float]]] = {}
        for genome in survivors:
            organelle = self.host.get_organelle(genome.organelle_id)
            if organelle is None:
                continue
            adapters = self.host._active_adapters(organelle)
            for module, count in adapters.items():
                if module in {"rank", "total"} or count <= 0:
                    continue
                utilisation = population.average_adapter_usage(
                    genome.organelle_id, module, limit=10
                )
                module_summary.setdefault(module, []).append((genome, utilisation))

        for _module, entries in module_summary.items():
            if len(entries) <= cap:
                continue
            entries.sort(key=lambda item: (item[1], population.average_roi(item[0].organelle_id)))
            over_subscribed = len(entries) - cap
            pruned = 0
            idx = 0
            while pruned < over_subscribed and idx < len(entries):
                genome, _utilisation = entries[idx]
                idx += 1
                organelle = self.host.get_organelle(genome.organelle_id)
                if organelle is None:
                    continue
                if genome.rank > 1:
                    target_rank = max(1, genome.rank - 1)
                    if self.host.resize_organelle_rank(genome.organelle_id, target_rank):
                        genome.rank = target_rank
                        pruned += 1
                        continue
                genome.gate_bias -= 0.25
                pruned += 1


__all__ = ["MorphogenesisController"]
