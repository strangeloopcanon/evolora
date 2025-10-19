from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.evolution.population import Genome, PopulationManager


def _build_loop(roi: float, balance: float = 0.2) -> tuple[EcologyLoop, Genome, ATPLedger]:
    config = EcologyConfig()
    config.assimilation_tuning.energy_floor = 1.0
    config.assimilation_tuning.energy_floor_roi = 0.1
    ledger = ATPLedger()
    ledger.configure_energy_cap(2.0)
    organelle_id = "org-test"
    ledger.ensure_energy(organelle_id, balance)
    ledger.set_energy(organelle_id, balance)
    host = SimpleNamespace(ledger=ledger)
    population = MagicMock(spec=PopulationManager)
    population.average_roi.return_value = roi
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=MagicMock(),
        population=population,
        assimilation=MagicMock(spec=AssimilationTester),
    )
    genome = Genome(organelle_id=organelle_id, drive_weights={}, gate_bias=0.0, rank=1)
    return loop, genome, ledger


def test_energy_topup_applies_when_roi_high() -> None:
    loop, genome, ledger = _build_loop(roi=0.4)
    before = ledger.energy_balance(genome.organelle_id)
    boosted = loop._maybe_top_up_energy(genome, before)
    assert pytest.approx(boosted, abs=1e-6) == 1.0
    assert ledger.energy_balance(genome.organelle_id) == pytest.approx(boosted, abs=1e-6)


def test_energy_topup_skipped_without_roi_signal() -> None:
    loop, genome, ledger = _build_loop(roi=0.05)
    before = ledger.energy_balance(genome.organelle_id)
    boosted = loop._maybe_top_up_energy(genome, before)
    assert boosted == pytest.approx(before, abs=1e-6)


def test_auto_tune_energy_floor_updates_config() -> None:
    config = EcologyConfig()
    config.assimilation_tuning.energy_floor = 0.0
    config.assimilation_tuning.energy_floor_roi = 0.0
    config.assimilation_tuning.energy_floor_base = 0.8
    config.assimilation_tuning.energy_floor_roi_base = 1.1
    config.environment.synthetic_batch_size = 2
    config.energy.m = 1.0
    ledger = ATPLedger()
    ledger.configure_energy_cap(3.0)
    host = SimpleNamespace(ledger=ledger)
    population = PopulationManager(config.evolution)
    genome = Genome(organelle_id="org-auto", drive_weights={}, gate_bias=0.0, rank=1)
    population.register(genome)
    samples = [
        (0.45, 1.4, 0.05),
        (0.52, 2.1, 0.12),
        (0.48, 2.4, 0.18),
        (0.50, 1.9, -0.04),
    ]
    for cost, roi, delta in samples:
        population.record_energy(genome.organelle_id, cost)
        population.record_roi(genome.organelle_id, roi)
        population.record_energy_delta(genome.organelle_id, delta)
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=MagicMock(),
        population=population,
        assimilation=MagicMock(spec=AssimilationTester),
    )
    summary: dict[str, object] = {"energy_balance": {genome.organelle_id: 1.5}}
    loop._auto_tune_assimilation_energy(summary)
    assert "assimilation_energy_tuning" in summary
    tuning = config.assimilation_tuning
    tuning_summary = summary["assimilation_energy_tuning"]
    assert tuning.energy_floor >= max(config.energy.m, config.assimilation_tuning.energy_floor_base)
    avg_cost = sum(cost for cost, _roi, _delta in samples) / len(samples)
    roi_required = 1.0 + config.energy.m / (config.environment.synthetic_batch_size * avg_cost)
    roi_cap = max(roi_required * 3.0, roi_required + 1.5, 5.0)
    assert tuning.energy_floor_roi >= 1.0 - 1e-6
    assert tuning.energy_floor_roi <= roi_cap + 1e-6
    assert tuning_summary["bandit_choice"] in {0, 1, 2, 3}
