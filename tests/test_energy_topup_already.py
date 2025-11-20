from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.evolution.population import Genome, PopulationManager


def test_topup_already_sufficient_branch() -> None:
    cfg = EcologyConfig()
    cfg.assimilation_tuning.energy_floor = 0.5
    cfg.assimilation_tuning.energy_floor_roi = 0.2
    ledger = ATPLedger()
    org = "org-x"
    ledger.configure_energy_cap(2.0)
    ledger.ensure_energy(org, 0.8)
    host = SimpleNamespace(ledger=ledger)
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    genome = Genome(organelle_id=org, drive_weights={}, gate_bias=0.0, rank=1)
    loop = EcologyLoop(
        config=cfg,
        host=host,
        environment=SimpleNamespace(),
        population=pop,
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )
    new_bal, info = loop._maybe_top_up_energy(genome, ledger.energy_balance(org))
    assert new_bal == ledger.energy_balance(org)
    assert info["status"] == "already_sufficient"
