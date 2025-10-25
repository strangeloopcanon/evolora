from statistics import mean

from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    PopulationManager,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def test_auto_tune_assimilation_energy_populates_summary() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    pop = PopulationManager(cfg.evolution)
    oid = host.spawn_organelle(rank=2)
    pop.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2))
    loop = EcologyLoop(
        config=cfg,
        host=host,
        environment=GridEnvironment(
            grid_cfg=cfg.grid,
            controller_cfg=cfg.controller,
            pricing_cfg=cfg.pricing,
            canary_cfg=cfg.canary,
            seed=11,
        ),
        population=pop,
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    # seed recent costs/rois so the tuner has data
    pop.energy[oid] = [0.9, 1.1, 1.0, 0.95]
    pop.roi[oid] = [0.8, 1.1, 1.3, 0.9]
    summary = {"merges": 0, "avg_roi": float(mean(pop.roi[oid]))}
    loop._auto_tune_assimilation_energy(summary)
    tuning = summary.get("assimilation_energy_tuning")
    assert isinstance(tuning, dict)
    assert "energy_floor" in tuning and "energy_floor_roi" in tuning
