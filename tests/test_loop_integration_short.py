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


def test_run_generation_with_comms_and_lp() -> None:
    cfg = EcologyConfig()
    cfg.comms.enabled = True
    cfg.curriculum.lp_mix = 0.2
    cfg.evaluation.enabled = False
    cfg.assimilation_tuning.per_cell_interval = 1
    cfg.environment.synthetic_batch_size = 1
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    pop = PopulationManager(cfg.evolution)
    oid = host.spawn_organelle(rank=2)
    pop.register(Genome(organelle_id=oid, drive_weights={"novelty": 0.4}, gate_bias=0.0, rank=2))
    loop = EcologyLoop(
        config=cfg,
        host=host,
        environment=GridEnvironment(
            grid_cfg=cfg.grid,
            controller_cfg=cfg.controller,
            pricing_cfg=cfg.pricing,
            canary_cfg=cfg.canary,
            seed=21,
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
    summary = loop.run_generation(batch_size=1)
    assert isinstance(summary, dict)
    # energy tuner should have produced values
    tuning = summary.get("assimilation_energy_tuning")
    assert isinstance(tuning, dict)
    assert "energy_floor" in tuning
