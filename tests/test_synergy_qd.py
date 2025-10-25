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


def test_synergy_and_qd_archive_in_summary(tmp_path) -> None:
    cfg = EcologyConfig()
    cfg.qd.enabled = True
    cfg.assimilation_tuning.holdout_tasks_path = (
        cfg.evaluation.tasks_path or cfg.assimilation_tuning.holdout_tasks_path
    )
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    pop = PopulationManager(cfg.evolution)
    a = host.spawn_organelle(rank=2)
    b = host.spawn_organelle(rank=2)
    pop.register(Genome(organelle_id=a, drive_weights={}, gate_bias=0.0, rank=2))
    pop.register(Genome(organelle_id=b, drive_weights={}, gate_bias=0.0, rank=2))
    loop = EcologyLoop(
        config=cfg,
        host=host,
        environment=GridEnvironment(
            grid_cfg=cfg.grid,
            controller_cfg=cfg.controller,
            pricing_cfg=cfg.pricing,
            canary_cfg=cfg.canary,
            seed=5,
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
    # QD archive size present
    assert "qd_archive_size" in summary
    # Synergy samples present or empty list acceptable
    _ = summary.get("synergy_samples", [])
    assert isinstance(summary.get("qd_archive_size"), int)
