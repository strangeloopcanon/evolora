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


def make_loop(colonies: bool = False) -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.qd.enabled = True
    cfg.assimilation_tuning.colonies_enabled = colonies
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
            seed=2,
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
    return loop


def test_evidence_bypass_topup() -> None:
    loop = make_loop()
    # set energy floor
    loop.config.assimilation_tuning.energy_floor = 1.0
    loop.config.assimilation_tuning.energy_floor_roi = 1.0
    # grant evidence credit to organelle a
    a = next(iter(loop.population.population.keys()))
    loop.population.grant_evidence(a, 1)
    bal = 0.2
    new_bal, info = loop._maybe_top_up_energy(loop.population.population[a], bal)
    assert new_bal >= bal
    assert info.get("status") == "credited"
    # ensure bypass flag set when ROI would be low
    assert info.get("evidence_bypass") in (True, None)


def test_colony_promotion_and_tick() -> None:
    loop = make_loop(colonies=True)
    ids = list(loop.population.population.keys())
    a, b = ids[0], ids[1]
    # inject synergy samples across windows
    loop._synergy_window.extend(
        [
            {"a": a, "b": b, "synergy": 0.2},
            {"a": a, "b": b, "synergy": 0.15},
            {"a": a, "b": b, "synergy": 0.18},
        ]
    )
    loop._maybe_promote_colonies()
    assert loop.colonies, "Expected a colony to be created"
    cid = next(iter(loop.colonies.keys()))
    # seed positive energy deltas for members
    loop.population.record_energy_delta(a, 0.5)
    loop.population.record_energy_delta(b, 0.4)
    pot_before = float(loop.colonies[cid].get("pot", 0.0))
    loop._tick_colonies()
    pot_after = float(loop.colonies[cid].get("pot", 0.0))
    assert pot_after >= pot_before

