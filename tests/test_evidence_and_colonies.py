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
    pop = PopulationManager(cfg.evolution, cfg.foraging)
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
            {
                "a": a,
                "b": b,
                "synergy": 0.2,
                "team": 1.20,
                "solo_a": 0.9,
                "solo_b": 0.7,
                "variance_ratio": 0.4,
                "team_var": 0.05,
                "solo_var_min": 0.2,
            },
            {
                "a": a,
                "b": b,
                "synergy": 0.17,
                "team": 1.21,
                "solo_a": 0.8,
                "solo_b": 0.95,
                "variance_ratio": 0.5,
                "team_var": 0.06,
                "solo_var_min": 0.22,
            },
            {
                "a": a,
                "b": b,
                "synergy": 0.19,
                "team": 1.19,
                "solo_a": 1.05,
                "solo_b": 0.85,
                "variance_ratio": 0.45,
                "team_var": 0.05,
                "solo_var_min": 0.2,
            },
        ]
    )
    loop._maybe_promote_colonies()
    assert loop.colonies, "Expected a colony to be created"
    cid = next(iter(loop.colonies.keys()))
    meta = loop.colonies[cid]
    meta["holdout_passes"] = meta.get("required_passes", 2)
    meta["review_interval"] = 999
    # seed positive energy deltas for members
    loop.population.record_energy_delta(a, 0.5)
    loop.population.record_energy_delta(b, 0.4)
    pot_before = float(loop.colonies[cid].get("pot", 0.0))
    loop._tick_colonies()
    pot_after = float(loop.colonies[cid].get("pot", 0.0))
    assert pot_after >= pot_before


def test_colony_dissolves_on_holdout_failure() -> None:
    loop = make_loop(colonies=True)
    ids = list(loop.population.population.keys())
    a, b = ids[0], ids[1]
    loop._synergy_window.extend(
        [
            {
                "a": a,
                "b": b,
                "synergy": 0.2,
                "team": 1.22,
                "solo_a": 0.85,
                "solo_b": 0.75,
                "variance_ratio": 0.4,
                "team_var": 0.05,
                "solo_var_min": 0.2,
            },
            {
                "a": a,
                "b": b,
                "synergy": 0.18,
                "team": 1.21,
                "solo_a": 0.8,
                "solo_b": 1.0,
                "variance_ratio": 0.5,
                "team_var": 0.06,
                "solo_var_min": 0.24,
            },
            {
                "a": a,
                "b": b,
                "synergy": 0.19,
                "team": 1.20,
                "solo_a": 1.05,
                "solo_b": 0.9,
                "variance_ratio": 0.45,
                "team_var": 0.05,
                "solo_var_min": 0.22,
            },
        ]
    )
    loop._maybe_promote_colonies()
    cid = next(iter(loop.colonies.keys()))
    meta = loop.colonies[cid]
    meta["holdout_passes"] = 0
    meta["holdout_failures"] = 0
    meta["review_interval"] = 1
    meta["last_review"] = loop.generation_index - 2

    def failing_stats(member_ids, _tasks):
        if len(member_ids) > 1:
            return {"mean": 0.05, "variance": 0.3, "series": [0.05]}
        return {"mean": 0.2, "variance": 0.05, "series": [0.2]}

    loop._team_holdout_stats = failing_stats  # type: ignore[assignment]
    loop._sample_holdout_tasks = lambda: ["dummy"] * 8  # type: ignore
    loop._tick_colonies()
    assert cid in loop.colonies
    meta = loop.colonies[cid]
    meta["last_review"] = loop.generation_index - 2
    loop._tick_colonies()
    assert cid not in loop.colonies
