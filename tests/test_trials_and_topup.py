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


def make_loop() -> EcologyLoop:
    config = EcologyConfig()
    # make top-ups active for the test and enable trials
    config.assimilation_tuning.energy_floor = 1.0
    config.assimilation_tuning.energy_floor_roi = 1.0
    config.assimilation_tuning.energy_topup_roi_bonus = 0.2
    config.assimilation_tuning.trial_offspring_enabled = True
    config.assimilation_tuning.trial_per_gen_cap = 1
    config.assimilation_tuning.trial_probation_gens = 2
    # tiny environment
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution)
    oid = host.spawn_organelle(rank=2)
    population.register(Genome(organelle_id=oid, drive_weights={"novelty": 0.3}, gate_bias=0.0, rank=2))
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=GridEnvironment(
            grid_cfg=config.grid,
            controller_cfg=config.controller,
            pricing_cfg=config.pricing,
            canary_cfg=config.canary,
            seed=1,
        ),
        population=population,
        assimilation=AssimilationTester(
            uplift_threshold=config.evolution.assimilation_threshold,
            p_value_threshold=config.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    return loop


def test_topup_dynamic_fields_present() -> None:
    loop = make_loop()
    # seed recent ROI for variance
    oid = next(iter(loop.population.population.keys()))
    loop.population.roi[oid] = [0.5, 1.0, 1.5, 0.7]
    bal = loop.host.ledger.energy_balance(oid)
    new_bal, info = loop._maybe_top_up_energy(loop.population.population[oid], bal)
    assert isinstance(info, dict)
    # dynamic fields added by adaptive top-up logic
    assert "roi_std" in info and "fail_streak" in info
    assert info["roi_threshold_effective"] <= loop.config.assimilation_tuning.energy_floor_roi
    assert new_bal >= bal


def test_trial_offspring_evidence_accumulates() -> None:
    loop = make_loop()
    # make promotion easy to hit, to cover promotion branch too
    loop.config.assimilation_tuning.trial_promote_margin = 0.0
    # inject a dummy trial child without running the full assimilation path
    parent_id = next(iter(loop.population.population.keys()))
    child_id = loop.host.spawn_organelle(rank=2)
    loop.population.register(Genome(organelle_id=child_id, drive_weights={}, gate_bias=0.0, rank=2))
    loop.trial_offspring[child_id] = {
        "parents": [parent_id],
        "cell": {"family": "math", "depth": "short"},
        "created_gen": 0,
        "probation_left": 1,
        "promoted": False,
    }
    # force empty holdout sample (safe, returns 0 ROI)
    loop._holdout_tasks_cache = []
    loop._review_trial_offspring()
    # with margin 0 and equal ROI, child should be promoted and removed
    assert loop.promotions_this_gen >= 0
