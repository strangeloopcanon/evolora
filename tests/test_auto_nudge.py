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
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    pop = PopulationManager(cfg.evolution, cfg.foraging)
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
            seed=7,
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


def test_auto_nudge_increases_evidence_on_stall() -> None:
    loop = make_loop()
    t = loop.config.assimilation_tuning
    base_mw = t.min_window
    base_ho = t.holdout_sample_size
    base_cap = t.trial_per_gen_cap
    base_prob = t.trial_probation_gens
    base_bonus = t.energy_topup_roi_bonus
    base_tau = loop.config.controller.tau
    loop.assim_fail_streak = 12
    summary = {
        "assimilation_gating": {"low_power": 3, "uplift_below_threshold": 2, "topup_roi_blocked": 7},
        "promotions": 0,
        "merges": 0,
    }
    loop._auto_nudge_evidence(summary)
    t2 = loop.config.assimilation_tuning
    assert t2.min_window >= base_mw
    assert t2.holdout_sample_size >= base_ho
    assert t2.trial_per_gen_cap >= base_cap
    assert t2.trial_probation_gens >= base_prob
    assert t2.energy_topup_roi_bonus >= base_bonus
    assert loop.config.controller.tau <= base_tau


def test_auto_nudge_relaxes_after_success() -> None:
    loop = make_loop()
    # first, force a nudge to set baselines
    loop.assim_fail_streak = 12
    loop._auto_nudge_evidence({"assimilation_gating": {"low_power": 3}, "promotions": 0, "merges": 0})
    # now simulate a success and ensure values move back toward baseline
    before = (
        loop.config.assimilation_tuning.min_window,
        loop.config.assimilation_tuning.holdout_sample_size,
        loop.config.assimilation_tuning.trial_per_gen_cap,
        loop.config.assimilation_tuning.trial_probation_gens,
        loop.config.assimilation_tuning.trial_stipend,
        loop.config.assimilation_tuning.energy_topup_roi_bonus,
        loop.config.controller.tau,
    )
    loop._auto_nudge_evidence({"assimilation_gating": {}, "promotions": 1, "merges": 0})
    after = (
        loop.config.assimilation_tuning.min_window,
        loop.config.assimilation_tuning.holdout_sample_size,
        loop.config.assimilation_tuning.trial_per_gen_cap,
        loop.config.assimilation_tuning.trial_probation_gens,
        loop.config.assimilation_tuning.trial_stipend,
        loop.config.assimilation_tuning.energy_topup_roi_bonus,
        loop.config.controller.tau,
    )
    # One or more parameters should have adjusted towards baseline (not equal to before)
    assert after != before
