from symbiont_ecology import ATPLedger, BanditRouter, HostKernel, load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import PopulationManager


def test_window_autotune_reduces_min_window(tmp_path):
    cfg = load_ecology_config("config/experiments/gemma_simple.yaml")
    cfg.assimilation_tuning.window_autotune = True
    cfg.assimilation_tuning.min_window = 10
    cfg.assimilation_tuning.min_window_min = 6
    ledger = ATPLedger(); router = BanditRouter(); host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=1,
                          reward_bonus=cfg.environment.success_reward_bonus,
                          failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
                          lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5))
    pop = PopulationManager(cfg.evolution)
    assim = AssimilationTester(cfg.evolution.assimilation_threshold, cfg.evolution.assimilation_p_value, 0)
    loop = EcologyLoop(config=cfg, host=host, environment=env, population=pop, assimilation=assim)
    before = cfg.assimilation_tuning.min_window
    summary = {"assimilation_gating": {"insufficient_scores": 60}}
    loop._auto_nudge_evidence(summary)
    after = cfg.assimilation_tuning.min_window
    assert after <= before and after >= cfg.assimilation_tuning.min_window_min

