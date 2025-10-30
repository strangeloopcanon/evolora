from symbiont_ecology import ATPLedger, BanditRouter, HostKernel, load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import PopulationManager, Genome


def test_colony_bandwidth_limits_reads(tmp_path):
    cfg = load_ecology_config("config/experiments/gemma_simple.yaml")
    cfg.comms.enabled = True
    cfg.assimilation_tuning.colonies_enabled = True
    cfg.assimilation_tuning.colony_bandwidth_frac = 0.01
    cfg.assimilation_tuning.colony_read_cap = 1
    cfg.assimilation_tuning.colony_post_cap = 0
    ledger = ATPLedger(); router = BanditRouter(); host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=7,
                          reward_bonus=cfg.environment.success_reward_bonus,
                          failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
                          lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5))
    pop = PopulationManager(cfg.evolution)
    oid = host.spawn_organelle(rank=cfg.host.max_lora_rank)
    pop.register(Genome(organelle_id=oid, drive_weights={"novelty": 0.0}, gate_bias=0.0, rank=cfg.host.max_lora_rank))
    assim = AssimilationTester(cfg.evolution.assimilation_threshold, cfg.evolution.assimilation_p_value, 0)
    loop = EcologyLoop(config=cfg, host=host, environment=env, population=pop, assimilation=assim)
    # seed colony meta with this organelle
    loop.colonies["col_test"] = {"members": [oid], "pot": 2.0, "reserve_ratio": 0.25}
    # add two messages to read
    env.post_message("other", "hint1", cost=0.0, ttl=3)
    env.post_message("other", "hint2", cost=0.0, ttl=3)
    # Force read_rate high so attempts>0
    g = pop.population[oid]
    g.post_rate = 0.0
    g.read_rate = 1.0
    # Run a tiny generation to trigger comms read step
    loop.run_generation(batch_size=1)
    meta = loop.colonies["col_test"]
    assert int(meta.get("reads_left", 0)) <= cfg.assimilation_tuning.colony_read_cap
    # Bandwidth should have been debited relative to initial pot
    assert float(meta.get("bandwidth_left", 0.0)) <= cfg.assimilation_tuning.colony_bandwidth_frac * 2.0
