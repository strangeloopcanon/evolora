from symbiont_ecology import ATPLedger, BanditRouter, HostKernel, load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import PopulationManager, Genome


def test_colony_bandwidth_limits_reads(tmp_path):
    cfg = load_ecology_config("config/experiments/gemma_simple.yaml")
    cfg.comms.enabled = True
    cfg.assimilation_tuning.colonies_enabled = True
    cfg.assimilation_tuning.colony_bandwidth_base = 1.0
    cfg.assimilation_tuning.colony_bandwidth_frac = 0.5
    cfg.assimilation_tuning.colony_hazard_bandwidth_scale = 0.25
    cfg.assimilation_tuning.colony_read_cap = 2
    cfg.assimilation_tuning.colony_post_cap = 1
    cfg.assimilation_tuning.colony_post_cap_hazard = 0
    cfg.assimilation_tuning.colony_read_cap_hazard = 1
    ledger = ATPLedger(); router = BanditRouter(); host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=7,
                          reward_bonus=cfg.environment.success_reward_bonus,
                          failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
                          lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5))
    pop = PopulationManager(cfg.evolution, cfg.foraging)
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
    expected_budget = min(
        cfg.assimilation_tuning.colony_bandwidth_base,
        2.0 * cfg.assimilation_tuning.colony_bandwidth_frac,
    )
    assert abs(meta.get("bandwidth_budget", 0.0) - expected_budget) < 1e-6
    assert meta.get("posts_budget") == cfg.assimilation_tuning.colony_post_cap
    assert meta.get("reads_budget") == cfg.assimilation_tuning.colony_read_cap
    assert int(meta.get("reads_left", 0)) <= cfg.assimilation_tuning.colony_read_cap


def test_colony_bandwidth_hazard_scaling():
    cfg = load_ecology_config("config/experiments/gemma_simple.yaml")
    cfg.comms.enabled = True
    cfg.assimilation_tuning.colonies_enabled = True
    cfg.assimilation_tuning.colony_bandwidth_base = 2.0
    cfg.assimilation_tuning.colony_bandwidth_frac = 0.5
    cfg.assimilation_tuning.colony_hazard_bandwidth_scale = 0.1
    cfg.assimilation_tuning.colony_post_cap = 3
    cfg.assimilation_tuning.colony_read_cap = 4
    cfg.assimilation_tuning.colony_post_cap_hazard = 1
    cfg.assimilation_tuning.colony_read_cap_hazard = 1
    ledger = ATPLedger(); router = BanditRouter(); host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=13,
                          reward_bonus=cfg.environment.success_reward_bonus,
                          failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
                          lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5))
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    oid = host.spawn_organelle(rank=cfg.host.max_lora_rank)
    pop.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=cfg.host.max_lora_rank))
    assim = AssimilationTester(cfg.evolution.assimilation_threshold, cfg.evolution.assimilation_p_value, 0)
    loop = EcologyLoop(config=cfg, host=host, environment=env, population=pop, assimilation=assim)
    loop.colonies["col_hazard"] = {"members": [oid], "pot": 5.0, "reserve_ratio": 0.25}
    loop._hazard_state[oid] = {"active": True, "z": -1.0}
    env.post_message("other", "hazard hint", cost=0.0, ttl=2)
    g = pop.population[oid]
    g.read_rate = 1.0
    loop.run_generation(batch_size=1)
    meta = loop.colonies["col_hazard"]
    expected_budget = min(cfg.assimilation_tuning.colony_bandwidth_base, 5.0 * cfg.assimilation_tuning.colony_bandwidth_frac) * cfg.assimilation_tuning.colony_hazard_bandwidth_scale
    assert abs(meta.get("bandwidth_budget", 0.0) - expected_budget) < 1e-6
    assert meta.get("posts_budget") == cfg.assimilation_tuning.colony_post_cap_hazard
    assert meta.get("reads_budget") == cfg.assimilation_tuning.colony_read_cap_hazard
    assert meta.get("hazard_members") == 1
