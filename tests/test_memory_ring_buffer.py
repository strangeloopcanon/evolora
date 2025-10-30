from symbiont_ecology.metrics.telemetry import EpisodeLog, RewardBreakdown
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology import ATPLedger, BanditRouter, HostKernel, load_ecology_config
from symbiont_ecology.environment.grid import GridEnvironment, GridTask


def test_in_memory_episode_ring_buffer(tmp_path):
    cfg = load_ecology_config("config/experiments/gemma_simple.yaml")
    cfg.metrics.in_memory_log_limit = 10
    ledger = ATPLedger(); router = BanditRouter(); host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host();
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=1,
                          reward_bonus=cfg.environment.success_reward_bonus,
                          failure_cost_multiplier=cfg.environment.failure_cost_multiplier,
                          lp_alpha=getattr(cfg.curriculum, "lp_alpha", 0.5))
    from symbiont_ecology.evolution.population import PopulationManager, Genome
    from symbiont_ecology.evolution.assimilation import AssimilationTester
    pop = PopulationManager(cfg.evolution)
    oid = host.spawn_organelle(rank=cfg.host.max_lora_rank)
    pop.register(Genome(organelle_id=oid, drive_weights={"novelty": 0.0}, gate_bias=0.0, rank=cfg.host.max_lora_rank))
    assim = AssimilationTester(cfg.evolution.assimilation_threshold, cfg.evolution.assimilation_p_value, 0)
    loop = EcologyLoop(config=cfg, host=host, environment=env, population=pop, assimilation=assim)

    # Push >limit episodes via the internal recording path
    task = GridTask(task_id="t1", cell=("word.count", "short"), prompt="Count words: 'a b c'", price=1.0,
                    target=3, family="word.count", depth="short", difficulty=0.1)
    for i in range(25):
        rb = RewardBreakdown(task_reward=0.0, novelty_bonus=0.0, competence_bonus=0.0, helper_bonus=0.0, risk_penalty=0.0, cost_penalty=0.0)
        loop._record_episode(task, oid, rb, host.step(prompt=task.prompt, intent="diag", max_routes=1, allowed_organelle_ids=[oid]).responses[oid],
                             {"energy_before":0.0,"energy_after":0.0,"revenue":0.0,"cost":0.0,"roi":0.0,"delta":0.0},
                             False, {})
    assert len(loop.logs) <= 10

