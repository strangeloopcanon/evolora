from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    HumanBandit,
    PopulationManager,
    TelemetrySink,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome
from symbiont_ecology.metrics.telemetry import RewardBreakdown


def test_loop_feedback_and_spawn(tmp_path) -> None:
    config = EcologyConfig()
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution, config.foraging)
    organelle_id = host.spawn_organelle(rank=2)
    genome = Genome(
        organelle_id=organelle_id,
        drive_weights={"novelty": 0.2},
        gate_bias=0.0,
        rank=2,
    )
    population.register(genome)
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=24,
    )
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=AssimilationTester(
            uplift_threshold=config.evolution.assimilation_threshold,
            p_value_threshold=config.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=HumanBandit(),
        sink=TelemetrySink(tmp_path, "episodes.jsonl", "assimilation.jsonl"),
    )

    responses = {organelle_id: ("Thanks for helping", 1.0)}
    human_feedback = loop._collect_human_feedback("Add 1 and 2", responses)
    assert human_feedback is not None
    base_rewards = {
        organelle_id: RewardBreakdown(
            task_reward=0.5,
            novelty_bonus=0.0,
            competence_bonus=0.0,
            helper_bonus=0.0,
            risk_penalty=0.0,
            cost_penalty=0.0,
        )
    }
    blended = loop._blend_rewards(base_rewards, human_feedback)
    assert blended[organelle_id].helper_bonus >= 0.0

    original_population = len(population.population)
    loop._spawn_replacement_from(genome)
    assert len(population.population) > original_population
