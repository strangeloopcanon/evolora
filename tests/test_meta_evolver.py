from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.meta import MetaEvolver


def test_meta_evolver_mutate_and_accept() -> None:
    config = EcologyConfig()
    config.meta.enabled = True
    config.meta.interval = 1
    config.meta.mutation_scale = 0.05
    config.meta.catastrophe_interval = 0
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
    )
    assimilator = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )
    evolver = MetaEvolver(config=config, environment=environment, assimilation=assimilator)

    info = evolver.step(generation=1, avg_roi=0.1)
    assert "meta_mutation" in info
    assert evolver.awaiting_eval

    info = evolver.step(generation=2, avg_roi=0.2)
    assert info.get("meta_accept") is True
    # interval=1 triggers a fresh mutation immediately after acceptance
    if "meta_mutation" not in info:
        info = evolver.step(generation=3, avg_roi=0.2)
    assert "meta_mutation" in info
    info = evolver.step(generation=4, avg_roi=0.01)
    assert info.get("meta_revert") is True


def test_meta_evolver_catastrophe() -> None:
    config = EcologyConfig()
    config.meta.enabled = True
    config.meta.interval = 10
    config.meta.catastrophe_interval = 2
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=123,
    )
    assimilator = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )
    evolver = MetaEvolver(config=config, environment=environment, assimilation=assimilator)
    baseline = {key: state.difficulty for key, state in environment.controller.cells.items()}
    info = evolver.step(generation=2, avg_roi=0.1)
    if config.meta.catastrophe_interval:
        assert info.get("catastrophe") is True
    post = {key: state.difficulty for key, state in environment.controller.cells.items()}
    assert baseline != post
