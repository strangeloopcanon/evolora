from symbiont_ecology.config import GridConfig, ControllerConfig, PricingConfig, CanaryConfig
from symbiont_ecology.environment.grid import EnvironmentController, GridEnvironment


def test_environment_controller_apply_parameters_updates_prices():
    grid_cfg = GridConfig(families=["word.count"], depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.1, max=10.0)
    canary_cfg = CanaryConfig(q_min=0.8)
    envc = EnvironmentController(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, lp_alpha=0.5, seed=42)
    # initial price
    cell = ("word.count", "short")
    before = envc.cells[cell].price
    envc.apply_parameters(price_base=2.0, price_k=0.5)
    after = envc.cells[cell].price
    assert after != before
    # adjusting tau should also work
    envc.apply_parameters(tau=0.6)
    assert envc.ctrl.tau == 0.6


def test_environment_controller_sampling_and_messages():
    grid_cfg = GridConfig(families=["word.count", "math"], depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.1, max=10.0)
    canary_cfg = CanaryConfig(q_min=0.8)
    envc = EnvironmentController(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, lp_alpha=0.5, seed=123)
    # First two calls should explore each cell at least once
    a = envc.sample_cell()
    b = envc.sample_cell()
    assert a in envc.cells and b in envc.cells
    # Message board TTL behavior
    env = GridEnvironment(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, seed=999)
    env.message_board.clear()
    assert env.post_message("org_x", "hello", ttl=2)
    msgs = env.read_messages(max_items=1)
    assert msgs and msgs[0]["text"] == "hello"
    # After two more reads, message should expire
    env.read_messages(max_items=1)
    env.read_messages(max_items=1)
    assert env.read_messages(max_items=1) == []
