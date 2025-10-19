"""Tests for evaluation manager reward shaping."""

import pytest

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.evaluation.manager import (
    EvaluationConfigRuntime,
    EvaluationManager,
    EvaluationTask,
)
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.routing.router import BanditRouter


@pytest.fixture()
def host_and_environment():
    config = EcologyConfig()
    config.energy.alpha = 1e-14
    config.energy.beta = 0.02
    config.energy.gamma = 0.001
    config.energy.lambda_p = 1e-7
    config.host.max_lora_rank = 2
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()
    organelle_id = host.spawn_organelle(rank=1)
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
    )
    return config, host, environment, organelle_id


def _make_manager(tasks, reward_weight=0.5):
    runtime = EvaluationConfigRuntime(
        enabled=True,
        cadence=1,
        tasks=list(tasks),
        sample_size=len(tasks),
        reward_weight=reward_weight,
    )
    return EvaluationManager(runtime, seed=42)


def test_evaluation_rewards_scale_with_roi(host_and_environment):
    _, host, environment, organelle_id = host_and_environment
    manager = _make_manager(
        [
            EvaluationTask(
                prompt="Add 2 and 3 and return the sum",
                target=5,
                family="math",
                depth="short",
            )
        ],
        reward_weight=0.6,
    )

    start_balance = host.ledger.accounts[organelle_id].balance
    start_energy = host.ledger.energy_balance(organelle_id)

    summary = manager.evaluate(host, environment)

    end_balance = host.ledger.accounts[organelle_id].balance
    end_energy = host.ledger.energy_balance(organelle_id)

    assert summary["accuracy"] == pytest.approx(1.0)
    assert summary["avg_roi"] > 0.0
    assert summary["avg_delta"] > 0.0
    assert end_balance > start_balance
    assert end_energy > start_energy


def test_evaluation_penalizes_failures(host_and_environment):
    config, host, environment, organelle_id = host_and_environment
    manager = _make_manager(
        [
            EvaluationTask(
                prompt="Return 9999 exactly",
                target=9999,
                family="math",
                depth="short",
            )
        ],
        reward_weight=0.8,
    )

    # Make the prompt hard by reducing price so failure is costly.
    cell = ("math", "short")
    environment.controller.cells[cell].price = 0.5

    start_balance = host.ledger.accounts[organelle_id].balance
    start_energy = host.ledger.energy_balance(organelle_id)

    summary = manager.evaluate(host, environment)

    end_balance = host.ledger.accounts[organelle_id].balance
    end_energy = host.ledger.energy_balance(organelle_id)

    assert summary["accuracy"] == pytest.approx(0.0)
    assert summary["avg_roi"] == pytest.approx(0.0)
    assert summary["avg_delta"] <= 0.0
    assert end_balance <= start_balance
    assert end_energy <= start_energy
