from symbiont_ecology import ATPLedger, BanditRouter, EcologyConfig, HostKernel
from symbiont_ecology.metrics.telemetry import RewardBreakdown


def test_host_spawn_and_step_creates_plan() -> None:
    config = EcologyConfig()
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()
    organelle_id = host.spawn_organelle(rank=2)
    result = host.step(prompt="Add 2 and 3.", intent="solve", max_routes=1)
    assert organelle_id in host.organelles
    assert result.envelope.plan is not None
    metrics = result.responses.get(organelle_id)
    assert metrics is not None
    assert metrics.answer.strip().startswith("5")
    assert metrics.tokens > 0
    rewards = {
        organelle_id: RewardBreakdown(
            task_reward=1.0,
            novelty_bonus=0.1,
            competence_bonus=0.1,
            helper_bonus=0.0,
            risk_penalty=0.0,
            cost_penalty=0.0,
        ),
    }
    host.apply_reward(result.envelope, rewards)
    assert ledger.accounts[organelle_id].balance > config.organism.initial_atp


def test_host_assimilation_seed_applies(monkeypatch) -> None:
    config = EcologyConfig()
    config.assimilation_tuning.seed_scale = 0.5
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    seed_id = host.spawn_organelle(rank=1)
    seed_organelle = host.get_organelle(seed_id)
    state = seed_organelle.export_adapter_state()
    host.assimilation_state = {key: tensor.clone() for key, tensor in state.items()}
    host.assimilation_weights = {key: 1.0 for key in state}
    host.retire_organelle(seed_id)

    from symbiont_ecology.organelles.hebbian import HebbianLoRAOrganelle

    original_import = HebbianLoRAOrganelle.import_adapter_state
    call_counter = {"count": 0}

    def wrapped(self, state_dict, alpha=1.0):
        call_counter["count"] += 1
        return original_import(self, state_dict, alpha)

    monkeypatch.setattr(HebbianLoRAOrganelle, "import_adapter_state", wrapped)

    new_id = host.spawn_organelle(rank=1)
    assert new_id in host.organelles
    assert call_counter["count"] >= 1
