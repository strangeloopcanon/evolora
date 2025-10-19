from symbiont_ecology import ATPLedger, BanditRouter, EcologyConfig, HostKernel
from symbiont_ecology.interfaces.messages import MessageEnvelope, Observation, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown


def test_apply_reward_mints_total_without_double_cost() -> None:
    config = EcologyConfig()
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    org_id = host.spawn_organelle(rank=2)

    # Build a minimal envelope
    obs = Observation(state={"text": "2+2"})
    env = MessageEnvelope(observation=obs, intent=host.router.intent_factory("test", []), plan=Plan(steps=[], confidence=0.1))

    before = host.ledger.accounts[org_id].balance
    reward = RewardBreakdown(
        task_reward=1.0,
        novelty_bonus=0.0,
        competence_bonus=0.0,
        helper_bonus=0.0,
        risk_penalty=0.0,
        cost_penalty=0.2,
    )
    host.apply_reward(env, {org_id: reward})
    after = host.ledger.accounts[org_id].balance

    # total = 1.0 - 0.2 = 0.8; minting should add exactly total (not subtract cost twice)
    assert abs((after - before) - (reward.total)) < 1e-6

