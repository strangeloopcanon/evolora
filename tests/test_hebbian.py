import torch

from symbiont_ecology.config import HebbianConfig
from symbiont_ecology.interfaces.messages import Intent, MessageEnvelope, Observation, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.organelles.base import OrganelleContext
from symbiont_ecology.organelles.hebbian import HebbianLoRAOrganelle


def _make_envelope(latent_dim: int) -> MessageEnvelope:
    latent = torch.linspace(0, 1, latent_dim, dtype=torch.float32).tolist()
    observation = Observation(state={"latent": latent, "text": "Add 1 and 2."})
    intent = Intent(goal="test", constraints=[], energy_budget=1.0)
    plan = Plan(steps=[], confidence=0.1)
    return MessageEnvelope(observation=observation, intent=intent, plan=plan)


def test_hebbian_rewards_adjust_adapter() -> None:
    hebbian = HebbianConfig(learning_rate=1e-2)
    context = OrganelleContext(organelle_id="test", hebbian=hebbian)
    organelle = HebbianLoRAOrganelle(
        input_dim=16,
        rank=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        context=context,
    )
    envelope = _make_envelope(16)
    original = organelle.adapter.lora_A.clone()
    envelope = organelle.forward(envelope)
    assert envelope.observation.state.get("answer") == "3"
    organelle.update(
        envelope,
        RewardBreakdown(
            task_reward=0.3,
            novelty_bonus=0.1,
            competence_bonus=0.2,
            helper_bonus=0.0,
            risk_penalty=0.0,
            cost_penalty=0.0,
        ),
    )
    assert not torch.equal(original, organelle.adapter.lora_A)
