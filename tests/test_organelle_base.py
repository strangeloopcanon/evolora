from types import SimpleNamespace

from symbiont_ecology.config import HebbianConfig
from symbiont_ecology.interfaces.messages import MessageEnvelope
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.organelles.base import Organelle, OrganelleContext


class DummyOrganelle(Organelle):
    def route_probability(self, observation: MessageEnvelope) -> float:  # noqa: ARG002
        return 0.5

    def forward(self, envelope: MessageEnvelope) -> MessageEnvelope:
        return envelope

    def update(self, envelope: MessageEnvelope, reward: RewardBreakdown) -> None:  # noqa: ARG002
        return None

    def import_adapter_state(self, state, alpha: float = 1.0) -> None:  # noqa: ARG002
        return None


def test_organelle_base_step_and_context() -> None:
    ctx = OrganelleContext(organelle_id="dummy", hebbian=HebbianConfig())
    org = DummyOrganelle("dummy", ctx)
    assert org.steps == 0
    org.step()
    assert org.steps == 1
    envelope = MessageEnvelope.model_construct()
    assert org.forward(envelope) is envelope
    org.update(envelope, RewardBreakdown(task_reward=0.0, novelty_bonus=0.0, competence_bonus=0.0, helper_bonus=0.0, risk_penalty=0.0, cost_penalty=0.0))
    assert org.export_adapter_state() == {}
    assert org.fisher_importance() == 0.0
