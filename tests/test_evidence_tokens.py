from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


class _Pop:
    def __init__(self, tokens: int):
        self.tokens = tokens

    def evidence_tokens(self, oid: str) -> int:  # noqa: ARG002
        return self.tokens

    def consume_evidence(self, oid: str, amount: int = 1) -> bool:  # noqa: ARG002
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


def test_apply_evidence_tokens_reduces_window():
    cfg = EcologyConfig()
    loop = object.__new__(EcologyLoop)
    loop.config = cfg
    loop.population = _Pop(tokens=2)
    loop._power_econ_stats = {}
    adjusted, used = loop._apply_evidence_tokens(
        "org", available_even=6, min_window=10, min_samples_required=2
    )
    assert used is True
    assert adjusted <= 6
    # No tokens left -> no change
    loop.population.tokens = 0
    adjusted_again, used_again = loop._apply_evidence_tokens(
        "org", available_even=4, min_window=8, min_samples_required=2
    )
    assert used_again is False
    assert adjusted_again == 8


def test_apply_evidence_tokens_uses_override_floor():
    cfg = EcologyConfig()
    loop = object.__new__(EcologyLoop)
    loop.config = cfg
    loop.config.assimilation_tuning.evidence_token_window = 12
    loop.config.assimilation_tuning.min_window_min = 2
    loop.population = _Pop(tokens=3)
    loop._power_econ_stats = {}
    adjusted, used = loop._apply_evidence_tokens(
        "org", available_even=2, min_window=12, min_samples_required=2
    )
    assert used is True
    assert adjusted == 2
