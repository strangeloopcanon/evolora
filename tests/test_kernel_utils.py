from types import SimpleNamespace

from symbiont_ecology import ATPLedger
from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.routing.router import BanditRouter


def test_host_count_tokens_fallback_and_energy_cost():
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    # Force tokenizer fallback
    host.backbone.tokenizer = None  # type: ignore[attr-defined]
    assert host._count_tokens("a b  c") == 3
    # Energy cost uses base + token_cost + rank_penalty
    organelle = SimpleNamespace(adapter=SimpleNamespace(rank=2))
    cost = host._compute_energy_cost("a b", organelle)
    assert cost > 0.0
