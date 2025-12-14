from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_request_and_apply_policy_parses_and_charges_energy() -> None:
    cfg = EcologyConfig()
    cfg.policy.enabled = True
    cfg.policy.energy_cost = 0.05
    cfg.policy.charge_tokens = True
    cfg.policy.token_cap = 64
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=SimpleNamespace(population={"orgX": SimpleNamespace(gate_bias=0.0)}),
        assimilation=SimpleNamespace(),
    )

    # Stub host step to return a simple JSON answer and minimal metrics
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
        oid = allowed_organelle_ids[0]
        env = SimpleNamespace(
            observation=SimpleNamespace(
                state={"answer": '{"budget_frac": 1.2, "gate_bias_delta": 0.5}'}
            )
        )
        metrics = SimpleNamespace(tokens=32)
        return SimpleNamespace(envelope=env, responses={oid: metrics})

    charged = {"total": 0.0}

    class Ledger:
        def energy_balance(self, oid: str) -> float:  # noqa: ARG002
            return 10.0

        def consume_energy(self, oid: str, amt: float) -> None:  # noqa: ARG002
            charged["total"] += float(amt)

    loop.host = SimpleNamespace(step=step, ledger=Ledger())

    loop._request_and_apply_policy("orgX")

    # Policy is stored and field parsed/rounded
    pol = loop._active_policies.get("orgX")
    assert isinstance(pol, dict)
    assert "budget_frac" in pol
    # Energy was charged (micro-cost scaled by tokens)
    assert charged["total"] > 0.0
    # Gate bias delta applied to genome
    assert loop.population.population["orgX"].gate_bias != 0.0
