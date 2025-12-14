import json
from types import SimpleNamespace

from symbiont_ecology import ATPLedger, EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import Genome, PopulationManager


def test_policy_attempts_and_parsed_counters_increment() -> None:
    cfg = EcologyConfig()
    cfg.policy.enabled = True
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    oid = "org_test"
    pop.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2))

    # Fake host returns a valid JSON policy
    ledger = ATPLedger()
    ledger.ensure_energy(oid, 1.0)

    def fake_step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids: list[str]):  # type: ignore[override]
        answer = json.dumps({"budget_frac": 1.1})
        metrics = SimpleNamespace(tokens=16)
        envelope = SimpleNamespace(observation=SimpleNamespace(state={"answer": answer}))
        return SimpleNamespace(envelope=envelope, responses={oid: metrics})

    host = SimpleNamespace(step=fake_step, ledger=ledger)

    loop = EcologyLoop(
        config=cfg,
        host=host,  # type: ignore[arg-type]
        environment=SimpleNamespace(),  # not used in policy method
        population=pop,
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    # Reset counters explicitly
    loop._policy_attempts_gen = 0
    loop._policy_parsed_gen = 0
    loop._request_and_apply_policy(oid)
    assert loop._policy_attempts_gen >= 1
    assert loop._policy_parsed_gen >= 1
