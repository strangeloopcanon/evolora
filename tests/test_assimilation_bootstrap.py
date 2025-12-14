from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.metrics.telemetry import EnergyTopUpEvent


def test_assimilation_update_thresholds() -> None:
    tester = AssimilationTester(uplift_threshold=0.1, p_value_threshold=0.05, safety_budget=3)
    tester.update_thresholds(uplift_threshold=0.2)
    assert tester.uplift_threshold == 0.2
    tester.update_thresholds(p_value_threshold=0.1)
    assert tester.p_value_threshold == 0.1


def test_assimilation_bootstrap_populates_ci_and_energy_topup() -> None:
    tester = AssimilationTester(uplift_threshold=0.1, p_value_threshold=0.05, safety_budget=10)
    tester.bootstrap_enabled = True
    tester.bootstrap_n = 25
    tester.permutation_n = 25
    result = tester.evaluate(
        organelle_id="org",
        control_scores=[0.0, 0.0, 0.0, 0.0],
        treatment_scores=[0.2, 0.2, 0.2, 0.2],
        safety_hits=0,
        energy_cost=0.1,
        energy_top_up={"status": "ok", "before": 1.0, "after": 2.0},
    )
    assert result.event.method.endswith("+bootstrap")
    assert result.event.ci_low is not None
    assert result.event.ci_high is not None
    assert result.event.energy_top_up is not None
    assert isinstance(result.event.energy_top_up, EnergyTopUpEvent)


def test_assimilation_returns_early_when_insufficient_samples() -> None:
    tester = AssimilationTester(uplift_threshold=0.1, p_value_threshold=0.05, safety_budget=10)
    tester.min_samples = 3
    result = tester.evaluate(
        organelle_id="org",
        control_scores=[0.0, 0.0],
        treatment_scores=[0.2, 0.2],
        safety_hits=0,
        energy_cost=0.1,
        energy_top_up={"status": "ok", "before": 1.0, "after": 2.0},
    )
    assert result.decision is False
    assert result.event.sample_size == 2
    assert result.event.energy_top_up is not None
    assert isinstance(result.event.energy_top_up, EnergyTopUpEvent)


def test_assimilation_truncates_mismatched_lengths_and_meta() -> None:
    tester = AssimilationTester(uplift_threshold=0.0, p_value_threshold=1.0, safety_budget=10)
    control_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
    treatment_scores = [0.1, 0.1, 0.1]
    control_meta = [{"family": "a"}] * len(control_scores)
    treatment_meta = [{"family": "a"}] * len(treatment_scores)
    result = tester.evaluate(
        organelle_id="org",
        control_scores=control_scores,
        treatment_scores=treatment_scores,
        safety_hits=0,
        energy_cost=0.1,
        control_meta=control_meta,
        treatment_meta=treatment_meta,
    )
    assert result.event.sample_size == len(treatment_scores)
