from symbiont_ecology.evolution.assimilation import AssimilationTester


def test_assimilation_ztest_includes_ci_and_power():
    tester = AssimilationTester(uplift_threshold=0.0, p_value_threshold=0.25, safety_budget=0)
    control = [0.0, 0.0, 0.1, -0.1, 0.05, -0.05]
    treatment = [c + 0.1 for c in control]
    result = tester.evaluate(
        organelle_id="org_test",
        control_scores=control,
        treatment_scores=treatment,
        safety_hits=0,
        energy_cost=0.0,
    )
    ev = result.event
    assert ev.ci_low is not None and ev.ci_high is not None
    assert ev.ci_low <= ev.uplift <= ev.ci_high
    assert ev.power is None or (0.0 <= ev.power <= 1.0)


def test_assimilation_bootstrap_includes_ci_and_power():
    tester = AssimilationTester(uplift_threshold=0.0, p_value_threshold=0.25, safety_budget=0)
    tester.bootstrap_enabled = True
    tester.bootstrap_n = 30
    tester.permutation_n = 30
    control = [0.0, 0.0, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02]
    treatment = [c + 0.08 for c in control]
    result = tester.evaluate(
        organelle_id="org_test",
        control_scores=control,
        treatment_scores=treatment,
        safety_hits=0,
        energy_cost=0.0,
    )
    ev = result.event
    assert ev.ci_low is not None and ev.ci_high is not None
    assert ev.ci_low <= ev.uplift <= ev.ci_high
    # power is optional but if present should be bounded
    assert ev.power is None or (0.0 <= ev.power <= 1.0)
