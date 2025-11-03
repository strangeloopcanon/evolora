from symbiont_ecology.environment.loops import EcologyLoop


def test_compute_mean_ci_basic_and_alpha_branch():
    # Basic stats
    ci_low, ci_high, mu, se = EcologyLoop._compute_mean_ci([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    assert mu == 2.0
    assert ci_high > mu and ci_low < mu
    # Alpha branch uses z=1.0 when not ~0.05
    ci_low2, ci_high2, mu2, se2 = EcologyLoop._compute_mean_ci([1.0, 2.0, 3.0], alpha=0.10)  # type: ignore[attr-defined]
    assert mu2 == mu and se2 == se
    # Narrower band under z=1.0
    assert (ci_high2 - ci_low2) < (ci_high - ci_low)


def test_compute_mean_ci_empty_series():
    ci_low, ci_high, mu, se = EcologyLoop._compute_mean_ci([])  # type: ignore[attr-defined]
    assert (ci_low, ci_high, mu) == (0.0, 0.0, 0.0)
    assert se == float("inf")
