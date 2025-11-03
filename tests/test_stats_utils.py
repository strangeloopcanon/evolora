from symbiont_ecology.environment.loops import EcologyLoop


def test_compute_mean_ci_basic():
    series = [1.0, 2.0, 3.0, 4.0]
    lo, hi, mu, se = EcologyLoop._compute_mean_ci(series)  # type: ignore[attr-defined]
    assert 2.0 <= mu <= 3.0
    assert hi > mu > lo
    # adding a repeated constant lowers SE
    series2 = series + [mu, mu, mu, mu]
    lo2, hi2, mu2, se2 = EcologyLoop._compute_mean_ci(series2)
    assert se2 <= se and abs(mu2 - mu) < 1e-6

