from symbiont_ecology.environment.loops import EcologyLoop


def test_compute_mean_ci_and_power_proxy_basic():
    # mean 1.0 with small variance -> tight CI and high power vs baseline 0.9
    series = [1.0, 1.0, 1.0, 1.0]
    lo, hi, mu, se = EcologyLoop._compute_mean_ci(series)
    assert 0.9 < lo <= mu <= hi
    power = EcologyLoop._power_proxy(mu, baseline=0.9, margin=0.01, se=se)
    assert 0.0 <= power <= 1.0

    # zero-length series -> zeros and infinite SE surrogate
    lo2, hi2, mu2, se2 = EcologyLoop._compute_mean_ci([])
    assert (lo2, hi2, mu2) == (0.0, 0.0, 0.0)
    assert se2 > 1e6 or se2 == float("inf")
