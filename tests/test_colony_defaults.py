from symbiont_ecology.config import EcologyConfig


def test_colony_defaults_present():
    cfg = EcologyConfig()
    t = cfg.assimilation_tuning
    assert hasattr(t, "colony_min_size") and isinstance(t.colony_min_size, int)
    assert hasattr(t, "colony_max_size") and isinstance(t.colony_max_size, int)
    assert hasattr(t, "colony_bandwidth_frac") and 0.0 <= t.colony_bandwidth_frac <= 1.0
    assert hasattr(t, "colony_post_cap") and isinstance(t.colony_post_cap, int)
    assert hasattr(t, "colony_read_cap") and isinstance(t.colony_read_cap, int)
    assert hasattr(t, "window_autotune") and isinstance(t.window_autotune, bool)
    assert hasattr(t, "min_window_min") and isinstance(t.min_window_min, int)
    assert hasattr(t, "min_window_max") and isinstance(t.min_window_max, int)

