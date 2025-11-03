from symbiont_ecology.config import EcologyConfig


def test_assimilation_tuning_new_team_fields_present():
    cfg = EcologyConfig()
    at = cfg.assimilation_tuning
    assert hasattr(at, "team_probe_per_gen") and isinstance(at.team_probe_per_gen, int)
    assert hasattr(at, "team_min_tasks") and isinstance(at.team_min_tasks, int)
    assert hasattr(at, "team_routing_probe_per_gen") and isinstance(at.team_routing_probe_per_gen, int)

