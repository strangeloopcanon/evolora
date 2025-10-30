from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_policy_defaults_present() -> None:
    cfg = EcologyConfig()
    pol = cfg.policy
    assert pol.enabled in (True, False)
    assert isinstance(pol.token_cap, int)
    assert isinstance(pol.energy_cost, float)
    assert pol.energy_cost >= 0.0
    assert isinstance(pol.charge_tokens, bool)
    assert isinstance(pol.allowed_fields, list)
    assert isinstance(pol.bias_strength, float)
    assert isinstance(pol.reserve_min, float)
    assert isinstance(pol.reserve_max, float)


def test_policy_parse_helper() -> None:
    allowed = ["cell_pref", "budget_frac", "reserve_ratio"]
    text = '{"cell_pref": {"family": "word.count", "depth": "short"}, "budget_frac": 1.5, "ignored": 123}'
    parsed = EcologyLoop._parse_policy_json(text, allowed)
    assert "cell_pref" in parsed and "budget_frac" in parsed and "ignored" not in parsed


def test_policy_parse_helper_handles_invalid() -> None:
    allowed = ["cell_pref"]
    # Missing braces should yield empty dict
    text = 'cell_pref: {"family": "word.count"}'
    assert EcologyLoop._parse_policy_json(text, allowed) == {}
