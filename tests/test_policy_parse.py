from symbiont_ecology.environment.loops import EcologyLoop


def test_parse_policy_json_filters_allowed_fields():
    text = (
        "```json\n"
        '{"cell_pref": {"family": "word.count", "depth": "short"}, '
        '"budget_frac": 1.5, "ignore": 1}\n'
        "```"
    )
    allowed = ["cell_pref", "budget_frac"]
    out = EcologyLoop._parse_policy_json(text, allowed)
    assert isinstance(out, dict)
    assert "cell_pref" in out and "budget_frac" in out
    assert "ignore" not in out
    assert isinstance(out["cell_pref"], dict)


def test_parse_policy_json_handles_invalid_gracefully():
    # No braces
    text = "no json here"
    out = EcologyLoop._parse_policy_json(text, ["budget_frac"])
    assert out == {}
    # Non-dict JSON should be rejected
    text = "[1,2,3]"
    out = EcologyLoop._parse_policy_json(text, ["budget_frac"])
    assert out == {}


def test_parse_policy_json_repairs_trailing_commas_and_single_quotes():
    text = (
        "```json\n"
        "{'budget_frac': 1.2, 'read': True,}\n"
        "```"
    )
    allowed = ["budget_frac", "read"]
    out = EcologyLoop._parse_policy_json(text, allowed)
    assert out.get("budget_frac") == 1.2
    assert out.get("read") is True


def test_parse_policy_json_normalizes_python_literals():
    text = "{'reserve_ratio': None, 'post': False}"
    allowed = ["reserve_ratio", "post"]
    out = EcologyLoop._parse_policy_json(text, allowed)
    assert "reserve_ratio" in out and out["reserve_ratio"] is None
    assert out.get("post") is False
