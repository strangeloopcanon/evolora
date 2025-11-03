from symbiont_ecology.environment.loops import EcologyLoop


def test_policy_parse_outer_object_repairs_and_extracts() -> None:
    # Outer JSON object without code fences; single quotes and trailing comma
    text = """
    Some preface...
    { 'budget_frac': 0.5, 'reserve_ratio': 0.2, }
    Some suffix...
    """
    allowed = ["budget_frac", "reserve_ratio", "post"]
    parsed = EcologyLoop._parse_policy_json(text, allowed)
    assert isinstance(parsed, dict)
    assert parsed.get("budget_frac") == 0.5
    assert parsed.get("reserve_ratio") == 0.2
    # Unknown fields are dropped
    assert "post" not in parsed


def test_policy_parse_prefers_fenced_json() -> None:
    text = """
    ```json
    {"budget_frac": 0.75, "reserve_ratio": 0.1}
    ```
    """
    allowed = ["budget_frac", "reserve_ratio"]
    parsed = EcologyLoop._parse_policy_json(text, allowed)
    assert parsed == {"budget_frac": 0.75, "reserve_ratio": 0.1}


def test_policy_parse_returns_empty_on_no_json() -> None:
    text = "no json here"
    allowed = ["budget_frac", "reserve_ratio"]
    parsed = EcologyLoop._parse_policy_json(text, allowed)
    assert parsed == {}
