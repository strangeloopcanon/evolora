from symbiont_ecology.environment.loops import EcologyLoop


def test_policy_parse_kv_pairs_fallback():
    text = "budget_frac=0.7 reserve_ratio:0.2 post=false"
    allowed = ["budget_frac", "reserve_ratio", "post"]
    out = EcologyLoop._parse_policy_json(text, allowed)
    assert out.get("budget_frac") == 0.7
    assert out.get("reserve_ratio") == 0.2
    assert out.get("post") is False

