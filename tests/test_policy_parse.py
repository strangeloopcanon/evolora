from symbiont_ecology.environment.loops import EcologyLoop


def test_policy_parse_tolerant_kv_and_json():
    allowed = ["budget_frac", "reserve_ratio", "post"]
    kv = "budget_frac=0.7 reserve_ratio:0.2 post=false"
    out = EcologyLoop._parse_policy_json(kv, allowed)
    assert out == {"budget_frac": 0.7, "reserve_ratio": 0.2, "post": False}

    js = '{"budget_frac":0.5, "reserve_ratio": 0.1}'
    out2 = EcologyLoop._parse_policy_json(js, allowed)
    assert out2 == {"budget_frac": 0.5, "reserve_ratio": 0.1}


def test_policy_parse_percentages():
    allowed = ["budget_frac", "reserve_ratio"]
    text = "budget_frac:60% reserve_ratio=15%"
    out = EcologyLoop._parse_policy_json(text, allowed)
    assert out == {"budget_frac": 0.6, "reserve_ratio": 0.15}


def test_policy_parse_null_string_and_filtering():
    allowed = ["tag", "budget_frac"]
    text = "tag='hello' budget_frac=null extra=1"
    out = EcologyLoop._parse_policy_json(text, allowed)
    assert out == {"tag": "hello", "budget_frac": None}


def test_policy_parse_non_object_json_falls_back_to_empty():
    allowed = ["budget_frac"]
    out = EcologyLoop._parse_policy_json("[1,2,3]", allowed)
    assert out == {}


def test_policy_parse_unclosed_outer_object_returns_empty():
    allowed = ["budget_frac"]
    out = EcologyLoop._parse_policy_json("{", allowed)
    assert out == {}
