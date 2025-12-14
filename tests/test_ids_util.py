from symbiont_ecology.utils.ids import short_uid


def test_short_uid_prefix_and_uniqueness():
    a = short_uid("org")
    b = short_uid("org")
    assert a.startswith("org_") and b.startswith("org_") and a != b
