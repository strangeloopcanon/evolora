from symbiont_ecology.environment.loops import EcologyLoop


def test_sanitize_telemetry_handles_nested():
    data = {
        "a": 1,
        "b": True,
        "c": {"x": 3.14, "y": [1, 2, {"z": None}]},
        "d": object(),
    }
    out = EcologyLoop._sanitize_telemetry(data)  # type: ignore[attr-defined]
    assert isinstance(out, dict)
    assert out["a"] == 1 and out["b"] is True
    assert isinstance(out["c"], dict) and isinstance(out["c"]["y"], list)
    assert isinstance(out["d"], str)

