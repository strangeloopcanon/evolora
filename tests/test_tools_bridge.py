from symbiont_ecology.environment.bridge import CalculatorTool, EchoTool, ToolRegistry


def test_tool_registry_and_tools():
    reg = ToolRegistry({"echo": EchoTool(), "calc": CalculatorTool()})
    assert reg.call("echo", text="hi") == "hi"
    assert reg.call("calc", expression="1+2*3") == "7"
    try:
        reg.call("missing")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_calculator_rejects_code_execution_payload():
    calc = CalculatorTool()
    result = calc(expression="__import__('os').system('echo injected')")
    assert result.startswith("error:")
