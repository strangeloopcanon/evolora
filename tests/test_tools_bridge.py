from symbiont_ecology.environment.bridge import ToolRegistry, EchoTool, CalculatorTool


def test_tool_registry_and_tools():
    reg = ToolRegistry({"echo": EchoTool(), "calc": CalculatorTool()})
    assert reg.call("echo", text="hi") == "hi"
    assert reg.call("calc", expression="1+2*3") == "7"
    try:
        reg.call("missing")
        assert False, "expected KeyError"
    except KeyError:
        pass

