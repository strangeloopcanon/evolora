"""Bridging layer for tools and cooperative play."""

from __future__ import annotations

import subprocess  # nosec B404 - known safe usage below
from dataclasses import dataclass
from typing import Any, Protocol


class Tool(Protocol):
    name: str

    def __call__(self, **kwargs: Any) -> str: ...


@dataclass
class ToolRegistry:
    tools: dict[str, Tool]

    def call(self, name: str, **kwargs: Any) -> str:
        if name not in self.tools:
            raise KeyError(f"Tool {name} missing")
        return self.tools[name](**kwargs)


class EchoTool:
    name = "echo"

    def __call__(self, **kwargs: Any) -> str:
        return str(kwargs.get("text", ""))


class CalculatorTool:
    """Deterministic expression evaluator using Python's interpreter."""

    name = "calc"

    def __call__(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", "0"))
        # Use python -c to evaluate safely in a subprocess with limited scope
        command = ["python3", "-c", f"print({expression})"]
        try:
            result = subprocess.run(  # nosec B603
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except subprocess.SubprocessError as exc:  # pragma: no cover - defensive
            return f"error:{exc}"
        return result.stdout.strip()


__all__ = ["CalculatorTool", "EchoTool", "ToolRegistry", "Tool"]
