"""Bridging layer for tools and cooperative play."""

from __future__ import annotations

import ast
import math
import operator
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
    """Deterministic arithmetic evaluator for trusted math expressions."""

    name = "calc"

    _BIN_OPS: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS: dict[type[ast.unaryop], Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _eval_node(self, node: ast.AST, *, depth: int = 0) -> float:
        if depth > 32:
            raise ValueError("expression_too_deep")
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, depth=depth + 1)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("unsupported_constant")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("non_finite")
            return result
        if isinstance(node, ast.UnaryOp):
            op = self._UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported_unary_op")
            return float(op(self._eval_node(node.operand, depth=depth + 1)))
        if isinstance(node, ast.BinOp):
            op = self._BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported_binary_op")
            left = self._eval_node(node.left, depth=depth + 1)
            right = self._eval_node(node.right, depth=depth + 1)
            result = float(op(left, right))
            if not math.isfinite(result):
                raise ValueError("non_finite")
            return result
        raise ValueError("unsupported_expression")

    def __call__(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", "0"))
        expression = expression.strip()
        if not expression:
            return "error:empty_expression"
        if len(expression) > 256:
            return "error:expression_too_long"
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_node(tree)
        except Exception as exc:  # pragma: no cover - defensive
            return f"error:{exc}"
        if result.is_integer():
            return str(int(result))
        return str(result)


__all__ = ["CalculatorTool", "EchoTool", "ToolRegistry", "Tool"]
