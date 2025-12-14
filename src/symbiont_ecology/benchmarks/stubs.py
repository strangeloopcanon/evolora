"""Deterministic stubs used by the benchmark harness.

These are intended for CI/smoke benchmarks so we can exercise the full ecology
stack without downloading large models.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from typing import Iterable, Optional

import torch

from symbiont_ecology.config import HostConfig
from symbiont_ecology.interfaces.messages import MessageEnvelope, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.organelles.base import Organelle, OrganelleContext


class BenchmarkStubBackbone:
    """Tiny deterministic backbone that produces a latent embedding."""

    def __init__(self, host_config: HostConfig) -> None:
        self.host_config = host_config
        self.device = torch.device("cpu")
        self.hidden_size = 64
        self.max_length = host_config.max_sequence_length
        self.tokenizer = None
        self.model = None

    def encode_text(
        self, text_batch: Iterable[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        target_device = device or self.device
        batch = list(text_batch)
        # Provide stable-but-informative latents by hashing the prompt text.
        latents: list[torch.Tensor] = []
        for text in batch:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed % (2**31 - 1)))
            latents.append(
                torch.randn(self.hidden_size, generator=gen, dtype=torch.float32, device="cpu")
            )
        stacked = torch.stack(latents, dim=0)
        return stacked.to(target_device)

    def parameters(self):  # pragma: no cover - stub for HostKernel.freeze_host
        return []


def _count_alpha_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text))


def _parse_ints(text: str) -> list[int]:
    return [int(num) for num in re.findall(r"-?\d+", text)]


def _safe_eval_math(expr: str) -> float | None:
    expr = expr.strip()
    if not expr:
        return None
    # Allow digits, whitespace, parentheses, and basic arithmetic operators.
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return None
    try:
        parsed = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError("Unsupported binary operator")
        raise ValueError("Unsupported expression")

    try:
        return float(_eval(parsed.body))
    except Exception:
        return None


def _camel_to_snake(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def solve_prompt(prompt: str) -> str:
    """Deterministically solve the built-in GridEnvironment prompt formats."""
    text = prompt.strip()
    lower = text.lower()

    if lower.startswith("add") or " respond with the number only" in lower and "add" in lower:
        nums = _parse_ints(text)
        if len(nums) >= 2:
            return str(nums[0] + nums[1])

    if lower.startswith("multiply") or "multiply" in lower or "product" in lower:
        nums = _parse_ints(text)
        if len(nums) >= 2:
            return str(nums[0] * nums[1])

    if lower.startswith("given the numbers") and "json array" in lower:
        nums = _parse_ints(text)
        return json.dumps(sorted(nums))

    if lower.startswith("sort the following letters"):
        letters = [tok.strip() for tok in text.split() if tok.strip().isalpha() and len(tok) == 1]
        if letters:
            return "".join(sorted(letters))

    if lower.startswith("count the number of words") or lower.startswith(
        "count the number of alphabetic words"
    ):
        snippet = ""
        if "`" in text:
            parts = text.split("`")
            if len(parts) >= 3:
                snippet = parts[1]
        if not snippet:
            match = re.search(r"'([^']+)'", text)
            if match:
                snippet = match.group(1)
        if snippet:
            return str(_count_alpha_words(snippet))

    if lower.startswith("evaluate the logical expression"):
        # "Evaluate ...: <EXPR>"
        expr = text.split(":", 1)[-1].strip()
        tokens = [tok for tok in expr.split() if tok]
        idx = 0

        def read_literal() -> bool | None:
            nonlocal idx
            negate = False
            if idx < len(tokens) and tokens[idx].upper() == "NOT":
                negate = True
                idx += 1
            if idx >= len(tokens):
                return None
            tok = tokens[idx].upper()
            idx += 1
            if tok == "TRUE":
                val = True
            elif tok == "FALSE":
                val = False
            else:
                return None
            return (not val) if negate else val

        truth_value = read_literal()
        if truth_value is None:
            return ""
        while idx < len(tokens):
            op = tokens[idx].upper()
            idx += 1
            rhs = read_literal()
            if rhs is None:
                break
            if op == "AND":
                truth_value = bool(truth_value and rhs)
            elif op == "OR":
                truth_value = bool(truth_value or rhs)
            else:
                break
        return "True" if truth_value else "False"

    if lower.startswith("given the sequence") and "what is the next number" in lower:
        nums = _parse_ints(text)
        if len(nums) >= 3:
            diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
            if len(set(diffs)) == 1:
                return str(nums[-1] + diffs[0])
            ratios: list[float] = []
            ok = True
            for i in range(len(nums) - 1):
                if nums[i] == 0:
                    ok = False
                    break
                ratios.append(nums[i + 1] / nums[i])
            if ok and ratios and max(ratios) - min(ratios) < 1e-6:
                return str(nums[-1] * ratios[0])
        return ""

    if lower.startswith("compute") and "respond with the number only" in lower:
        expr = text[len("Compute") :].split(".", 1)[0].strip()
        computed = _safe_eval_math(expr)
        if computed is None or not math.isfinite(computed):
            return ""
        # Prefer integer formatting when possible.
        if abs(computed - round(computed)) < 1e-9:
            return str(int(round(computed)))
        return str(computed)

    if lower.startswith("convert the variable name") and "snake_case" in lower:
        match = re.search(r"`([^`]+)`", text)
        if match:
            return _camel_to_snake(match.group(1))
        return ""

    return ""


class BenchmarkStubOrganelle(Organelle):
    """Deterministic organelle that solves grid prompts without a language model."""

    def __init__(
        self,
        backbone: BenchmarkStubBackbone,
        rank: int,
        context: OrganelleContext,
        activation_bias: float = 0.0,
    ) -> None:
        super().__init__(organelle_id=context.organelle_id, context=context)
        self.backbone = backbone
        self.rank = int(rank)
        self.activation_bias = float(activation_bias)

    def route_probability(
        self, observation: MessageEnvelope
    ) -> float:  # pragma: no cover - trivial
        return 1.0

    def forward(self, envelope: MessageEnvelope) -> MessageEnvelope:
        prompt = str(envelope.observation.state.get("text", "") or "")
        answer = solve_prompt(prompt)
        if answer:
            envelope.observation.state["answer"] = answer
        plan_steps = envelope.plan.steps if envelope.plan else []
        plan_steps.append(f"bench::{self.organelle_id}")
        envelope.plan = Plan(steps=plan_steps, confidence=1.0)
        return envelope

    def update(
        self, envelope: MessageEnvelope, reward: RewardBreakdown
    ) -> None:  # pragma: no cover
        del envelope, reward
        self.step()

    def trainable_parameters(self) -> int:
        return int(self.rank * self.backbone.hidden_size * 2)

    def export_adapter_state(
        self,
    ) -> dict[str, torch.Tensor]:  # pragma: no cover - unused in CI suite
        return {}

    def import_adapter_state(
        self, state: dict[str, torch.Tensor], alpha: float = 1.0
    ) -> None:  # pragma: no cover
        del state, alpha

    def active_adapters(self) -> dict[str, int]:
        return {"rank": int(self.rank)}


__all__ = ["BenchmarkStubBackbone", "BenchmarkStubOrganelle", "solve_prompt"]
