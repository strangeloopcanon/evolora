"""Hebbian LoRA organelles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from symbiont_ecology.config import HebbianConfig
from symbiont_ecology.environment.bridge import ToolRegistry
from symbiont_ecology.interfaces.messages import MessageEnvelope, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.organelles.base import Organelle, OrganelleContext
from symbiont_ecology.utils.ids import short_uid
from symbiont_ecology.utils.torch_utils import clamp_norm, ensure_dtype, no_grad


@dataclass
class HebbianState:
    traces_a: torch.Tensor
    traces_b: torch.Tensor
    baseline: float


class HebbianLoRAAdapter(nn.Module):
    """Low-rank adapter updated via reward-modulated Hebbian learning."""

    def __init__(
        self,
        input_dim: int,
        rank: int,
        config: HebbianConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.config = config
        self.lora_A = nn.Parameter(torch.zeros(input_dim, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, input_dim, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.lora_A)
        nn.init.xavier_uniform_(self.lora_B)
        self.state = HebbianState(
            traces_a=torch.zeros_like(self.lora_A),
            traces_b=torch.zeros_like(self.lora_B),
            baseline=0.0,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = cast(torch.Tensor, hidden @ self.lora_A @ self.lora_B)
        return hidden + delta

    def update_traces(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        decay = self.config.trace_decay
        with no_grad():
            projected = pre @ self.lora_A
            self.state.traces_a.mul_(decay).add_(pre.transpose(0, 1) @ projected / pre.shape[0])
            self.state.traces_b.mul_(decay).add_(projected.transpose(0, 1) @ post / pre.shape[0])

    def apply_reward(self, reward: float) -> None:
        lr = self.config.learning_rate
        centered = reward - self.state.baseline
        self.state.baseline = (
            self.config.reward_baseline_decay * self.state.baseline
            + (1 - self.config.reward_baseline_decay) * reward
        )
        with no_grad():
            update_a = clamp_norm(self.state.traces_a * centered * lr, self.config.max_update_norm)
            update_b = clamp_norm(self.state.traces_b * centered * lr, self.config.max_update_norm)
            self.lora_A.add_(update_a)
            self.lora_B.add_(update_b)


class HebbianLoRAOrganelle(Organelle):
    def __init__(
        self,
        input_dim: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        context: OrganelleContext,
        activation_bias: float = 0.0,
    ) -> None:
        organelle_id = context.organelle_id or short_uid("org")
        super().__init__(organelle_id=organelle_id, context=context)
        self.adapter = HebbianLoRAAdapter(input_dim, rank, context.hebbian, dtype, device)
        self.activation_bias = activation_bias
        self.device = device
        self.dtype = dtype
        self.tools = ToolRegistry({})
        from symbiont_ecology.environment.bridge import CalculatorTool

        self.tools.tools.setdefault("echo", lambda **kw: str(kw.get("text", "")))
        self.tools.tools.setdefault("calc", CalculatorTool())

    def route_probability(self, observation: MessageEnvelope) -> float:
        novelty = float(len(observation.observation.state.get("novel_tokens", [])))
        prob = torch.sigmoid(torch.tensor(novelty + self.activation_bias)).item()
        return prob

    def forward(self, envelope: MessageEnvelope) -> MessageEnvelope:
        latent_values = envelope.observation.state.get("latent")
        if latent_values is None:
            latent_values = torch.randn(1, self.adapter.lora_A.shape[0], device=self.device)
        else:
            latent_values = torch.tensor(latent_values, device=self.device)
        if latent_values.dim() == 1:
            latent_values = latent_values.unsqueeze(0)
        latent_values = ensure_dtype(latent_values, self.adapter.lora_A.dtype)
        enriched = self.adapter(latent_values)
        plan_steps = envelope.plan.steps if envelope.plan else []
        plan_steps.append(f"adapted::{self.organelle_id}")
        if envelope.plan:
            envelope.plan = envelope.plan.model_copy(
                update={"steps": plan_steps, "confidence": 0.5}
            )
        else:
            envelope.plan = Plan(steps=plan_steps, confidence=0.5)
        envelope.observation.state["latent"] = enriched.squeeze(0).tolist()
        prompt = envelope.observation.state.get("text", "")
        answer = self._derive_answer(prompt)
        if answer:
            envelope.observation.state["answer"] = answer
            envelope.plan.steps.append(f"answer::{answer}")
        return envelope

    def update(self, envelope: MessageEnvelope, reward: RewardBreakdown) -> None:
        latent_values = envelope.observation.state.get("latent")
        if latent_values is None:
            latent = torch.randn(1, self.adapter.lora_A.shape[0], device=self.device)
        else:
            latent = torch.tensor(latent_values, device=self.device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent = ensure_dtype(latent, self.adapter.lora_A.dtype)
        post = self.adapter(latent)
        self.adapter.update_traces(latent, post)
        self.adapter.apply_reward(reward.total)
        self.step()

    # ------------------------------------------------------------------
    def get_rank(self) -> int:
        return int(self.adapter.rank)

    def trainable_parameters(self) -> int:
        return int(self.adapter.lora_A.numel() + self.adapter.lora_B.numel())

    def estimate_trainable(self, new_rank: int) -> int:
        new_rank = max(new_rank, 0)
        input_dim = self.adapter.lora_A.shape[0]
        return int(2 * input_dim * new_rank)

    def resize_rank(self, new_rank: int) -> bool:
        new_rank = max(1, new_rank)
        if new_rank == self.adapter.rank:
            return False
        input_dim = self.adapter.lora_A.shape[0]
        dtype = self.adapter.lora_A.dtype
        device = self.adapter.lora_A.device
        new_adapter = HebbianLoRAAdapter(
            input_dim=input_dim,
            rank=new_rank,
            config=self.context.hebbian,
            dtype=dtype,
            device=device,
        )
        old_rank = min(self.adapter.rank, new_rank)
        if old_rank > 0:
            with torch.no_grad():
                new_adapter.lora_A[:, :old_rank].copy_(self.adapter.lora_A[:, :old_rank])
                new_adapter.lora_B[:old_rank, :].copy_(self.adapter.lora_B[:old_rank, :])
                new_adapter.state.traces_a[:, :old_rank].copy_(
                    self.adapter.state.traces_a[:, :old_rank]
                )
                new_adapter.state.traces_b[:old_rank, :].copy_(
                    self.adapter.state.traces_b[:old_rank, :]
                )
        new_adapter.state.baseline = self.adapter.state.baseline
        self.adapter = new_adapter
        return True

    # ------------------------------------------------------------------
    def export_adapter_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        state["adapter.lora_A"] = self.adapter.lora_A.detach().cpu().clone()
        state["adapter.lora_B"] = self.adapter.lora_B.detach().cpu().clone()
        return state

    def import_adapter_state(self, state: dict[str, torch.Tensor], alpha: float = 1.0) -> None:
        if not state or alpha <= 0.0:
            return
        alpha = float(alpha)
        device = self.adapter.lora_A.device
        dtype_a = self.adapter.lora_A.dtype
        dtype_b = self.adapter.lora_B.dtype
        with torch.no_grad():
            incoming_a = state.get("adapter.lora_A")
            if incoming_a is not None and incoming_a.shape == self.adapter.lora_A.shape:
                delta_a = incoming_a.to(device=device, dtype=dtype_a)
                self.adapter.lora_A.add_(delta_a * alpha)
            incoming_b = state.get("adapter.lora_B")
            if incoming_b is not None and incoming_b.shape == self.adapter.lora_B.shape:
                delta_b = incoming_b.to(device=device, dtype=dtype_b)
                self.adapter.lora_B.add_(delta_b * alpha)

    def active_adapters(self) -> dict[str, int]:
        return {"dense": 1 if self.adapter.rank > 0 else 0, "rank": int(self.adapter.rank)}

    def _derive_answer(self, prompt: str) -> str:
        text = prompt.lower()
        numbers = [int(num) for num in re.findall(r"-?\d+", text)]
        if "add" in text and len(numbers) >= 2:
            return str(self.tools.call(name="calc", expression=f"{numbers[0]}+{numbers[1]}"))
        if ("multiply" in text or "product" in text) and len(numbers) >= 2:
            return str(self.tools.call(name="calc", expression=f"{numbers[0]}*{numbers[1]}"))
        if "reverse" in text:
            match = re.search(r"'([^']+)'", prompt)
            if match:
                return match.group(1)[::-1]
        if "prime" in text:
            upper_match = re.search(r"up to (\d+)", text)
            if upper_match:
                upper = int(upper_match.group(1))
                primes = [str(value) for value in range(2, upper + 1) if self._is_prime(value)]
                return ",".join(primes)
        return ""

    @staticmethod
    def _is_prime(value: int) -> bool:
        if value < 2:
            return False
        for factor in range(2, int(value**0.5) + 1):
            if value % factor == 0:
                return False
        return True


__all__ = ["HebbianLoRAAdapter", "HebbianLoRAOrganelle"]
