"""Hebbian-like PEFT LoRA organelle on a Hugging Face causal LM backbone."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from symbiont_ecology.host.backbone import HFBackbone
from symbiont_ecology.interfaces.messages import MessageEnvelope, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.organelles.base import Organelle, OrganelleContext
from symbiont_ecology.utils.ids import short_uid
from symbiont_ecology.utils.torch_utils import clamp_norm


@dataclass
class TraceStore:
    pre: torch.Tensor | None = None  # [hidden]
    post: torch.Tensor | None = None  # [hidden]


class HebbianPEFTOrganelle(Organelle):
    def __init__(
        self,
        backbone: HFBackbone,
        rank: int,
        context: OrganelleContext,
        activation_bias: float = 0.0,
    ) -> None:
        organelle_id = context.organelle_id or short_uid("peft")
        super().__init__(organelle_id=organelle_id, context=context)
        self.backbone = backbone
        self.rank = rank
        self.activation_bias = activation_bias
        self.device = backbone.device
        self.tokenizer = backbone.tokenizer
        base_model = backbone.model
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=max(rank * 2, 1),
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        if isinstance(base_model, PeftModel):
            base_model.add_adapter(self.organelle_id, lora_cfg)
            model = base_model
        else:
            model = get_peft_model(base_model, lora_cfg)
            try:
                backbone.model = model  # share PEFT-wrapped backbone to avoid duplicate adapters
            except AttributeError:
                pass
            # rename default adapter to organelle id if necessary
            try:
                model.add_adapter(self.organelle_id, lora_cfg)
                model.set_adapter(self.organelle_id)
            except Exception:
                pass
        self.model = model
        try:
            self.model.set_adapter(self.organelle_id)
        except Exception:
            pass
        self.model.to(self.device)
        self.model.eval()
        self.traces = TraceStore()
        self._fisher_importance: float = 0.0
        self._last_activation_scale: float = 0.0

    def route_probability(self, observation: MessageEnvelope) -> float:
        novelty = float(len(observation.observation.state.get("novel_tokens", [])))
        prob = torch.sigmoid(torch.tensor(novelty + self.activation_bias)).item()
        return prob

    @torch.inference_mode()
    def forward(self, envelope: MessageEnvelope) -> MessageEnvelope:
        try:
            self.model.set_adapter(self.organelle_id)
        except Exception:
            pass
        text = envelope.observation.state.get("text", "")
        if not text:
            return envelope
        enc = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.backbone.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc, output_hidden_states=True)
        hidden_last = outputs.hidden_states[-1]  # [B, T, H]
        pre = self.model.get_input_embeddings()(enc["input_ids"])  # [B, T, H]
        # Store trace as average over sequence
        self.traces.pre = pre.mean(dim=(0, 1)).detach().to(self.device)
        self.traces.post = hidden_last[:, -1, :].mean(dim=0).detach().to(self.device)
        try:
            activation_scale = float(pre.pow(2).mean().item() + hidden_last.pow(2).mean().item())
        except Exception:
            activation_scale = 0.0
        if math.isfinite(activation_scale):
            self._last_activation_scale = activation_scale

        # Generate answer; max_new_tokens and sampling knobs come from host config
        host_cfg = self.backbone.host_config
        try:
            max_new = int(getattr(host_cfg, "gen_max_new_tokens", 48))
        except Exception:
            max_new = 48
        max_new = max(1, min(512, max_new))
        temperature = float(getattr(host_cfg, "temperature", 0.0) or 0.0)
        top_p = float(getattr(host_cfg, "top_p", 1.0) or 1.0)
        intent_goal = ""
        if getattr(envelope, "intent", None) is not None:
            intent_goal = str(getattr(envelope.intent, "goal", "") or "").lower()
        if "team probe" in intent_goal or "team holdout" in intent_goal:
            probe_temp = float(getattr(host_cfg, "team_probe_temperature", 0.0) or 0.0)
            probe_top_p = float(getattr(host_cfg, "team_probe_top_p", 1.0) or 1.0)
            if probe_temp > 0.0:
                temperature = probe_temp
            if probe_top_p < 1.0:
                top_p = probe_top_p
        do_sample = bool(temperature > 0.0 or top_p < 1.0)
        gen_kwargs = {"max_new_tokens": max_new, "do_sample": do_sample}
        if do_sample:
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
            if 0.0 < top_p < 1.0:
                gen_kwargs["top_p"] = top_p
        gen_ids = self.model.generate(**enc, **gen_kwargs)
        answer = self.tokenizer.decode(
            gen_ids[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        envelope.observation.state["answer"] = answer.strip()

        plan_steps = envelope.plan.steps if envelope.plan else []
        plan_steps.append(f"peft::{self.organelle_id}")
        if envelope.plan:
            envelope.plan = envelope.plan.model_copy(
                update={"steps": plan_steps, "confidence": 0.5}
            )
        else:
            envelope.plan = Plan(steps=plan_steps, confidence=0.5)
        return envelope

    def update(self, envelope: MessageEnvelope, reward: RewardBreakdown) -> None:
        if self.traces.pre is None or self.traces.post is None:
            return
        baseline = float(getattr(self.context, "reward_baseline", 0.0))
        centered = reward.total - baseline
        if centered == 0.0:
            return
        # Small, reward-modulated update across all lora layers
        for _name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                self._update_lora_pair(module, centered)
        self.step()
        activation_scale = self._last_activation_scale
        if activation_scale <= 0.0:
            try:
                activation_scale = float(
                    (self.traces.pre.pow(2).mean() + self.traces.post.pow(2).mean()).item()
                )
            except Exception:
                activation_scale = 0.0
        if activation_scale <= 0.0:
            return
        fisher_delta = max(abs(centered), 1e-6) * activation_scale
        decay = 0.9
        if not math.isfinite(fisher_delta):
            return
        self._fisher_importance = decay * self._fisher_importance + (1.0 - decay) * fisher_delta

    def _update_lora_pair(self, module: nn.Module, scale: float) -> None:
        # module.lora_A / lora_B are dict-like keyed by adapter name; choose first
        try:
            adapter_names = list(module.lora_A.keys())
            if not adapter_names:
                return
            adapter = adapter_names[0]
            lora_a = module.lora_A[adapter]
            lora_b = module.lora_B[adapter]
            a_weight = lora_a.weight
            b_weight = lora_b.weight
        except Exception:  # pragma: no cover - defensive against PEFT internals
            return

        # Build low-rank update from traces using a random projection to rank r to fit shapes
        r = a_weight.shape[0] if a_weight.dim() == 2 else self.rank
        in_features = a_weight.shape[1] if a_weight.shape[0] == r else a_weight.shape[0]
        out_features = b_weight.shape[1] if b_weight.shape[0] == r else b_weight.shape[0]
        rng = torch.Generator(device=self.device)
        rng.manual_seed(int(abs(hash(self.organelle_id)) % (2**31 - 1)))

        pre = self.traces.pre[:in_features]
        post = self.traces.post[:out_features]
        # simple outer products projected to rank r; ensure consistent dtype
        ones_a = torch.ones(1, r, device=self.device, dtype=pre.dtype)
        ones_b = torch.ones(r, 1, device=self.device, dtype=post.dtype)
        delta_a = pre.unsqueeze(1) @ ones_a
        delta_b = ones_b @ post.unsqueeze(0)
        delta_a = clamp_norm(delta_a * scale * 1e-3, self.context.hebbian.max_update_norm).to(
            a_weight.dtype
        )
        delta_b = clamp_norm(delta_b * scale * 1e-3, self.context.hebbian.max_update_norm).to(
            b_weight.dtype
        )

        with torch.no_grad():
            if a_weight.shape == delta_a.shape:
                a_weight.add_(delta_a)
            elif a_weight.shape == delta_a.t().shape:
                a_weight.add_(delta_a.t())
            if b_weight.shape == delta_b.shape:
                b_weight.add_(delta_b)
            elif b_weight.shape == delta_b.t().shape:
                b_weight.add_(delta_b.t())

    # ------------------------------------------------------------------
    def get_rank(self) -> int:
        return int(self.rank)

    def export_adapter_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        try:
            self.model.set_adapter(self.organelle_id)
        except Exception:
            pass
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            lora_a = module.lora_A
            lora_b = module.lora_B
            if not isinstance(lora_a, dict) or self.organelle_id not in lora_a:
                continue
            weight_a = lora_a[self.organelle_id].weight.detach().cpu().clone()
            weight_b = lora_b[self.organelle_id].weight.detach().cpu().clone()
            state[f"{name}.lora_A"] = weight_a
            state[f"{name}.lora_B"] = weight_b
        return state

    def fisher_importance(self) -> float:
        if self._fisher_importance > 0.0 and math.isfinite(self._fisher_importance):
            return float(self._fisher_importance)
        total = 0.0
        for _name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            try:
                adapter_names = list(module.lora_A.keys())
            except Exception:
                continue
            if not adapter_names:
                continue
            adapter = adapter_names[0]
            try:
                weight_a = module.lora_A[adapter].weight
                weight_b = module.lora_B[adapter].weight
            except Exception:
                continue
            try:
                total += float(weight_a.pow(2).sum().item() + weight_b.pow(2).sum().item())
            except Exception:
                continue
        return max(total, 0.0)

    def import_adapter_state(self, state: dict[str, torch.Tensor], alpha: float = 1.0) -> None:
        if not state or alpha <= 0.0:
            return
        alpha = float(alpha)
        try:
            self.model.set_adapter(self.organelle_id)
        except Exception:
            pass
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            lora_a = module.lora_A
            lora_b = module.lora_B
            if not isinstance(lora_a, dict) or self.organelle_id not in lora_a:
                continue
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state:
                incoming_a = state[key_a].to(
                    lora_a[self.organelle_id].weight.device, lora_a[self.organelle_id].weight.dtype
                )
                with torch.no_grad():
                    if incoming_a.shape == lora_a[self.organelle_id].weight.shape:
                        lora_a[self.organelle_id].weight.add_(incoming_a * alpha)
            if key_b in state:
                incoming_b = state[key_b].to(
                    lora_b[self.organelle_id].weight.device, lora_b[self.organelle_id].weight.dtype
                )
                with torch.no_grad():
                    if incoming_b.shape == lora_b[self.organelle_id].weight.shape:
                        lora_b[self.organelle_id].weight.add_(incoming_b * alpha)

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        """Replace adapter weights with a saved snapshot.

        Unlike `import_adapter_state` (which is additive and used for merging), this loader
        overwrites LoRA weights to match the provided snapshot exactly.
        """
        if not state:
            return
        try:
            self.model.set_adapter(self.organelle_id)
        except Exception:
            pass
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            lora_a = module.lora_A
            lora_b = module.lora_B
            if not isinstance(lora_a, dict) or self.organelle_id not in lora_a:
                continue
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state:
                incoming_a = state[key_a].to(
                    lora_a[self.organelle_id].weight.device,
                    lora_a[self.organelle_id].weight.dtype,
                )
                with torch.no_grad():
                    if incoming_a.shape == lora_a[self.organelle_id].weight.shape:
                        lora_a[self.organelle_id].weight.copy_(incoming_a)
            if key_b in state:
                incoming_b = state[key_b].to(
                    lora_b[self.organelle_id].weight.device,
                    lora_b[self.organelle_id].weight.dtype,
                )
                with torch.no_grad():
                    if incoming_b.shape == lora_b[self.organelle_id].weight.shape:
                        lora_b[self.organelle_id].weight.copy_(incoming_b)

    def trainable_parameters(self) -> int:
        total = 0
        for module in self.model.modules():
            if hasattr(module, "lora_A"):
                for adapter, lora in module.lora_A.items():
                    if adapter == self.organelle_id:
                        total += lora.weight.numel()
            if hasattr(module, "lora_B"):
                for adapter, lora in module.lora_B.items():
                    if adapter == self.organelle_id:
                        total += lora.weight.numel()
        return total

    def estimate_trainable(self, new_rank: int) -> int:
        current = max(self.trainable_parameters(), 1)
        base_rank = max(self.rank, 1)
        return int(current * (max(new_rank, 1) / base_rank))

    def resize_rank(self, new_rank: int) -> bool:
        new_rank = max(1, new_rank)
        if new_rank == self.rank:
            return False
        old_weights: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_a = module.lora_A
                lora_b = module.lora_B
                if isinstance(lora_a, dict) and self.organelle_id in lora_a:
                    old_weights[name] = (
                        lora_a[self.organelle_id].weight.detach().clone(),
                        lora_b[self.organelle_id].weight.detach().clone(),
                    )
        try:
            if hasattr(self.model, "delete_adapter"):
                self.model.delete_adapter(self.organelle_id)
        except Exception:
            pass
        lora_cfg = LoraConfig(
            r=new_rank,
            lora_alpha=max(new_rank * 2, 1),
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        base_model = self.backbone.model
        if isinstance(base_model, PeftModel):
            base_model.add_adapter(self.organelle_id, lora_cfg)
            base_model.set_adapter(self.organelle_id)
            self.model = base_model
        else:
            model = get_peft_model(base_model, lora_cfg)
            try:
                model.add_adapter(self.organelle_id, lora_cfg)
                model.set_adapter(self.organelle_id)
            except Exception:
                pass
            self.model = model
        for name, module in self.model.named_modules():
            if (
                name not in old_weights
                or not hasattr(module, "lora_A")
                or not hasattr(module, "lora_B")
            ):
                continue
            lora_a = module.lora_A
            lora_b = module.lora_B
            if not isinstance(lora_a, dict) or self.organelle_id not in lora_a:
                continue
            old_a, old_b = old_weights[name]
            adapter = self.organelle_id
            new_a = lora_a[adapter].weight
            new_b = lora_b[adapter].weight
            try:
                projected_a, projected_b = self._recompose_from_history(
                    new_rank=new_rank,
                    new_a=new_a,
                    new_b=new_b,
                    old_a=old_a,
                    old_b=old_b,
                    module_name=name,
                )
                with torch.no_grad():
                    new_a.copy_(projected_a)
                    new_b.copy_(projected_b)
            except RuntimeError:
                rows = min(old_a.shape[0], new_a.shape[0])
                cols = min(old_a.shape[1], new_a.shape[1])
                rows_b = min(old_b.shape[0], new_b.shape[0])
                cols_b = min(old_b.shape[1], new_b.shape[1])
                with torch.no_grad():
                    new_a[:rows, :cols].copy_(old_a[:rows, :cols])
                    new_b[:rows_b, :cols_b].copy_(old_b[:rows_b, :cols_b])
        self.rank = new_rank
        return True

    def active_adapters(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        if hasattr(self.model, "peft_config"):
            config = self.model.peft_config
            if isinstance(config, dict) and self.organelle_id in config:
                target = config[self.organelle_id].target_modules
                for module_name in target:
                    summary[module_name] = summary.get(module_name, 0) + 1
        if not summary:
            summary["total"] = 0
        summary["rank"] = int(self.rank)
        return summary

    @staticmethod
    def _recompose_from_history(
        new_rank: int,
        new_a: torch.Tensor,
        new_b: torch.Tensor,
        old_a: torch.Tensor,
        old_b: torch.Tensor,
        module_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = new_a.device
        dtype = new_a.dtype
        target_rank = new_rank
        out_features, _ = new_b.shape
        _, in_features = new_a.shape
        projected_a = torch.zeros(target_rank, in_features, device=device, dtype=dtype)
        projected_b = torch.zeros(out_features, target_rank, device=device, dtype=dtype)
        try:
            delta = old_b.to(device=device, dtype=torch.float32) @ old_a.to(
                device=device, dtype=torch.float32
            )
            if delta.numel() == 0:
                return projected_a, projected_b
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            keep = min(target_rank, s.shape[0])
            if keep > 0:
                sqrt_s = torch.sqrt(s[:keep].clamp_min(1e-9))
                left = u[:, :keep] * sqrt_s
                right = torch.diag(sqrt_s) @ vh[:keep, :]
                projected_b[:, :keep] = left.to(dtype=dtype, device=device)
                projected_a[:keep, :] = right.to(dtype=dtype, device=device)
            if target_rank > keep:
                remaining = target_rank - keep
                rng = torch.Generator(device=device)
                seed_material = f"{module_name}:{target_rank}:{in_features}:{out_features}".encode(
                    "utf-8"
                )
                digest = hashlib.sha1(seed_material, usedforsecurity=False).hexdigest()
                seed = int(digest[:16], 16) % (2**31 - 1)
                rng.manual_seed(seed)
                noise_a = (
                    torch.randn(
                        remaining,
                        in_features,
                        device=device,
                        dtype=dtype,
                        generator=rng,
                    )
                    * 1e-3
                )
                noise_b = (
                    torch.randn(
                        out_features,
                        remaining,
                        device=device,
                        dtype=dtype,
                        generator=rng,
                    )
                    * 1e-3
                )
                projected_a[keep:, :] = noise_a
                projected_b[:, keep:] = noise_b
            return projected_a, projected_b
        except RuntimeError as exc:  # pragma: no cover - fallback path
            raise RuntimeError("SVD recomposition failed") from exc


__all__ = ["HebbianPEFTOrganelle"]
