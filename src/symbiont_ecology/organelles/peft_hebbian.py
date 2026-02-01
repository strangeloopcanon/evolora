"""Hebbian-like PEFT LoRA organelle on a Hugging Face causal LM backbone."""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Tuple, cast

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
        self._lora_cfg = lora_cfg
        if isinstance(base_model, PeftModel):
            model = base_model
            try:
                model.add_adapter(self.organelle_id, lora_cfg)
            except Exception:
                pass
        else:
            model = get_peft_model(base_model, lora_cfg, adapter_name=self.organelle_id)
            backbone.model = model  # type: ignore[assignment]  # share PEFT-wrapped backbone to avoid duplicate adapters
        self.model = model
        self._ensure_active_adapter()
        self.model.to(self.device)
        self.model.eval()
        self.traces = TraceStore()
        self._fisher_importance: float = 0.0
        self._last_activation_scale: float = 0.0

    def _ensure_active_adapter(self) -> bool:
        """Ensure this organelle's adapter exists and is active.

        This guards against accidental adapter divergence collapse where an organelle
        silently runs with a different active adapter (e.g., after deletions).
        """
        try:
            self.model.set_adapter(self.organelle_id)
            return True
        except Exception:
            pass
        try:
            cfg = getattr(self, "_lora_cfg", None)
            add_adapter = getattr(self.model, "add_adapter", None)
            set_adapter = getattr(self.model, "set_adapter", None)
            if cfg is None or not callable(add_adapter) or not callable(set_adapter):
                return False
            add_adapter(self.organelle_id, cfg)
            set_adapter(self.organelle_id)
            return True
        except Exception:
            return False

    def route_probability(self, observation: MessageEnvelope) -> float:
        novelty = float(len(observation.observation.state.get("novel_tokens", [])))
        prob = torch.sigmoid(torch.tensor(novelty + self.activation_bias)).item()
        return prob

    @torch.inference_mode()
    def forward(self, envelope: MessageEnvelope) -> MessageEnvelope:
        if not self._ensure_active_adapter():
            envelope.observation.state["answer"] = ""
            return envelope
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
        try:
            envelope.observation.state["prompt_tokens"] = int(enc["input_ids"].shape[1])
        except Exception:
            envelope.observation.state["prompt_tokens"] = 0

        # Pre-trace: prefer the host latent (contextual last-token embedding); fall back to prompt embeddings.
        pre_vec: torch.Tensor | None = None
        latent = envelope.observation.state.get("latent")
        if isinstance(latent, list) and latent:
            try:
                pre_vec = torch.tensor(latent, device=self.device, dtype=torch.float32)
            except Exception:
                pre_vec = None
        if pre_vec is None:
            try:
                prompt_emb = self.model.get_input_embeddings()(enc["input_ids"])  # [B, T, H]
                pre_vec = prompt_emb.mean(dim=(0, 1)).detach()
            except Exception:
                pre_vec = None

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
        gen_kwargs: dict[str, object] = {"max_new_tokens": max_new, "do_sample": do_sample}
        if do_sample:
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
            if 0.0 < top_p < 1.0:
                gen_kwargs["top_p"] = top_p
        gen_out = self.model.generate(
            **enc,
            **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_ids = getattr(gen_out, "sequences", gen_out)
        try:
            prompt_len = int(enc["input_ids"].shape[1])
            envelope.observation.state["generated_tokens"] = max(
                0, int(gen_ids[0].shape[0]) - prompt_len
            )
        except Exception:
            envelope.observation.state["generated_tokens"] = 0

        # Post-trace: a reward-modulated summary of the generated tokens, weighted by token surprisal
        # (low-probability tokens get higher credit assignment).
        post_vec: torch.Tensor | None = None
        try:
            prompt_len = int(enc["input_ids"].shape[1])
            gen_len = max(0, int(gen_ids.shape[1]) - prompt_len)
        except Exception:
            prompt_len = 0
            gen_len = 0

        if gen_len > 0:
            try:
                gen_token_ids = gen_ids[:, prompt_len:]
                gen_emb = self.model.get_input_embeddings()(gen_token_ids)  # [B, gen_len, H]
            except Exception:
                gen_token_ids = None
                gen_emb = None

            weights: torch.Tensor | None = None
            scores = getattr(gen_out, "scores", None)
            if (
                gen_token_ids is not None
                and isinstance(scores, (list, tuple))
                and len(scores) >= gen_len
            ):
                try:
                    logprobs: list[torch.Tensor] = []
                    for step_idx in range(gen_len):
                        step_scores = scores[step_idx]
                        if step_scores is None:
                            continue
                        step_scores_0 = step_scores[0].float()
                        token_id = int(gen_token_ids[0, step_idx].item())
                        token_score = step_scores_0[token_id]
                        log_denom = torch.logsumexp(step_scores_0, dim=-1)
                        logprobs.append(token_score - log_denom)
                    if len(logprobs) == gen_len:
                        logprobs_t = torch.stack(logprobs)  # [gen_len]
                        envelope.observation.state["gen_mean_logprob"] = float(
                            logprobs_t.mean().item()
                        )
                        envelope.observation.state["gen_min_logprob"] = float(
                            logprobs_t.min().item()
                        )
                        envelope.observation.state["gen_max_logprob"] = float(
                            logprobs_t.max().item()
                        )
                        surprisal = (-logprobs_t).clamp(min=0.0, max=5.0)
                        denom = float(surprisal.sum().item())
                        if denom > 0.0 and math.isfinite(denom):
                            weights = surprisal / float(denom)
                except Exception:
                    weights = None

            if gen_emb is not None:
                if weights is None:
                    weights = torch.full((gen_len,), 1.0 / float(gen_len), device=gen_emb.device)
                w = weights.to(device=gen_emb.device, dtype=gen_emb.dtype).view(1, gen_len, 1)
                post_vec = (gen_emb * w).sum(dim=1).mean(dim=0).detach()

        if post_vec is None:
            post_vec = pre_vec

        self.traces.pre = pre_vec.detach().to(self.device) if pre_vec is not None else None
        self.traces.post = post_vec.detach().to(self.device) if post_vec is not None else None
        if self.traces.pre is not None and self.traces.post is not None:
            try:
                activation_scale = float(
                    self.traces.pre.pow(2).mean().item() + self.traces.post.pow(2).mean().item()
                )
            except Exception:
                activation_scale = 0.0
            if math.isfinite(activation_scale):
                self._last_activation_scale = activation_scale

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

    def update(self, envelope: MessageEnvelope, reward: RewardBreakdown) -> bool:
        if self.traces.pre is None or self.traces.post is None:
            return False
        family_key = "global"
        try:
            state = envelope.observation.state
            family_val = state.get("task_family") or state.get("family") or ""
            family_str = str(family_val).strip()
            if family_str:
                family_key = family_str
        except Exception:
            family_key = "global"
        # Learning signal: exclude compute cost. Evolution already accounts for compute via energy/ROI,
        # and using cost inside the plasticity rule biases learning toward "being cheap" over "being right".
        signal = float(reward.total + reward.cost_penalty)
        baseline_map = getattr(self.context, "reward_baseline_by_family", None)
        if isinstance(baseline_map, dict):
            baseline = float(baseline_map.get(family_key, 0.0))
        else:
            baseline = float(getattr(self.context, "reward_baseline", 0.0))
        decay = float(getattr(self.context.hebbian, "reward_baseline_decay", 0.99) or 0.99)
        if math.isfinite(decay):
            decay = max(0.0, min(1.0, decay))
            try:
                updated = decay * baseline + (1.0 - decay) * signal
                if isinstance(baseline_map, dict):
                    baseline_map[family_key] = updated
                else:
                    self.context.reward_baseline = updated
            except Exception:
                pass
        centered = signal - baseline
        # Learn from improvements (including high partial credit), without reinforcing mediocrity.
        # Many regex tasks provide fractional `task_reward` (case ratios), so requiring exact success
        # makes learning too sparse (especially for debugging/mutation-effect). We instead:
        # - track a moving baseline of progress (task_reward clamped to [0,1])
        # - only update when progress clears the baseline by a small margin (or is above a floor)
        try:
            progress = float(reward.task_reward)
        except Exception:
            progress = 0.0
        progress = max(0.0, min(1.0, progress))
        progress_map = getattr(self.context, "progress_baseline_by_family", None)
        if isinstance(progress_map, dict):
            prev_progress_baseline = float(progress_map.get(family_key, 0.0))
        else:
            prev_progress_baseline = float(getattr(self.context, "progress_baseline", 0.0))
        # Use a faster decay than the reward baseline so the gate adapts quickly.
        progress_decay = 0.9
        if math.isfinite(prev_progress_baseline):
            prev_progress_baseline = max(0.0, min(1.0, prev_progress_baseline))
        else:
            prev_progress_baseline = 0.0
        try:
            updated_progress = (
                progress_decay * prev_progress_baseline + (1.0 - progress_decay) * progress
            )
            if isinstance(progress_map, dict):
                progress_map[family_key] = updated_progress
            else:
                self.context.progress_baseline = updated_progress
        except Exception:
            pass
        # Gate: compare progress to the *per-family* baseline (different families have different reward scales).
        # We remove the absolute floor (0.05) to allow learning on hard tasks (e.g. mutation_effect)
        # where the model might be improving from 0.0 to 0.01.
        required_progress = prev_progress_baseline
        if progress < required_progress:
            return False
        # Negative updates tend to destabilize this non-gradient Hebbian rule; learn from improvements only.
        if centered <= 0.0:
            return False
        # Small, reward-modulated update across all lora layers
        for module_name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                self._update_lora_pair(module, centered, module_name=module_name)
        self.step()
        did_update = True
        activation_scale = self._last_activation_scale
        if activation_scale <= 0.0:
            try:
                activation_scale = float(
                    (self.traces.pre.pow(2).mean() + self.traces.post.pow(2).mean()).item()
                )
            except Exception:
                activation_scale = 0.0
        if activation_scale <= 0.0:
            return did_update
        fisher_delta = max(abs(centered), 1e-6) * activation_scale
        decay = 0.9
        if not math.isfinite(fisher_delta):
            return did_update
        self._fisher_importance = decay * self._fisher_importance + (1.0 - decay) * fisher_delta
        return did_update

    def _update_lora_pair(self, module: nn.Module, scale: float, *, module_name: str = "") -> None:
        # module.lora_A / lora_B are dict-like keyed by adapter name; update this organelle's adapter.
        try:
            adapter = self.organelle_id
            lora_a_dict = getattr(module, "lora_A", None)
            lora_b_dict = getattr(module, "lora_B", None)
            # Check for dict-like interface (dict or ModuleDict)
            if lora_a_dict is None or lora_b_dict is None:
                return
            if not hasattr(lora_a_dict, "__contains__") or not hasattr(lora_b_dict, "__contains__"):
                return
            if adapter not in lora_a_dict or adapter not in lora_b_dict:
                return
            lora_a = lora_a_dict[adapter]
            lora_b = lora_b_dict[adapter]
            a_weight = lora_a.weight
            b_weight = lora_b.weight
        except Exception:  # pragma: no cover - defensive against PEFT internals
            return

        # Build low-rank update from traces using a random projection to rank r to fit shapes
        r = a_weight.shape[0] if a_weight.dim() == 2 else self.rank
        in_features = a_weight.shape[1] if a_weight.shape[0] == r else a_weight.shape[0]
        out_features = b_weight.shape[1] if b_weight.shape[0] == r else b_weight.shape[0]
        module_name = module_name or ""

        def _stable_seed(material: str) -> int:
            digest = hashlib.sha1(material.encode("utf-8"), usedforsecurity=False).hexdigest()
            return int(digest[:16], 16) % (2**31 - 1)

        pre = self.traces.pre
        post = self.traces.post

        def _match_dim(vec: torch.Tensor, target: int, label: str) -> torch.Tensor:
            if vec.shape[0] == target:
                return vec
            if vec.shape[0] > target:
                return vec[:target]
            # Expand via a deterministic CountSketch-like projection.
            # Keep the projection fixed per-module to avoid introducing unstable noise.
            rng = torch.Generator(device=vec.device)
            seed = _stable_seed(
                f"hebbian-proj:{label}:{self.organelle_id}:{module_name}:{vec.shape[0]}:{target}"
            )
            rng.manual_seed(seed)
            idx = torch.randint(0, vec.shape[0], (target,), device=vec.device, generator=rng)
            signs = (
                torch.randint(0, 2, (target,), device=vec.device, generator=rng, dtype=vec.dtype)
                * 2
                - 1
            )
            return vec[idx] * signs

        if pre is None or post is None:
            return
        pre = _match_dim(pre, in_features, "pre")
        post = _match_dim(post, out_features, "post")

        # Sample Rademacher sign vector s of length r for CountSketch-like update
        # s in {-1, 1}
        rng = torch.Generator(device=self.device)
        rng.manual_seed(
            _stable_seed(f"hebbian-s:{self.organelle_id}:{module_name}:{self.steps}:{r}")
        )
        s = torch.randint(0, 2, (r,), device=self.device, generator=rng, dtype=pre.dtype) * 2 - 1

        # delta_a = pre @ s.T (outer product) -> [in, r]
        # delta_b = s @ post.T (outer product) -> [r, out]
        # Normalization: sqrt(r) so that delta_a @ delta_b ~ r * pre @ post / r = pre @ post
        norm = math.sqrt(r)
        delta_a = torch.outer(pre, s) / norm
        delta_b = torch.outer(s, post) / norm

        lr = 1e-3
        try:
            lr = float(getattr(self.context.hebbian, "learning_rate", lr))
        except Exception:
            lr = 1e-3
        if not math.isfinite(lr) or lr <= 0.0:
            return
        delta_a = clamp_norm(delta_a * scale * lr, self.context.hebbian.max_update_norm).to(
            a_weight.dtype
        )
        delta_b = clamp_norm(delta_b * scale * lr, self.context.hebbian.max_update_norm).to(
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

    def inherit_from(self, parent: Organelle) -> None:
        """Inherit LoRA weights from a parent organelle.

        This is the key "heredity" mechanism for evolution: offspring should start from
        the parent's learned adapter weights (even when ranks differ).

        Important: keep this implementation CPU-friendly (no full-matrix SVD), since
        MPS linalg support is incomplete and we don't want evolution to stall/crash.
        """

        parent_state = parent.export_adapter_state()
        if not parent_state:
            return

        def _std_a(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
            if tensor.ndim != 2:
                raise ValueError("Expected rank-2 LoRA A tensor")
            # LoRA A is typically [r, in]; if transposed [in, r], flip.
            if tensor.shape[0] <= tensor.shape[1]:
                return tensor, False
            return tensor.t(), True

        def _std_b(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
            if tensor.ndim != 2:
                raise ValueError("Expected rank-2 LoRA B tensor")
            # LoRA B is typically [out, r]; if transposed [r, out], flip.
            if tensor.shape[0] >= tensor.shape[1]:
                return tensor, False
            return tensor.t(), True

        try:
            self.model.set_adapter(self.organelle_id)
        except Exception:
            pass

        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            try:
                if self.organelle_id not in module.lora_A or self.organelle_id not in module.lora_B:
                    continue
            except Exception:
                continue

            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a not in parent_state or key_b not in parent_state:
                continue
            old_a_raw = parent_state[key_a]
            old_b_raw = parent_state[key_b]
            if not isinstance(old_a_raw, torch.Tensor) or not isinstance(old_b_raw, torch.Tensor):
                continue

            try:
                new_a_param = module.lora_A[self.organelle_id].weight
                new_b_param = module.lora_B[self.organelle_id].weight
            except Exception:
                continue

            # Fast path: exact shape match.
            if old_a_raw.shape == new_a_param.shape and old_b_raw.shape == new_b_param.shape:
                with torch.no_grad():
                    new_a_param.copy_(
                        old_a_raw.to(device=new_a_param.device, dtype=new_a_param.dtype)
                    )
                    new_b_param.copy_(
                        old_b_raw.to(device=new_b_param.device, dtype=new_b_param.dtype)
                    )
                continue

            try:
                old_a, _old_a_t = _std_a(old_a_raw.detach().cpu())
                old_b, _old_b_t = _std_b(old_b_raw.detach().cpu())

                new_a_view, new_a_t = _std_a(new_a_param)
                new_b_view, new_b_t = _std_b(new_b_param)
            except Exception:
                continue

            r_old = old_a.shape[0]
            r_new = new_a_view.shape[0]
            if r_old <= 0 or r_new <= 0:
                continue
            if old_b.shape[1] != r_old or new_b_view.shape[1] != r_new:
                continue

            in_overlap = min(old_a.shape[1], new_a_view.shape[1])
            out_overlap = min(old_b.shape[0], new_b_view.shape[0])
            if in_overlap <= 0 or out_overlap <= 0:
                continue

            old_a_f = old_a[:, :in_overlap].to(dtype=torch.float32)
            old_b_f = old_b[:out_overlap, :].to(dtype=torch.float32)
            try:
                a_norm = torch.linalg.norm(old_a_f, dim=1)
                b_norm = torch.linalg.norm(old_b_f, dim=0)
            except Exception:
                continue
            scores = a_norm * b_norm
            if scores.numel() != r_old:
                continue

            keep = min(r_new, r_old)
            try:
                selected = torch.argsort(scores, descending=True)[:keep]
            except Exception:
                selected = torch.arange(keep)

            projected_a = torch.zeros((r_new, new_a_view.shape[1]), dtype=torch.float32)
            projected_b = torch.zeros((new_b_view.shape[0], r_new), dtype=torch.float32)
            for new_idx, old_idx in enumerate(selected.tolist()):
                projected_a[new_idx, :in_overlap] = old_a_f[old_idx, :in_overlap]
                projected_b[:out_overlap, new_idx] = old_b_f[:out_overlap, old_idx]

            if r_new > keep:
                rng = torch.Generator(device="cpu")
                seed_material = f"inherit:{self.organelle_id}:{name}:{r_new}".encode("utf-8")
                digest = hashlib.sha1(seed_material, usedforsecurity=False).hexdigest()
                seed = int(digest[:16], 16) % (2**31 - 1)
                rng.manual_seed(seed)
                projected_a[keep:, :] = (
                    torch.randn(
                        (r_new - keep, new_a_view.shape[1]),
                        dtype=torch.float32,
                        generator=rng,
                    )
                    * 1e-3
                )
                projected_b[:, keep:] = (
                    torch.randn(
                        (new_b_view.shape[0], r_new - keep),
                        dtype=torch.float32,
                        generator=rng,
                    )
                    * 1e-3
                )

            projected_a = projected_a.to(device=new_a_param.device, dtype=new_a_param.dtype)
            projected_b = projected_b.to(device=new_b_param.device, dtype=new_b_param.dtype)
            with torch.no_grad():
                if new_a_t:
                    new_a_param.copy_(projected_a.t())
                else:
                    new_a_param.copy_(projected_a)
                if new_b_t:
                    new_b_param.copy_(projected_b.t())
                else:
                    new_b_param.copy_(projected_b)

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
            try:
                if self.organelle_id not in module.lora_A or self.organelle_id not in module.lora_B:
                    continue
                weight_a = module.lora_A[self.organelle_id].weight.detach().cpu().clone()
                weight_b = module.lora_B[self.organelle_id].weight.detach().cpu().clone()
            except Exception:  # pragma: no cover - defensive against PEFT internals
                continue
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
                adapter = self.organelle_id
                if adapter not in module.lora_A or adapter not in module.lora_B:
                    continue
            except Exception:
                continue
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
            try:
                if self.organelle_id not in module.lora_A or self.organelle_id not in module.lora_B:
                    continue
            except Exception:  # pragma: no cover
                continue
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state:
                incoming_a = state[key_a].to(
                    module.lora_A[self.organelle_id].weight.device,
                    module.lora_A[self.organelle_id].weight.dtype,
                )
                with torch.no_grad():
                    if incoming_a.shape == module.lora_A[self.organelle_id].weight.shape:
                        module.lora_A[self.organelle_id].weight.add_(incoming_a * alpha)
            if key_b in state:
                incoming_b = state[key_b].to(
                    module.lora_B[self.organelle_id].weight.device,
                    module.lora_B[self.organelle_id].weight.dtype,
                )
                with torch.no_grad():
                    if incoming_b.shape == module.lora_B[self.organelle_id].weight.shape:
                        module.lora_B[self.organelle_id].weight.add_(incoming_b * alpha)

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
            try:
                if self.organelle_id not in module.lora_A or self.organelle_id not in module.lora_B:
                    continue
            except Exception:  # pragma: no cover
                continue
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state:
                incoming_a = state[key_a].to(
                    module.lora_A[self.organelle_id].weight.device,
                    module.lora_A[self.organelle_id].weight.dtype,
                )
                with torch.no_grad():
                    if incoming_a.shape == module.lora_A[self.organelle_id].weight.shape:
                        module.lora_A[self.organelle_id].weight.copy_(incoming_a)
            if key_b in state:
                incoming_b = state[key_b].to(
                    module.lora_B[self.organelle_id].weight.device,
                    module.lora_B[self.organelle_id].weight.dtype,
                )
                with torch.no_grad():
                    if incoming_b.shape == module.lora_B[self.organelle_id].weight.shape:
                        module.lora_B[self.organelle_id].weight.copy_(incoming_b)

    def trainable_parameters(self) -> int:
        total = 0
        for module in self.model.modules():
            lora_a_dict = getattr(module, "lora_A", None)
            lora_b_dict = getattr(module, "lora_B", None)
            if lora_a_dict is not None and hasattr(lora_a_dict, "items"):
                for adapter, lora in cast(Any, lora_a_dict).items():
                    if adapter == self.organelle_id and hasattr(lora, "weight"):
                        total += int(lora.weight.numel())
            if lora_b_dict is not None and hasattr(lora_b_dict, "items"):
                for adapter, lora in cast(Any, lora_b_dict).items():
                    if adapter == self.organelle_id and hasattr(lora, "weight"):
                        total += int(lora.weight.numel())
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
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue
            try:
                if self.organelle_id not in lora_a or self.organelle_id not in lora_b:
                    continue
                old_weights[name] = (
                    lora_a[self.organelle_id].weight.detach().clone(),
                    lora_b[self.organelle_id].weight.detach().clone(),
                )
            except Exception:
                continue
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
            old_a, old_b = old_weights[name]
            adapter = self.organelle_id
            try:
                if adapter not in lora_a or adapter not in lora_b:
                    continue
                new_a = lora_a[adapter].weight
                new_b = lora_b[adapter].weight
            except Exception:
                continue
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
                adapter_config = config[self.organelle_id]
                target = getattr(adapter_config, "target_modules", None)
                if target is not None:
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


class BackpropPEFTOrganelle(HebbianPEFTOrganelle):
    """PEFT organelle that learns via per-organelle backprop (online supervised updates)."""

    def __init__(
        self,
        backbone: HFBackbone,
        rank: int,
        context: OrganelleContext,
        activation_bias: float = 0.0,
    ) -> None:
        super().__init__(
            backbone=backbone,
            rank=rank,
            context=context,
            activation_bias=activation_bias,
        )
        self._optimizer: torch.optim.Optimizer | None = None
        replay_size = 32
        try:
            replay_size = int(os.getenv("EVOLORA_BP_REPLAY_SIZE", replay_size) or replay_size)
        except Exception:
            replay_size = 32
        self._replay: deque[tuple[str, str, float]] = deque(maxlen=max(0, replay_size))
        replay_k = 1
        try:
            replay_k = int(os.getenv("EVOLORA_BP_REPLAY_K", replay_k) or replay_k)
        except Exception:
            replay_k = 1
        self._replay_k = max(0, replay_k)

    def _adapter_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for _name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            try:
                if self.organelle_id not in module.lora_A or self.organelle_id not in module.lora_B:
                    continue
                params.append(module.lora_A[self.organelle_id].weight)
                params.append(module.lora_B[self.organelle_id].weight)
            except Exception:
                continue
        seen: set[int] = set()
        unique: list[nn.Parameter] = []
        for param in params:
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            unique.append(param)
        return unique

    def _reset_optimizer(self) -> None:
        self._optimizer = None

    def _ensure_optimizer(self) -> torch.optim.Optimizer | None:
        if self._optimizer is not None:
            return self._optimizer
        params = self._adapter_parameters()
        if not params:
            return None
        lr = 1e-4
        try:
            lr = float(getattr(self.context.hebbian, "learning_rate", lr))
        except Exception:
            lr = 1e-4
        if not math.isfinite(lr) or lr <= 0.0:
            lr = 1e-4
        self._optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        return self._optimizer

    def resize_rank(self, new_rank: int) -> bool:
        changed = super().resize_rank(new_rank)
        if changed:
            self._reset_optimizer()
        return changed

    def inherit_from(self, parent: Organelle) -> None:
        super().inherit_from(parent)
        self._reset_optimizer()

    def update(self, envelope: MessageEnvelope, reward: RewardBreakdown) -> bool:
        completion = envelope.observation.state.get("supervised_completion")
        if completion is None:
            return False
        state = envelope.observation.state
        prompt_val = state.get("task_prompt") or state.get("text") or ""
        prompt = prompt_val if isinstance(prompt_val, str) else str(prompt_val)
        completion_text = str(completion).strip()
        if not completion_text:
            return False

        # Focus learning on failures / low progress to avoid overfitting to a single canonical target.
        try:
            progress = float(reward.task_reward)
        except Exception:
            progress = 0.0
        if not math.isfinite(progress):
            progress = 0.0
        progress = max(0.0, min(1.0, progress))
        weight = 1.0 - progress

        # Ensure the correct adapter is active; never fall back to a different adapter.
        if not self._ensure_active_adapter():
            return False

        optimizer = self._ensure_optimizer()
        if optimizer is None:
            return False

        max_len = int(getattr(self.backbone, "max_length", 256) or 256)
        max_len = max(8, max_len)
        replay_examples: list[tuple[str, str, float]] = []
        try:
            if self._replay_k > 0 and len(self._replay) > 0:
                seed_material = f"bp-replay:{self.organelle_id}:{self.steps}".encode("utf-8")
                seed = int(
                    hashlib.sha1(seed_material, usedforsecurity=False).hexdigest()[:16], 16
                ) % (2**31 - 1)
                rng = random.Random(seed)
                candidates = list(self._replay)
                rng.shuffle(candidates)
                replay_examples = candidates[: self._replay_k]
        except Exception:
            replay_examples = []

        def _format_prompt(text: str) -> str:
            trimmed = str(text or "").rstrip()
            return f"{trimmed}\n" if trimmed else ""

        def _encode_example(
            prompt_text: str, completion_target: str
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
            full_prompt = _format_prompt(prompt_text)
            try:
                prompt_ids = self.tokenizer(
                    full_prompt,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors="pt",
                )["input_ids"][0]
                completion_ids = self.tokenizer(
                    completion_target,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors="pt",
                )["input_ids"][0]
            except Exception:
                return None

            if int(completion_ids.numel()) <= 0:
                return None

            # Reserve some room for completion tokens; truncate prompt first.
            reserve = min(int(completion_ids.numel()), 64)
            reserve = max(1, min(reserve, max_len - 1))
            prompt_budget = max(0, max_len - reserve)
            if int(prompt_ids.numel()) > prompt_budget:
                prompt_ids = prompt_ids[:prompt_budget]
            remaining = max_len - int(prompt_ids.numel())
            if remaining <= 0:
                return None
            if int(completion_ids.numel()) > remaining:
                completion_ids = completion_ids[:remaining]
            if int(completion_ids.numel()) <= 0:
                return None

            input_ids_1d = torch.cat([prompt_ids, completion_ids], dim=0).to(
                device=self.device, dtype=torch.long
            )
            input_ids = input_ids_1d.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            prompt_len = int(prompt_ids.numel())
            if prompt_len > 0:
                labels[:, :prompt_len] = -100
            return input_ids, attention_mask, labels, int(input_ids.shape[1])

        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]] = []
        if weight > 0.0:
            encoded = _encode_example(prompt, completion_text)
            if encoded is not None:
                input_ids, attention_mask, labels, tokens = encoded
                examples.append((input_ids, attention_mask, labels, float(weight), tokens))
        for rp_prompt, rp_completion, rp_progress in replay_examples:
            rp_weight = 0.25 * max(0.1, 1.0 - float(rp_progress))
            if rp_weight <= 0.0:
                continue
            encoded = _encode_example(rp_prompt, rp_completion)
            if encoded is None:
                continue
            input_ids, attention_mask, labels, tokens = encoded
            examples.append((input_ids, attention_mask, labels, float(rp_weight), tokens))

        if not examples:
            try:
                self._replay.append((prompt, completion_text, float(progress)))
            except Exception:
                pass
            return False

        self.model.train()
        total_tokens = 0
        try:
            optimizer.zero_grad(set_to_none=True)
            had_grad = False
            for input_ids, attention_mask, labels, ex_weight, tokens in examples:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    continue
                scaled = loss * float(ex_weight)
                if not torch.isfinite(scaled):
                    continue
                scaled.backward()
                had_grad = True
                total_tokens += int(tokens)
            if not had_grad:
                return False

            max_norm = 1.0
            try:
                max_norm = float(getattr(self.context.hebbian, "max_update_norm", max_norm))
            except Exception:
                max_norm = 1.0
            if math.isfinite(max_norm) and max_norm > 0.0:
                try:
                    torch.nn.utils.clip_grad_norm_(self._adapter_parameters(), max_norm)
                except Exception:
                    pass

            optimizer.step()
            self.step()
        finally:
            # Keep generation deterministic.
            self.model.eval()

        # Compute budget accounting: record a training forward pass plus an estimated backward cost.
        if total_tokens <= 0:
            total_tokens = sum(int(tokens) for *_rest, tokens in examples)
        hidden = int(getattr(self.backbone, "hidden_size", 768) or 768)
        forward_flops = float(total_tokens * hidden * 2)
        envelope.observation.state["update_forward_tokens"] = int(total_tokens)
        envelope.observation.state["update_forward_flops"] = forward_flops
        envelope.observation.state["update_backward_flops"] = float(forward_flops * 2.0)
        try:
            self._replay.append((prompt, completion_text, float(progress)))
        except Exception:
            pass
        return True


__all__ = ["BackpropPEFTOrganelle", "HebbianPEFTOrganelle"]
