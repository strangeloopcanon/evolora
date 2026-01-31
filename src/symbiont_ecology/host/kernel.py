"""Host kernel orchestrating organelles."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

from symbiont_ecology.config import EcologyConfig, HebbianConfig
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.host.backbone import HFBackbone
from symbiont_ecology.interfaces.messages import MessageEnvelope, Observation, Plan
from symbiont_ecology.metrics.telemetry import ComputeBudget, RewardBreakdown, RouteEvent
from symbiont_ecology.organelles import peft_hebbian
from symbiont_ecology.organelles.base import Organelle, OrganelleContext
from symbiont_ecology.routing.router import BanditRouter
from symbiont_ecology.utils.ids import short_uid
from symbiont_ecology.utils.torch_utils import resolve_device


@dataclass
class RouteMetrics:
    answer: str
    tokens: int
    latency_ms: float
    prompt_tokens: int
    generated_tokens: int
    trainable_params: int
    flops_estimate: float
    memory_gb: float
    active_adapters: dict[str, int]
    recurrent_passes: int = 1


@dataclass
class HostStepResult:
    envelope: MessageEnvelope
    routes: list[RouteEvent]
    responses: dict[str, RouteMetrics]
    latency_ms: float


@dataclass(frozen=True)
class _SoupContribution:
    alpha: float
    tensor: torch.Tensor
    role: int | None
    noise: float


@dataclass
class HostKernel:
    config: EcologyConfig
    router: BanditRouter
    ledger: ATPLedger
    compute_budget: ComputeBudget | None = None
    device: torch.device = field(init=False)
    backbone: HFBackbone = field(init=False)
    organelles: dict[str, Organelle] = field(default_factory=dict)
    assimilation_state: dict[str, torch.Tensor] = field(default_factory=dict)
    assimilation_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.config.host.device)
        self.backbone = HFBackbone(self.config.host)

    def freeze_host(self) -> None:
        for parameter in self._iter_host_parameters():
            parameter.requires_grad = False
        # configure energy cap
        self.ledger.configure_energy_cap(self.config.energy.Emax)

    def spawn_organelle(
        self,
        rank: int,
        hebbian_config: HebbianConfig | None = None,
        activation_bias: float = 0.0,
        organelle_id: str | None = None,
    ) -> str:
        if len(self.organelles) >= self.config.organism.max_organelles:
            raise RuntimeError("Organism capacity reached")
        organelle_id = organelle_id or short_uid("org")
        context = OrganelleContext(
            organelle_id=organelle_id,
            hebbian=hebbian_config or self.config.hebbian,
            reward_baseline=0.0,
            progress_baseline=0.0,
            traces=None,
        )
        plasticity = str(os.getenv("EVOLORA_PLASTICITY", "hebbian") or "hebbian").strip().lower()
        organelle_cls: type[peft_hebbian.HebbianPEFTOrganelle]
        if plasticity in {"backprop", "bp", "grad", "gradient"}:
            organelle_cls = peft_hebbian.BackpropPEFTOrganelle
        else:
            organelle_cls = peft_hebbian.HebbianPEFTOrganelle
        min_rank = int(getattr(self.config.host, "min_lora_rank", 1) or 1)
        min_rank = max(1, min_rank)
        rank = max(min_rank, min(int(rank), int(self.config.host.max_lora_rank)))
        organelle = organelle_cls(
            backbone=self.backbone,
            rank=rank,
            context=context,
            activation_bias=activation_bias,
        )
        self.organelles[organelle_id] = organelle
        self.ledger.ensure(organelle_id, self.config.organism.initial_atp)
        self.ledger.ensure_energy(organelle_id, self.config.energy.Emax * 0.5)
        self.router.update_bandit(organelle_id, prior=0.5)
        self._apply_assimilation_seed(organelle)
        return organelle_id

    def step(
        self,
        prompt: str,
        intent: str,
        constraints: Sequence[str] | None = None,
        max_routes: int = 3,
        allowed_organelle_ids: Sequence[str] | None = None,
        latent_prefix: list[float] | None = None,
        latent_mix: float | None = None,
        recurrent_passes: int | None = None,
    ) -> HostStepResult:
        training_intent = self._is_training_intent(intent)
        observation = Observation(state={"text": prompt})
        plan = Plan(steps=[], confidence=0.1)
        envelope = MessageEnvelope(
            observation=observation,
            intent=self.router.intent_factory(intent, constraints or []),
            plan=plan,
        )
        t0 = time.time()
        latent = self.backbone.encode_text([prompt], device=self.device)
        # Optional C2C latent prefix blending
        if latent_prefix is not None:
            try:
                mix = float(latent_mix) if latent_mix is not None else 0.5
                mix = max(0.0, min(1.0, mix))
                lp = torch.tensor(latent_prefix, device=self.device, dtype=latent.dtype).view_as(
                    latent
                )
                latent = (1.0 - mix) * latent + mix * lp
            except Exception:
                pass
        envelope.observation.state["latent"] = latent.squeeze(0).tolist()
        prompt_tokens_fallback = self._count_tokens(prompt)
        hidden = int(
            getattr(self.backbone, "hidden_size", prompt_tokens_fallback) or prompt_tokens_fallback
        )
        if self.compute_budget is not None:
            # Account for the backbone latent extraction pass (no generation tokens).
            try:
                flops = float(prompt_tokens_fallback * hidden * 2)
                self.compute_budget.record_forward(prompt_tokens_fallback, 0, flops, training=False)
            except Exception:
                pass
        organelle_pool = (
            {oid: self.organelles[oid] for oid in allowed_organelle_ids if oid in self.organelles}
            if allowed_organelle_ids is not None
            else self.organelles
        )
        routed = self.router.select(organelle_pool, envelope, k=max_routes)
        route_events: list[RouteEvent] = []
        responses: dict[str, RouteMetrics] = {}
        prompt_tokens = prompt_tokens_fallback
        for organelle in routed:
            env = envelope.model_copy(deep=True)
            base_prompt = prompt
            history: list[str] = []
            passes = recurrent_passes or self._default_recurrence_passes(intent)
            passes = max(1, passes)
            prompt_tokens_total = 0
            generated_tokens_total = 0
            for pass_idx in range(passes):
                working_text = self._format_recurrence_prompt(
                    base_prompt, history, pass_idx + 1, passes
                )
                env.observation.state["text"] = working_text
                env.observation.state["recurrent_pass"] = pass_idx + 1
                env.observation.state["recurrent_total"] = passes
                env.observation.state["recurrent_history"] = list(history)
                env = organelle.forward(env)
                try:
                    pt = env.observation.state.get("prompt_tokens")
                    if isinstance(pt, int) and pt > 0:
                        prompt_tokens_total += pt
                    gt = env.observation.state.get("generated_tokens")
                    if isinstance(gt, int) and gt >= 0:
                        generated_tokens_total += gt
                except Exception:
                    pass
                answer_step = env.observation.state.get("answer", "")
                if answer_step:
                    history.append(answer_step)
            envelope = env
            answer = envelope.observation.state.get("answer", "")
            trainable = self._trainable_params(organelle)
            adapters = self._active_adapters(organelle)
            if prompt_tokens_total <= 0:
                prompt_tokens_total = prompt_tokens
            total_tokens = prompt_tokens_total + max(0, generated_tokens_total)
            route_events.append(
                RouteEvent(
                    organelle_id=organelle.organelle_id,
                    reward=0.0,
                    novelty=0.0,
                    competence_gain=0.0,
                    helper_credit=0.0,
                    risk_penalty=0.0,
                    cost_penalty=0.0,
                    atp_delta=0.0,
                    tokens=total_tokens,
                    latency_ms=0.0,
                )
            )
            flops = float(total_tokens * hidden * 2)
            memory_gb = float(total_tokens * hidden * 2) / (1024**3)
            responses[organelle.organelle_id] = RouteMetrics(
                answer=answer,
                tokens=total_tokens,
                latency_ms=0.0,
                prompt_tokens=prompt_tokens_total,
                generated_tokens=max(0, generated_tokens_total),
                trainable_params=trainable,
                flops_estimate=flops,
                memory_gb=memory_gb,
                active_adapters=adapters,
                recurrent_passes=passes,
            )
            if self.compute_budget is not None:
                try:
                    self.compute_budget.record_forward(
                        prompt_tokens_total,
                        max(0, generated_tokens_total),
                        flops,
                        training=training_intent,
                    )
                except Exception:
                    pass
        latency_ms = (time.time() - t0) * 1000.0
        for evt in route_events:
            evt.latency_ms = latency_ms
            metrics = responses.get(evt.organelle_id)
            if metrics is not None:
                responses[evt.organelle_id] = RouteMetrics(
                    answer=metrics.answer,
                    tokens=metrics.tokens,
                    latency_ms=latency_ms,
                    prompt_tokens=metrics.prompt_tokens,
                    generated_tokens=metrics.generated_tokens,
                    trainable_params=metrics.trainable_params,
                    flops_estimate=metrics.flops_estimate,
                    memory_gb=metrics.memory_gb,
                    active_adapters=metrics.active_adapters,
                    recurrent_passes=metrics.recurrent_passes,
                )
        return HostStepResult(
            envelope=envelope, routes=route_events, responses=responses, latency_ms=latency_ms
        )

    def apply_reward(
        self,
        envelope: MessageEnvelope,
        rewards: dict[str, RewardBreakdown],
    ) -> None:
        intent_goal = ""
        if getattr(envelope, "intent", None) is not None:
            intent_goal = str(getattr(envelope.intent, "goal", "") or "")
        training_intent = self._is_training_intent(intent_goal)
        for organelle_id, breakdown in rewards.items():
            organelle = self.organelles.get(organelle_id)
            if organelle is None:
                continue
            did_update = False
            try:
                did_update = bool(organelle.update(envelope, breakdown))
            except Exception:
                did_update = False
            if did_update and self.compute_budget is not None:
                try:
                    state = {}
                    try:
                        state = envelope.observation.state
                    except Exception:
                        state = {}
                    update_tokens = state.get("update_forward_tokens")
                    update_forward_flops = state.get("update_forward_flops")
                    if isinstance(update_tokens, int) and isinstance(
                        update_forward_flops, (int, float)
                    ):
                        self.compute_budget.record_forward(
                            prompt_tokens=update_tokens,
                            generated_tokens=0,
                            flops=float(update_forward_flops),
                            training=training_intent,
                        )
                    backward_flops = state.get("update_backward_flops")
                    if isinstance(backward_flops, (int, float)) and float(backward_flops) > 0.0:
                        self.compute_budget.record_hebbian_update(
                            estimated_flops=float(backward_flops), training=training_intent
                        )
                    else:
                        params = self._trainable_params(organelle)
                        # Estimate: 2 FLOPs per param (scale + add)
                        hebbian_flops = float(params * 2)
                        self.compute_budget.record_hebbian_update(
                            estimated_flops=hebbian_flops, training=training_intent
                        )
                except Exception:
                    pass
            # Mint ATP based on total reward (which already subtracts cost_penalty)
            net_gain = max(0.0, breakdown.total)
            self.ledger.credit(organelle_id, net_gain * self.config.organism.atp_mint_rate)
            self.router.observe(organelle_id, breakdown.total)

    def _default_recurrence_passes(self, intent: str) -> int:
        host_cfg = self.config.host
        if not getattr(host_cfg, "recurrence_enabled", False):
            return 1
        intent_lower = intent.lower()
        eval_markers = (
            "evaluation",
            "assimilation",
            "holdout",
            "team probe",
            "team holdout",
            "colony infer",
        )
        if any(marker in intent_lower for marker in eval_markers):
            passes = getattr(host_cfg, "recurrence_eval_passes", 1)
        else:
            passes = getattr(host_cfg, "recurrence_train_passes", 1)
        return max(1, int(passes))

    @staticmethod
    def _is_training_intent(intent: str) -> bool:
        intent_key = str(intent).strip().lower()
        return intent_key in {"solve task", "team probe", "team handoff", "team episode"}

    def _format_recurrence_prompt(
        self,
        base_prompt: str,
        history: list[str],
        pass_idx: int,
        total_passes: int,
    ) -> str:
        if not history:
            return base_prompt
        template = getattr(self.config.host, "recurrence_history_template", "")
        scratch = "\n".join([f"[Pass {i + 1}] {entry}" for i, entry in enumerate(history)])
        if template:
            try:
                suffix = template.format(
                    history=scratch, pass_idx=pass_idx, total_passes=total_passes
                )
            except Exception:
                suffix = f"Previous passes:\n{scratch}\nRefine your answer (pass {pass_idx}/{total_passes})."
        else:
            suffix = (
                f"Previous passes:\n{scratch}\nRefine your answer (pass {pass_idx}/{total_passes})."
            )
        return f"{base_prompt}\n\n{suffix}".strip()

    def _compute_energy_cost(self, prompt: str, organelle: Organelle) -> float:
        base = self.config.organism.atp_burn_per_call
        token_cost = max(1, len(prompt.split())) * 0.02
        rank_penalty = 0.0
        adapter = getattr(organelle, "adapter", None)
        if adapter is not None and hasattr(adapter, "rank"):
            rank_penalty = float(getattr(adapter, "rank", 1)) * 0.05
        return float(base + token_cost + rank_penalty)

    def _count_tokens(self, prompt: str) -> int:
        tokenizer = getattr(self.backbone, "tokenizer", None)
        if tokenizer is None:
            return len(prompt.split())
        try:
            return len(tokenizer(prompt)["input_ids"])
        except Exception:
            return len(prompt.split())

    def _trainable_params(self, organelle: Organelle) -> int:
        if hasattr(organelle, "trainable_parameters"):
            try:
                value = int(organelle.trainable_parameters())
                return max(value, 0)
            except Exception:
                pass
        total = 0
        adapter = getattr(organelle, "adapter", None)
        if adapter is not None and hasattr(adapter, "parameters"):
            for param in adapter.parameters():
                if param.requires_grad:
                    total += param.numel()
        return total

    def _active_adapters(self, organelle: Organelle) -> dict[str, int]:
        if hasattr(organelle, "active_adapters"):
            try:
                data = organelle.active_adapters()
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def resize_organelle_rank(self, organelle_id: str, new_rank: int) -> bool:
        organelle = self.organelles.get(organelle_id)
        if organelle is None:
            return False
        min_rank = int(getattr(self.config.host, "min_lora_rank", 1) or 1)
        min_rank = max(1, min_rank)
        new_rank = max(min_rank, min(int(new_rank), int(self.config.host.max_lora_rank)))
        if hasattr(organelle, "resize_rank"):
            try:
                changed = bool(organelle.resize_rank(new_rank))
            except Exception:
                return False
            if not changed:
                return False
            if hasattr(organelle, "rank"):
                try:
                    organelle.rank = new_rank
                except Exception:
                    pass
            return True
        return False

    def total_backbone_params(self) -> int:
        if not hasattr(self, "_backbone_param_cache"):
            self._backbone_param_cache = sum(
                int(param.numel()) for param in self._iter_host_parameters()
            )
        return int(self._backbone_param_cache)

    def total_trainable_parameters(self) -> int:
        return sum(self._trainable_params(org) for org in self.organelles.values())

    def estimate_trainable(self, organelle: Organelle, new_rank: int) -> int:
        if hasattr(organelle, "estimate_trainable"):
            try:
                return int(organelle.estimate_trainable(new_rank))
            except Exception:
                pass
        current = max(self._trainable_params(organelle), 1)
        base_rank = getattr(organelle, "rank", 1) or 1
        return int(current * (max(new_rank, 1) / max(int(base_rank), 1)))

    def _iter_host_parameters(self):
        backbone = self.backbone
        if hasattr(backbone, "parameters"):
            yield from backbone.parameters()
        if hasattr(backbone, "model") and hasattr(backbone.model, "parameters"):
            yield from backbone.model.parameters()

    def merge_organelle_into_host(self, organelle_id: str, alpha: float = 1.0) -> None:
        organelle = self.organelles.get(organelle_id)
        if organelle is None:
            raise KeyError(f"Unknown organelle {organelle_id}")
        snapshot = organelle.export_adapter_state()
        if snapshot:
            alpha = float(max(alpha, 0.0))
            for key, tensor in snapshot.items():
                tensor_cpu = tensor.detach().cpu().clone()
                if key not in self.assimilation_state:
                    self.assimilation_state[key] = torch.zeros_like(tensor_cpu)
                    self.assimilation_weights[key] = 0.0
                self.assimilation_state[key].add_(tensor_cpu * alpha)
                self.assimilation_weights[key] = self.assimilation_weights.get(key, 0.0) + alpha
        self.ledger.charge(organelle_id, self.ledger.accounts[organelle_id].balance)

    def merge_lora_soup(
        self,
        soup: dict[str, float],
        target_rank: int,
        *,
        roles: dict[str, int] | None = None,
        mode: str | None = None,
        mutation_meta: dict[str, dict[str, object]] | None = None,
    ) -> None:
        soup_state, total_alpha_map = self.build_lora_soup_state(
            soup,
            target_rank,
            roles=roles,
            mode=mode,
            mutation_meta=mutation_meta,
        )
        for key, combined in soup_state.items():
            if key not in self.assimilation_state:
                self.assimilation_state[key] = torch.zeros_like(combined)
                self.assimilation_weights[key] = 0.0
            self.assimilation_state[key].add_(combined)
            self.assimilation_weights[key] = self.assimilation_weights.get(key, 0.0) + float(
                total_alpha_map.get(key, 0.0)
            )

    def retire_organelle(self, organelle_id: str) -> None:
        self.organelles.pop(organelle_id, None)
        self._delete_backbone_adapter(organelle_id)
        self.router.arms.pop(organelle_id, None)
        self.ledger.accounts.pop(organelle_id, None)
        self.ledger.energy_accounts.pop(organelle_id, None)

    def _delete_backbone_adapter(self, adapter_name: str) -> None:
        """Best-effort removal of a PEFT adapter from the shared backbone model.

        Long runs can create and retire many organelles. When organelles share a single PEFT-wrapped
        backbone, failing to delete retired adapters can lead to unbounded memory growth.
        """
        model = getattr(self.backbone, "model", None)
        delete_adapter = getattr(model, "delete_adapter", None)
        if not callable(delete_adapter):
            return

        fallback = next(iter(self.organelles.keys()), None)
        set_adapter = getattr(model, "set_adapter", None)
        if callable(set_adapter) and fallback:
            try:
                set_adapter(fallback)
            except Exception:
                pass

        try:
            delete_adapter(adapter_name)
        except Exception:
            pass

    def build_lora_soup_state(
        self,
        soup: dict[str, float],
        target_rank: int,
        *,
        roles: dict[str, int] | None = None,
        mode: str | None = None,
        mutation_meta: dict[str, dict[str, object]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Construct a weighted LoRA soup state without committing it to the host.

        Returns a tuple of (state_dict, alpha_sum_per_key).
        """
        contributions: dict[str, list[_SoupContribution]] = {}
        normalized_roles: dict[str, int] | None = None
        min_rank = int(getattr(self.config.host, "min_lora_rank", 1) or 1)
        min_rank = max(1, min_rank)
        if roles:
            # ensure only include roles for known members
            normalized_roles = {str(k): int(v) for k, v in roles.items() if isinstance(v, int)}
        alpha_sum: dict[str, float] = {}
        for organelle_id, alpha in soup.items():
            organelle = self.organelles.get(organelle_id)
            if organelle is None:
                continue
            snapshot = organelle.export_adapter_state()
            meta: dict[str, object] = {}
            if mutation_meta:
                meta = mutation_meta.get(organelle_id, {})

            dropout_raw = meta.get("dropout", [])
            dropout_items = dropout_raw if isinstance(dropout_raw, list) else []
            dropout_patterns = {
                str(item) for item in dropout_items if isinstance(item, str) and item
            }

            duplication_raw = meta.get("duplication", {})
            duplication_items = duplication_raw if isinstance(duplication_raw, dict) else {}
            duplication_map = {
                str(key): float(value)
                for key, value in duplication_items.items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }

            rank_noise_raw = meta.get("rank_noise", {})
            rank_noise_items = rank_noise_raw if isinstance(rank_noise_raw, dict) else {}
            rank_noise_map = {
                str(key): float(value)
                for key, value in rank_noise_items.items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }
            for key, tensor in snapshot.items():
                tensor_cpu = tensor.detach().cpu().clone()
                if dropout_patterns and any(pattern in key for pattern in dropout_patterns):
                    continue
                scale = 1.0
                for pattern, factor in duplication_map.items():
                    if pattern in key:
                        scale += max(0.0, factor)
                role_value = None
                if normalized_roles and organelle_id in normalized_roles:
                    role_value = normalized_roles[organelle_id]
                noise = 0.0
                for pattern, delta in rank_noise_map.items():
                    if pattern in key:
                        noise += float(delta)
                contribution = _SoupContribution(
                    alpha=float(alpha) * scale,
                    tensor=tensor_cpu,
                    role=role_value,
                    noise=noise,
                )
                contributions.setdefault(key, []).append(contribution)
                alpha_sum[key] = alpha_sum.get(key, 0.0) + contribution.alpha
            account = self.ledger.accounts.get(organelle_id)
            if account is not None:
                self.ledger.charge(organelle_id, account.balance)
        soup_state: dict[str, torch.Tensor] = {}
        for key, parts in contributions.items():
            if not parts:
                continue
            use_block = (
                mode == "block"
                and normalized_roles is not None
                and all(part.role is not None for part in parts)
            )
            noise_total = sum(part.noise for part in parts)
            effective_rank = target_rank
            if noise_total:
                effective_rank = int(round(target_rank + noise_total))
            effective_rank = max(min_rank, min(self.config.host.max_lora_rank, effective_rank))
            if use_block:
                try:
                    combined = self._combine_block_diagonal(parts, effective_rank)
                except Exception:
                    combined = None
            else:
                combined = None
                for part in parts:
                    alpha_val = part.alpha
                    tensor = part.tensor
                    combined = (
                        tensor * alpha_val if combined is None else combined + tensor * alpha_val
                    )
            if combined is None:
                continue
            if combined.ndim == 2 and effective_rank > 0 and mode != "block":
                try:
                    pre_norm = float(combined.norm().item())
                    u, s, vh = torch.linalg.svd(combined, full_matrices=False)
                    rank = min(effective_rank, s.shape[0])
                    combined = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
                    if rank < s.shape[0]:
                        post_norm = float(combined.norm().item())
                        if (
                            pre_norm > 0.0
                            and post_norm > 0.0
                            and math.isfinite(pre_norm)
                            and math.isfinite(post_norm)
                        ):
                            scale = pre_norm / post_norm
                            scale = max(0.5, min(2.0, scale))
                            combined = combined * float(scale)
                except Exception:
                    pass
            soup_state[key] = combined
        return soup_state, alpha_sum

    @staticmethod
    def _combine_block_diagonal(parts: list[_SoupContribution], target_rank: int) -> torch.Tensor:
        ordered = sorted(parts, key=lambda item: item.role if item.role is not None else -1)
        if not ordered:
            raise ValueError("No parts to combine")
        base = ordered[0].tensor
        if base.ndim != 2:
            raise ValueError("Block-diagonal merge expects 2D tensors")
        axis = 0 if base.shape[0] <= base.shape[1] else 1
        other_dim = base.shape[1 - axis]
        scaled: list[torch.Tensor] = []
        for part in ordered:
            tensor = part.tensor
            if tensor.ndim != 2:
                raise ValueError("Incompatible tensor rank for block merge")
            if tensor.shape[1 - axis] != other_dim:
                raise ValueError("Mismatched dimensions for block merge")
            alpha_val = part.alpha
            scaled.append(tensor * alpha_val)
        combined = torch.cat(scaled, dim=axis)
        if axis == 0:
            combined = combined[: target_rank if target_rank > 0 else combined.shape[0], :]
        else:
            combined = combined[:, : target_rank if target_rank > 0 else combined.shape[1]]
        return combined

    def list_organelle_ids(self) -> list[str]:
        return list(self.organelles.keys())

    def get_organelle(self, organelle_id: str) -> Optional[Organelle]:
        return self.organelles.get(organelle_id)

    def measure_adapter_magnitude(self, organelle_id: str) -> float:
        organelle = self.organelles.get(organelle_id)
        if organelle is None:
            return 0.0
        # If the organelle exposes a direct measurement, use it.
        if hasattr(organelle, "measure_magnitude"):
            try:
                return float(organelle.measure_magnitude())
            except Exception:
                pass
        snapshot = organelle.export_adapter_state()
        if not snapshot:
            return 0.0
        total_mag = 0.0
        count = 0
        for key, tensor in snapshot.items():
            if "lora_B" in key:
                try:
                    total_mag += float(tensor.abs().max().item())
                    count += 1
                except Exception:
                    pass
        return total_mag / max(1, count)

    # ------------------------------------------------------------------
    def export_organelle_adapter(self, organelle_id: str, path: str | Path) -> None:
        """Persist an organelle's adapter state to disk via safetensors.

        Creates parent directories as needed.
        """
        org = self.organelles.get(organelle_id)
        if org is None:
            raise KeyError(f"Unknown organelle {organelle_id}")
        state = org.export_adapter_state()
        if not state:
            raise ValueError(f"Organelle {organelle_id} exported an empty adapter state")
        tensor_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(key, str):
                raise TypeError("Adapter state keys must be strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError("Adapter state values must be torch.Tensors")
            tensor_state[key] = value.detach().cpu().contiguous()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensor_state, str(p))

    def import_organelle_adapter(
        self, organelle_id: str, path: str | Path, alpha: float = 1.0
    ) -> None:
        """Load an adapter state from disk and import into the organelle.

        If the organelle does not implement import_adapter_state, this is a no-op.
        """
        org = self.organelles.get(organelle_id)
        if org is None:
            raise KeyError(f"Unknown organelle {organelle_id}")
        p = Path(path)
        state = load_file(str(p))
        try:
            org.import_adapter_state(state, alpha=alpha)
        except Exception:
            pass

    def load_organelle_adapter(self, organelle_id: str, path: str | Path) -> None:
        """Load an adapter state from disk, preferring exact replacement.

        Some organelles implement a dedicated `load_adapter_state(state)` which replaces
        the current adapter weights. When unavailable, this falls back to
        `import_adapter_state(state, alpha=1.0)`.
        """
        org = self.organelles.get(organelle_id)
        if org is None:
            raise KeyError(f"Unknown organelle {organelle_id}")
        p = Path(path)
        state = load_file(str(p))
        loader = getattr(org, "load_adapter_state", None)
        if callable(loader):
            loader(state)
            return
        importer = getattr(org, "import_adapter_state", None)
        if callable(importer):
            importer(state, alpha=1.0)
            return
        raise AttributeError(f"Organelle {organelle_id} does not support adapter loading")

    # ------------------------------------------------------------------
    def _apply_assimilation_seed(self, organelle: Organelle) -> None:
        scale = getattr(self.config.assimilation_tuning, "seed_scale", 0.0)
        if scale <= 0.0 or not self.assimilation_state:
            return
        try:
            avg_state: dict[str, torch.Tensor] = {}
            for key, tensor in self.assimilation_state.items():
                weight = max(self.assimilation_weights.get(key, 0.0), 1e-6)
                avg_state[key] = (tensor / weight).detach().clone()
            organelle.import_adapter_state(avg_state, alpha=scale)
        except NotImplementedError:
            return
        except AttributeError:
            return


__all__ = ["HostKernel", "HostStepResult", "RouteMetrics"]
