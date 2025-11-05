"""Host kernel orchestrating organelles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import time
from typing import Optional

import torch

from symbiont_ecology.config import EcologyConfig, HebbianConfig
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.host.gemma import GemmaBackbone
from symbiont_ecology.interfaces.messages import MessageEnvelope, Observation, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown, RouteEvent
from symbiont_ecology.organelles.base import Organelle, OrganelleContext
from symbiont_ecology.organelles import peft_hebbian
from symbiont_ecology.routing.router import BanditRouter
from symbiont_ecology.utils.ids import short_uid
from symbiont_ecology.utils.torch_utils import resolve_device


@dataclass
class RouteMetrics:
    answer: str
    tokens: int
    latency_ms: float
    prompt_tokens: int
    trainable_params: int
    flops_estimate: float
    memory_gb: float
    active_adapters: dict[str, int]


@dataclass
class HostStepResult:
    envelope: MessageEnvelope
    routes: list[RouteEvent]
    responses: dict[str, RouteMetrics]
    latency_ms: float


@dataclass
class HostKernel:
    config: EcologyConfig
    router: BanditRouter
    ledger: ATPLedger
    device: torch.device = field(init=False)
    backbone: GemmaBackbone = field(init=False)
    organelles: dict[str, Organelle] = field(default_factory=dict)
    assimilation_state: dict[str, torch.Tensor] = field(default_factory=dict)
    assimilation_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.config.host.device)
        self.backbone = GemmaBackbone(self.config.host)

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
    ) -> str:
        if len(self.organelles) >= self.config.organism.max_organelles:
            raise RuntimeError("Organism capacity reached")
        organelle_id: str = short_uid("org")
        context = OrganelleContext(
            organelle_id=organelle_id,
            hebbian=hebbian_config or self.config.hebbian,
            reward_baseline=0.0,
            traces=None,
        )
        organelle_cls = getattr(peft_hebbian, "HebbianPEFTOrganelle")
        organelle = organelle_cls(
            backbone=self.backbone,
            rank=min(rank, self.config.host.max_lora_rank),
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
    ) -> HostStepResult:
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
                lp = torch.tensor(latent_prefix, device=self.device, dtype=latent.dtype).view_as(latent)
                latent = (1.0 - mix) * latent + mix * lp
            except Exception:
                pass
        envelope.observation.state["latent"] = latent.squeeze(0).tolist()
        organelle_pool = (
            {oid: self.organelles[oid] for oid in allowed_organelle_ids if oid in self.organelles}
            if allowed_organelle_ids is not None
            else self.organelles
        )
        routed = self.router.select(organelle_pool, envelope, k=max_routes)
        route_events: list[RouteEvent] = []
        responses: dict[str, RouteMetrics] = {}
        prompt_tokens = self._count_tokens(prompt)
        hidden = getattr(self.backbone, "hidden_size", prompt_tokens)
        for organelle in routed:
            envelope = organelle.forward(envelope)
            answer = envelope.observation.state.get("answer", "")
            trainable = self._trainable_params(organelle)
            adapters = self._active_adapters(organelle)
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
                    tokens=prompt_tokens,
                    latency_ms=0.0,
                )
            )
            flops = float(prompt_tokens * hidden * 2)
            memory_gb = float(prompt_tokens * hidden * 2) / (1024**3)
            responses[organelle.organelle_id] = RouteMetrics(
                answer=answer,
                tokens=prompt_tokens,
                latency_ms=0.0,
                prompt_tokens=prompt_tokens,
                trainable_params=trainable,
                flops_estimate=flops,
                memory_gb=memory_gb,
                active_adapters=adapters,
            )
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
                    trainable_params=metrics.trainable_params,
                    flops_estimate=metrics.flops_estimate,
                    memory_gb=metrics.memory_gb,
                    active_adapters=metrics.active_adapters,
                )
        return HostStepResult(
            envelope=envelope, routes=route_events, responses=responses, latency_ms=latency_ms
        )

    def apply_reward(
        self,
        envelope: MessageEnvelope,
        rewards: dict[str, RewardBreakdown],
    ) -> None:
        for organelle_id, breakdown in rewards.items():
            organelle = self.organelles.get(organelle_id)
            if organelle is None:
                continue
            organelle.update(envelope, breakdown)
            # Mint ATP based on total reward (which already subtracts cost_penalty)
            net_gain = max(0.0, breakdown.total)
            self.ledger.credit(organelle_id, net_gain * self.config.organism.atp_mint_rate)
            self.router.observe(organelle_id, breakdown.total)

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
            return len(tokenizer(prompt)["input_ids"])  # type: ignore[index]
        except Exception:
            return len(prompt.split())

    def _trainable_params(self, organelle: Organelle) -> int:
        if hasattr(organelle, "trainable_parameters"):
            try:
                value = int(organelle.trainable_parameters())  # type: ignore[attr-defined]
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
                data = organelle.active_adapters()  # type: ignore[attr-defined]
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def resize_organelle_rank(self, organelle_id: str, new_rank: int) -> bool:
        organelle = self.organelles.get(organelle_id)
        if organelle is None:
            return False
        if hasattr(organelle, "resize_rank"):
            try:
                changed = bool(organelle.resize_rank(new_rank))  # type: ignore[attr-defined]
            except Exception:
                return False
            if not changed:
                return False
            if hasattr(organelle, "rank"):
                try:
                    organelle.rank = new_rank  # type: ignore[attr-defined]
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
                return int(organelle.estimate_trainable(new_rank))  # type: ignore[attr-defined]
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

    def merge_lora_soup(self, soup: dict[str, float], target_rank: int) -> None:
        soup_state, total_alpha_map = self.build_lora_soup_state(soup, target_rank)
        for key, combined in soup_state.items():
            if key not in self.assimilation_state:
                self.assimilation_state[key] = torch.zeros_like(combined)
                self.assimilation_weights[key] = 0.0
            self.assimilation_state[key].add_(combined)
            self.assimilation_weights[key] = self.assimilation_weights.get(key, 0.0) + float(total_alpha_map.get(key, 0.0))

    def retire_organelle(self, organelle_id: str) -> None:
        self.organelles.pop(organelle_id, None)
        self.router.arms.pop(organelle_id, None)
        self.ledger.accounts.pop(organelle_id, None)
        self.ledger.energy_accounts.pop(organelle_id, None)

    def build_lora_soup_state(self, soup: dict[str, float], target_rank: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Construct a weighted LoRA soup state without committing it to the host.

        Returns a tuple of (state_dict, alpha_sum_per_key).
        """
        contributions: dict[str, list[tuple[float, torch.Tensor]]] = {}
        alpha_sum: dict[str, float] = {}
        for organelle_id, alpha in soup.items():
            organelle = self.organelles.get(organelle_id)
            if organelle is None:
                continue
            snapshot = organelle.export_adapter_state()
            for key, tensor in snapshot.items():
                tensor_cpu = tensor.detach().cpu().clone()
                contributions.setdefault(key, []).append((alpha, tensor_cpu))
                alpha_sum[key] = alpha_sum.get(key, 0.0) + alpha
            account = self.ledger.accounts.get(organelle_id)
            if account is not None:
                self.ledger.charge(organelle_id, account.balance)
        soup_state: dict[str, torch.Tensor] = {}
        for key, parts in contributions.items():
            if not parts:
                continue
            combined = None
            for alpha, tensor in parts:
                combined = tensor * alpha if combined is None else combined + tensor * alpha
            if combined is None:
                continue
            if combined.ndim == 2 and target_rank > 0:
                try:
                    u, s, vh = torch.linalg.svd(combined, full_matrices=False)
                    rank = min(target_rank, s.shape[0])
                    combined = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
                except Exception:
                    pass
            soup_state[key] = combined
        return soup_state, alpha_sum

    def list_organelle_ids(self) -> list[str]:
        return list(self.organelles.keys())

    def get_organelle(self, organelle_id: str) -> Optional[Organelle]:
        return self.organelles.get(organelle_id)

    # ------------------------------------------------------------------
    def export_organelle_adapter(self, organelle_id: str, path: str | 'Path') -> None:
        """Persist an organelle's adapter state to disk via torch.save.

        Creates parent directories as needed.
        """
        import os
        from pathlib import Path as _P
        import torch as _t
        org = self.organelles.get(organelle_id)
        if org is None:
            raise KeyError(f"Unknown organelle {organelle_id}")
        state = org.export_adapter_state()
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _t.save(state, p)

    def import_organelle_adapter(self, organelle_id: str, path: str | 'Path', alpha: float = 1.0) -> None:
        """Load an adapter state from disk and import into the organelle.

        If the organelle does not implement import_adapter_state, this is a no-op.
        """
        from pathlib import Path as _P
        import torch as _t
        org = self.organelles.get(organelle_id)
        if org is None:
            raise KeyError(f"Unknown organelle {organelle_id}")
        p = _P(path)
        state = _t.load(p, map_location="cpu")
        try:
            org.import_adapter_state(state, alpha=alpha)  # type: ignore[attr-defined]
        except Exception:
            pass

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
