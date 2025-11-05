"""Configuration models for the symbiotic ecology."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HebbianConfig(BaseModel):
    learning_rate: float = Field(1e-3, ge=1e-6, le=1e-1)
    trace_decay: float = Field(0.95, ge=0.0, le=1.0)
    reward_baseline_decay: float = Field(0.99, ge=0.0, le=1.0)
    max_update_norm: float = Field(1.0, ge=1e-3)


class HostConfig(BaseModel):
    backbone_model: str = Field("google/gemma-3-270m-it")
    tokenizer: Optional[str] = None
    revision: Optional[str] = None
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    device: str = Field("auto", description="torch device string")
    max_lora_rank: int = Field(8, ge=1)
    cache_dir: Optional[Path] = None
    max_sequence_length: int = Field(256, ge=8)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    gen_max_new_tokens: int = Field(48, ge=1, le=512, description="Max new tokens to generate per answer")


class OrganismConfig(BaseModel):
    max_organelles: int = Field(32, ge=1)
    initial_atp: float = Field(100.0, ge=0.0)
    atp_mint_rate: float = Field(1.0, ge=0.0)
    atp_burn_per_call: float = Field(0.1, ge=0.0)


class EvolutionConfig(BaseModel):
    population_size: int = Field(64, ge=1)
    mutation_rate: float = Field(0.2, ge=0.0, le=1.0)
    assimilation_threshold: float = Field(0.02)
    assimilation_p_value: float = Field(0.01)
    niching_bins: int = Field(8, ge=1)
    max_merges_per_gen: int = Field(2, ge=0)
    min_population: int = Field(3, ge=0)
    max_population: int = Field(16, ge=1)


class GridConfig(BaseModel):
    families: list[str] = Field(
        default_factory=lambda: [
            "math",
            "json_repair",
            "string.sort",
            "word.count",
            "logic.bool",
            "math.sequence",
        ]
    )
    depths: list[str] = Field(default_factory=lambda: ["short", "medium", "long"])


class ControllerConfig(BaseModel):
    tau: float = Field(0.5, ge=0.0, le=1.0)
    beta: float = Field(0.2, ge=0.0, le=1.0)
    eta: float = Field(0.5, ge=0.0, le=2.0)


class PricingConfig(BaseModel):
    base: float = Field(1.0, ge=0.0)
    k: float = Field(1.5, ge=0.0)
    min: float = Field(0.3, ge=0.0)
    max: float = Field(2.0, ge=0.0)


class CurriculumConfig(BaseModel):
    lp_mix: float = Field(
        0.0, ge=0.0, le=1.0, description="Probability to route by learning progress instead of ROI bandit"
    )
    lp_alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="Smoothing for learning progress EMA per cell"
    )
    alp_auto_mix: bool = Field(False, description="Automatically tune lp_mix based on learning progress dispersion")
    lp_mix_min: float = Field(0.05, ge=0.0, le=1.0)
    lp_mix_max: float = Field(0.6, ge=0.0, le=1.0)
    lp_window: int = Field(5, ge=1)


class EnergyConfig(BaseModel):
    Emax: float = Field(5.0, ge=0.0)
    m: float = Field(1.0, ge=0.0)
    alpha: float = Field(1e-14, ge=0.0)
    beta: float = Field(0.02, ge=0.0)
    gamma: float = Field(0.001, ge=0.0)
    lambda_p: float = Field(1e-7, ge=0.0)
    bankruptcy_grace: int = Field(3, ge=1)
    cost_scale: float = Field(1.0, ge=0.0, le=1.0)


class CanaryConfig(BaseModel):
    q_min: float = Field(0.95, ge=0.0, le=1.0)


class PopulationStrategyConfig(BaseModel):
    mu: int = Field(4, ge=1)
    lambda_: int = Field(12, ge=1, alias="lambda")
    max_population: int = Field(16, ge=1)


class AssimilationTuningConfig(BaseModel):
    per_cell_interval: int = Field(5, ge=1)
    max_merges_per_cell: int = Field(1, ge=0)
    seed_scale: float = Field(0.2, ge=0.0, le=1.0)
    soup_size: int = Field(3, ge=2)
    hf_prompts: list[str] = Field(default_factory=list)
    probe_max_other_cells: int | None = Field(
        None,
        ge=0,
        description="Maximum number of non-focus cells to probe during global checks (None = all).",
    )
    probe_required_passes: int | None = Field(
        None,
        ge=1,
        description="Minimum successful probe count (including focus cell) required to merge (None = all).",
    )
    energy_floor: float = Field(
        0.0,
        ge=0.0,
        description="Minimum energy balance to attempt assimilation; 0 disables automatic top-ups.",
    )
    energy_floor_roi: float = Field(
        0.0,
        ge=0.0,
        description="ROI threshold that must be met to receive an energy floor top-up.",
    )
    energy_floor_base: float = Field(
        0.6,
        ge=0.0,
        description="Meta-evolved baseline for the assimilation energy floor.",
    )
    energy_floor_roi_base: float = Field(
        0.9,
        ge=0.0,
        description="Meta-evolved baseline ROI requirement for energy floor eligibility.",
    )
    holdout_tasks_path: Optional[Path] = Field(
        None,
        description="Optional JSONL of holdout tasks evaluated before merges (falls back to evaluation tasks).",
    )
    holdout_sample_size: int = Field(
        4,
        ge=1,
        description="Number of holdout tasks sampled per assimilation gate.",
    )
    holdout_margin: float = Field(
        0.05,
        ge=0.0,
        description="Minimum ROI improvement on holdout tasks required to accept a merge.",
    )
    min_window: int = Field(
        4,
        ge=2,
        description="Minimum even window length of recent scores used for uplift testing.",
    )
    window_step: int = Field(
        2,
        ge=2,
        description="Step size when expanding the score window for assimilation tests.",
    )
    adaptive_decay: float = Field(
        0.85,
        ge=0.5,
        le=1.0,
        description="Multiplicative decay applied to uplift threshold when merges keep failing.",
    )
    adaptive_floor: float = Field(
        0.0005,
        ge=0.0,
        description="Floor for the adaptive uplift threshold.",
    )
    holdout_max_retries: int = Field(
        1,
        ge=0,
        description="Number of additional holdout retries before rejecting a merge.",
    )
    holdout_margin_step: float = Field(
        0.02,
        ge=0.0,
        description="Margin reduction applied on each holdout retry.",
    )
    bootstrap_uplift_enabled: bool = Field(
        False, description="Use bootstrap/permutation uplift test instead of z-test"
    )
    bootstrap_samples: int = Field(200, ge=10)
    permutation_samples: int = Field(200, ge=10)
    min_uplift_samples: int = Field(2, ge=1)
    dr_enabled: bool = Field(
        False, description="Use stratified doubly-robust uplift estimator when metadata is available."
    )
    dr_strata: list[str] = Field(
        default_factory=lambda: ["family", "depth"],
        description="Task metadata fields used for stratification during DR uplift estimation.",
    )
    dr_min_stratum_size: int = Field(2, ge=1, description="Minimum paired samples per stratum to contribute to uplift")
    dr_min_power: float = Field(0.2, ge=0.0, le=1.0, description="Minimum statistical power required when DR estimator is used")
    merge_audit_enabled: bool = Field(False)
    energy_topup_roi_bonus: float = Field(
        0.0,
        ge=0.0,
        description="Amount subtracted from the ROI threshold when deciding on energy top-ups.",
    )
    gating_snapshot_limit: int = Field(
        48,
        ge=1,
        description="Maximum number of assimilation gating/attempt samples retained per run.",
    )
    # Offspring merge mode -------------------------------------------------
    merge_mode: str = Field(
        "strict",
        description="strict | offspring | hybrid: offspring allows energy-eligible trial children; hybrid tries strict then offspring",
    )
    trial_offspring_enabled: bool = Field(False)
    trial_per_gen_cap: int = Field(2, ge=0)
    trial_stipend: float = Field(0.5, ge=0.0)
    trial_probation_gens: int = Field(5, ge=1)
    trial_promote_margin: float = Field(0.02, ge=0.0)
    trial_min_power: float = Field(0.1, ge=0.0, le=1.0)
    merge_method: str = Field(
        "naive",
        description="Merging method for strict path: naive | fisher_svd",
    )
    # Colonies (optional)
    colonies_enabled: bool = Field(False)
    colony_synergy_delta: float = Field(0.1, ge=0.0)
    colony_variance_improve: float = Field(0.2, ge=0.0)
    colony_windows: int = Field(3, ge=1)
    colony_review_interval: int = Field(6, ge=1)
    colony_required_passes: int = Field(2, ge=1)
    colony_max_failures: int = Field(2, ge=1)
    # Colony adaptation (size & bandwidth)
    colony_min_size: int = Field(2, ge=1)
    colony_max_size: int = Field(3, ge=1)
    colony_bandwidth_frac: float = Field(0.02, ge=0.0, le=1.0, description="Fraction of colony pot usable for comms per generation")
    colony_post_cap: int = Field(2, ge=0)
    colony_read_cap: int = Field(3, ge=0)
    tau_relief_window: int = Field(12, ge=1)
    tau_relief_step: float = Field(0.01, ge=0.0)
    tau_relief_max: float = Field(0.15, ge=0.0)
    roi_relief_window: int = Field(8, ge=1)
    roi_relief_step: float = Field(0.05, ge=0.0)
    roi_relief_max: float = Field(0.5, ge=0.0)
    # Window auto-tune
    window_autotune: bool = Field(False, description="Auto-adjust assimilation evidence window to reduce insufficient_scores")
    min_window_min: int = Field(6, ge=2)
    # Team probe knobs (colony focus)
    team_probe_per_gen: int = Field(2, ge=0, description="How many team probe pairs to evaluate per generation")
    team_min_tasks: int = Field(8, ge=1, description="Minimum tasks required for team promotion CI gate")
    team_routing_probe_per_gen: int = Field(2, ge=0, description="How many router co-routing probes to run per generation")
    # Team router/composition knobs
    team_router_enabled: bool = Field(False, description="If true, a fraction of tasks may be routed to 2-member teams")
    team_vote_enabled: bool = Field(True, description="If true, select best-of-two answer (majority vote for 2)")
    team_handoff_enabled: bool = Field(False, description="If true, allow solver→checker single revise step (not yet implemented)")
    team_handoff_cap_per_gen: int = Field(4, ge=0, description="Max handoff revisions allowed per generation")
    team_handoff_cost: float = Field(0.05, ge=0.0, description="Energy cost charged to the checker for a handoff revision")
    team_max_routes_per_gen: int = Field(8, ge=0, description="Max team episodes per generation across all pairs")
    team_min_power: float = Field(0.2, ge=0.0, le=1.0, description="Minimum power proxy required for team promotions")
    team_holdout_margin: float | None = Field(None, description="Team-specific holdout margin; falls back to holdout_margin if None")
    promotion_target_rate: float = Field(0.2, ge=0.0, le=1.0, description="Target promotions per generation (normalized)")
    promotion_adjust_step: float = Field(0.002, ge=0.0, le=0.1, description="Step size to adjust team thresholds toward target")
    # Optional smoothing for controller
    promotion_ema_alpha: float = Field(0.3, ge=0.0, le=1.0, description="EMA smoothing for observed promotions in controller")
    team_margin_min: float = Field(0.0, ge=0.0)
    team_margin_max: float = Field(0.1, ge=0.0)
    team_power_min: float = Field(0.05, ge=0.0, le=1.0)
    team_power_max: float = Field(0.5, ge=0.0, le=1.0)
    # Evidence scheduler
    evidence_boost_enabled: bool = Field(False, description="Boost task allocation for near-threshold candidates")
    evidence_boost_factor: int = Field(1, ge=0, description="Extra per-org tasks when boosted")
    evidence_boost_cap: int = Field(8, ge=0, description="Max boosted tasks across the population per generation")
    evidence_boost_roi: float = Field(1.5, ge=0.0, description="ROI threshold to consider an org a boost candidate")
    colony_expand_delta: float = Field(0.03, ge=0.0, description="ΔROI threshold to expand 2→3")
    colony_shrink_delta: float = Field(-0.01, le=0.0, description="ΔROI threshold to shrink")
    colony_stipend: float = Field(0.2, ge=0.0, description="Stipend per pass for colony members from pot")
    min_window_max: int = Field(24, ge=2)


class LimitConfig(BaseModel):
    lora_budget_frac: float = Field(0.03, ge=0.0, le=1.0)
    max_active_adapters_per_layer: int = Field(2, ge=1)


class EvaluationConfig(BaseModel):
    enabled: bool = False
    cadence: int = Field(10, ge=1)
    tasks_path: Optional[Path] = None
    sample_size: Optional[int] = None
    reward_weight: float = Field(0.5, ge=0.0)


class MetaEvolutionConfig(BaseModel):
    enabled: bool = True
    interval: int = Field(5, ge=1)
    mutation_scale: float = Field(0.1, ge=0.0)
    catastrophe_interval: int = Field(0, ge=0)
    catastrophe_scale: float = Field(0.5, ge=0.0, le=1.0)


class DiversityConfig(BaseModel):
    enabled: bool = True
    energy_gini_cap: float = Field(0.9, ge=0.0, le=1.0)
    max_species_energy_share: float = Field(0.6, ge=0.0, le=1.0)


class QDConfig(BaseModel):
    enabled: bool = False
    cost_bins: int = Field(3, ge=1)


class CommsConfig(BaseModel):
    enabled: bool = False
    post_cost: float = Field(0.2, ge=0.0)
    read_cost: float = Field(0.1, ge=0.0)
    credit_frac: float = Field(0.2, ge=0.0, le=1.0)
    ttl: int = Field(10, ge=1)
    # Cache-to-Cache (C2C) latent comms
    c2c_enabled: bool = Field(False, description="Enable cache-to-cache latent communications")
    c2c_post_cost: float = Field(0.1, ge=0.0)
    c2c_read_cost: float = Field(0.05, ge=0.0)
    c2c_ttl: int = Field(5, ge=1)
    c2c_mix: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor for latent_prefix vs current prompt latent")

class PolicyConfig(BaseModel):
    enabled: bool = Field(False, description="If true, each organism may propose a JSON policy once per generation")
    token_cap: int = Field(64, ge=4, le=256, description="Soft cap for policy prompt; still charged to energy")
    energy_cost: float = Field(0.05, ge=0.0, description="Fixed micro-cost charged when a policy is requested")
    charge_tokens: bool = Field(
        False,
        description="If true, scale policy energy cost by used tokens / token_cap (capped at 1.0)",
    )
    allowed_fields: list[str] = Field(
        default_factory=lambda: [
            "cell_pref",
            "budget_frac",
            "explore_rate",
            "reserve_ratio",
            "read",
            "post",
            "partner_id",
            "trial",
            "gate_bias_delta",
        ]
    )
    bias_strength: float = Field(0.3, ge=0.0, le=1.0, description="Probability to honor cell_pref over controller routing")
    reserve_min: float = Field(0.0, ge=0.0, le=1.0)
    reserve_max: float = Field(0.75, ge=0.0, le=1.0)


class EnvironmentConfig(BaseModel):
    synthetic_batch_size: int = Field(8, ge=1)
    human_bandit_batch_size: int = Field(4, ge=1)
    max_episode_steps: int = Field(128, ge=1)
    energy_budget: float = Field(50.0, ge=0.0)
    success_reward_bonus: float = Field(0.5, ge=0.0)
    failure_cost_multiplier: float = Field(0.7, ge=0.0, le=1.0)
    auto_batch: bool = Field(False)
    batch_min: int = Field(1, ge=1)
    batch_max: int = Field(4, ge=1)


class HumanBanditConfig(BaseModel):
    enabled: bool = True
    preference_weight: float = Field(0.2, ge=0.0)
    helper_weight: float = Field(0.1, ge=0.0)
    frequency: float = Field(1.0, ge=0.0, le=1.0)


class MetricsConfig(BaseModel):
    root: Path = Field(Path("artifacts"))
    episodes_file: str = "episodes.jsonl"
    assimilation_file: str = "assimilation.jsonl"
    in_memory_log_limit: int = Field(256, ge=0, description="Max episodes kept in memory per run (0 = disable in‑memory cache)")


class EcologyConfig(BaseModel):
    mode: Literal["baseline", "production"] = "baseline"
    host: HostConfig = Field(default_factory=HostConfig)  # type: ignore[arg-type]
    organism: OrganismConfig = Field(default_factory=OrganismConfig)  # type: ignore[arg-type]
    hebbian: HebbianConfig = Field(default_factory=HebbianConfig)  # type: ignore[arg-type]
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)  # type: ignore[arg-type]
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)  # type: ignore[arg-type]
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)  # type: ignore[arg-type]
    grid: GridConfig = Field(default_factory=GridConfig)  # type: ignore[arg-type]
    controller: ControllerConfig = Field(default_factory=ControllerConfig)  # type: ignore[arg-type]
    pricing: PricingConfig = Field(default_factory=PricingConfig)  # type: ignore[arg-type]
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)  # type: ignore[arg-type]
    energy: EnergyConfig = Field(default_factory=EnergyConfig)  # type: ignore[arg-type]
    canary: CanaryConfig = Field(default_factory=CanaryConfig)  # type: ignore[arg-type]
    population_strategy: PopulationStrategyConfig = Field(default_factory=PopulationStrategyConfig)  # type: ignore[arg-type]
    assimilation_tuning: AssimilationTuningConfig = Field(default_factory=AssimilationTuningConfig)  # type: ignore[arg-type]
    limits: LimitConfig = Field(default_factory=LimitConfig)  # type: ignore[arg-type]
    meta: MetaEvolutionConfig = Field(default_factory=MetaEvolutionConfig)  # type: ignore[arg-type]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)  # type: ignore[arg-type]
    human_bandit: HumanBanditConfig = Field(default_factory=HumanBanditConfig)  # type: ignore[arg-type]
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)  # type: ignore[arg-type]
    qd: QDConfig = Field(default_factory=QDConfig)  # type: ignore[arg-type]
    comms: CommsConfig = Field(default_factory=CommsConfig)  # type: ignore[arg-type]
    policy: PolicyConfig = Field(default_factory=PolicyConfig)  # type: ignore[arg-type]


__all__ = [
    "EcologyConfig",
    "EvolutionConfig",
    "HebbianConfig",
    "HostConfig",
    "EnvironmentConfig",
    "OrganismConfig",
    "MetricsConfig",
    "GridConfig",
    "ControllerConfig",
    "PricingConfig",
    "EnergyConfig",
    "CanaryConfig",
    "PopulationStrategyConfig",
    "AssimilationTuningConfig",
    "LimitConfig",
    "EvaluationConfig",
    "MetaEvolutionConfig",
    "HumanBanditConfig",
    "DiversityConfig",
]


def load_ecology_config(path: Path | str) -> EcologyConfig:
    """Load an ecology configuration from a YAML file.

    Strategy:
    - Try omegaconf for full YAML support.
    - If omegaconf is missing OR parsing fails (malformed YAML), fall back to a
      minimal, robust parser that supports the simple two-level structure used in tests.
    """
    # Preferred path: omegaconf (rich YAML with interpolation)
    try:
        from omegaconf import OmegaConf  # type: ignore
        try:
            conf = OmegaConf.load(Path(path))
            data = OmegaConf.to_container(conf, resolve=True)
            return EcologyConfig.model_validate(data)
        except Exception:
            # Parsing failed; defer to the lightweight fallback below
            pass
    except ImportError:
        # No omegaconf available; use fallback parser below
        pass

    # Fallback: minimal YAML-ish parser for simple two-level configs used in tests
    # Avoids adding heavy deps in constrained environments and tolerates minor formatting issues.
    text = Path(path).read_text()
    result: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line:
            # New top-level section
            key = line.split(":", 1)[0].strip()
            current = {}
            result[key] = current
            continue
        if current is None:
            # Malformed; skip stray lines before any section
            continue
        # Expect an indented key: value
        parts = line.strip().split(":", 1)
        if len(parts) != 2:
            continue
        k, v = parts[0].strip(), parts[1].strip()
        # Parse simple list syntax: [a, b, c]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            items: list[object] = []
            if inner:
                for tok in inner.split(","):
                    # For lists, keep items as strings to satisfy typed fields like GridConfig.families
                    items.append(tok.strip())
            current[k] = items
        else:
            # Scalar: try bool/int/float else string
            vv = v.strip()
            if vv.lower() in ("true", "false"):
                current[k] = vv.lower() == "true"
            elif vv.replace(".", "", 1).isdigit():
                current[k] = float(vv) if "." in vv else int(vv)
            else:
                current[k] = vv
    return EcologyConfig.model_validate(result)


__all__.append("load_ecology_config")
