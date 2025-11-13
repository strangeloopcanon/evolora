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
    recurrence_enabled: bool = Field(
        False, description="Allow a single organelle to run multiple internal reasoning passes per call."
    )
    recurrence_train_passes: int = Field(
        1,
        ge=1,
        description="How many passes to run during normal/task episodes when recurrence is enabled.",
    )
    recurrence_eval_passes: int = Field(
        1,
        ge=1,
        description="How many passes to run during evaluation/holdout intents when recurrence is enabled.",
    )
    recurrence_history_template: str = Field(
        "Previous passes:\\n{history}\\nRefine your answer (pass {pass_idx}/{total_passes}).",
        description="Template appended to the base prompt when recurrence history exists. "
        "{history} expands to formatted prior answers.",
    )


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
    mutation_rank_noise_prob: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Probability that a mutation inserts layer-specific rank noise.",
    )
    mutation_rank_noise_scale: float = Field(
        1.0,
        ge=0.0,
        description="Std-dev for Gaussian rank noise (in rank units).",
    )
    mutation_dropout_prob: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Probability that a mutation toggles an adapter dropout mask.",
    )
    mutation_dropout_decay: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Chance to remove an existing adapter dropout mask during mutation.",
    )
    mutation_duplication_prob: float = Field(
        0.18,
        ge=0.0,
        le=1.0,
        description="Probability that a mutation duplicates a high-value adapter slice.",
    )
    mutation_duplication_scale: float = Field(
        0.5,
        ge=0.0,
        description="Std-dev for duplication amplification factors.",
    )
    mutation_layer_tags: list[str] = Field(
        default_factory=lambda: ["attn", "mlp", "proj"],
        description="Patterns used to target layer-specific mutations (rank noise, dropout, duplication).",
    )


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
    # Power economics & evidence tokens
    price_premium_alpha: float = Field(0.25, ge=0.0, description="Scale factor applied to task price when power_need > 0")
    price_premium_cap: float = Field(1.3, ge=1.0, description="Maximum multiplier applied to task price for low-power organs")
    power_target: float = Field(0.75, ge=0.0, le=1.0, description="Target statistical power; below this we mint evidence tokens")
    evidence_token_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum power_need before minting evidence tokens")
    evidence_token_mint: int = Field(1, ge=0, description="Number of evidence tokens minted when under-powered")
    evidence_token_cap: int = Field(3, ge=0, description="Maximum tokens retained per organelle")
    evidence_token_window: int = Field(2, ge=0, description="How many samples short of min_window tokens can bridge")
    info_topup_gap: int = Field(2, ge=0, description="If score count is within this many of min_window, allow info-aware top-up")
    info_topup_roi_slack: float = Field(0.15, ge=0.0, description="ROI slack allowed when granting info-aware top-ups")
    gating_snapshot_limit: int = Field(
        48,
        ge=1,
        description="Maximum number of assimilation gating/attempt samples retained per run.",
    )
    assimilation_history_limit: int = Field(
        120,
        ge=0,
        description="Maximum assimilation history entries stored per organelle/cell (0 disables truncation).",
    )
    assimilation_history_summary: int = Field(
        6,
        ge=1,
        description="Recent history entries per organelle/cell included in generation summaries.",
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
    colony_expand_delta: float = Field(0.12, ge=0.0, description="Minimum ΔROI required when expanding colony membership")
    colony_expand_windows: int = Field(3, ge=1, description="Consecutive reviews that must meet expand thresholds before adding a member")
    colony_shrink_delta: float = Field(-0.02, description="If holdout ΔROI drops below this threshold the colony may shrink")
    colony_review_interval: int = Field(6, ge=1)
    colony_required_passes: int = Field(2, ge=1)
    colony_max_failures: int = Field(2, ge=1)
    # Colony adaptation (size & bandwidth)
    colony_min_size: int = Field(2, ge=1)
    colony_max_size: int = Field(3, ge=1)
    colony_bandwidth_base: float = Field(2.0, ge=0.0, description="Base colony bandwidth allowance per generation (energy units)")
    colony_bandwidth_frac: float = Field(0.02, ge=0.0, le=1.0, description="Scale factor applied to colony pot when computing comms bandwidth")
    colony_hazard_bandwidth_scale: float = Field(0.3, ge=0.0, le=1.0, description="Multiplier applied to colony bandwidth when any member is in hazard")
    colony_post_cap: int = Field(2, ge=0)
    colony_read_cap: int = Field(3, ge=0)
    colony_post_cap_hazard: int = Field(1, ge=0, description="Post cap to use while colony hazard is active")
    colony_read_cap_hazard: int = Field(1, ge=0, description="Read cap to use while colony hazard is active")
    colony_reserve_ticket_multiplier: float = Field(
        3.0,
        ge=0.0,
        description="Baseline reserve floor as a multiple of ticket cost.",
    )
    colony_reserve_ratio: float = Field(
        0.25,
        ge=0.0,
        description="Fraction of expected near-term energy spend to keep in reserve.",
    )
    colony_reserve_cost_window: int = Field(
        6,
        ge=1,
        description="Window (episodes) to estimate expected colony costs for reserve checks.",
    )
    colony_reserve_bandwidth_scale: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Scale applied to colony comms bandwidth while reserve mode is active.",
    )
    colony_reserve_post_cap: int = Field(
        1,
        ge=0,
        description="Post cap while reserve mode is active.",
    )
    colony_reserve_read_cap: int = Field(
        1,
        ge=0,
        description="Read cap while reserve mode is active.",
    )
    colony_tax_rate: float = Field(0.1, ge=0.0, le=0.5, description="Per-episode tax rate routed into the colony pot")
    colony_subsidy_threshold: float = Field(1.0, ge=0.0, description="Energy threshold (multiples of ticket) below which members receive subsidies")
    colony_subsidy_fraction: float = Field(0.25, ge=0.0, le=1.0, description="Fraction of colony pot eligible for subsidy per member")
    colony_trait_mutation_scale: float = Field(0.05, ge=0.0, description="Std-dev multiplier for colony trait mutation")
    colony_selection_enabled: bool = Field(False)
    colony_selection_interval: int = Field(20, ge=1)
    colony_selection_alpha: float = Field(1.0)
    colony_selection_beta: float = Field(0.2)
    colony_selection_gamma: float = Field(0.0)
    colony_selection_margin: float = Field(0.05, ge=0.0)
    colony_selection_reward_frac: float = Field(0.25, ge=0.0, le=1.0)
    colony_selection_min_pool: float = Field(0.0, ge=0.0)
    colony_tier_count: int = Field(1, ge=1, description="Number of colony tiers (>=1 disables migration)")
    colony_tier_promote_passes: int = Field(3, ge=1, description="Holdout passes required to promote a colony tier")
    colony_tier_promote_delta: float = Field(0.1, description="Minimum ΔROI holdout mean to promote a tier")
    colony_tier_demote_failures: int = Field(3, ge=0, description="Holdout failures triggering demotion")
    colony_tier_demote_delta: float = Field(-0.05, description="ΔROI floor triggering demotion")
    colony_tier_hazard_floor: float = Field(-2.0, description="Hazard z-score floor triggering demotion")
    colony_tier_cooldown: int = Field(3, ge=0, description="Generations before the next tier migration is allowed")
    colony_tier_bandwidth_boost: float = Field(
        0.15,
        ge=0.0,
        description="Additional bandwidth multiplier per tier (applied as 1 + tier * boost)",
    )
    colony_winter_window: int = Field(
        6,
        ge=2,
        description="Rolling ROI window for colony winter hazard detection.",
    )
    colony_winter_z_kappa: float = Field(
        1.0,
        ge=0.0,
        description="Z-score threshold (negative) that triggers winter mode.",
    )
    colony_winter_bandwidth_scale: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Scale applied to colony comms bandwidth while winter mode is active.",
    )
    colony_winter_post_cap: int = Field(
        0,
        ge=0,
        description="Post cap during winter mode.",
    )
    colony_winter_read_cap: int = Field(
        1,
        ge=0,
        description="Read cap during winter mode.",
    )
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
    team_probe_synergy_delta: float = Field(0.12, ge=0.0, description="Required fractional lift (vs solo sum) for sustained team probes")
    team_probe_variance_nu: float = Field(0.25, ge=0.0, description="Required fractional variance reduction for sustained team probes")
    team_probe_sustain: int = Field(3, ge=1, description="Consecutive probe windows required before flagging sustained synergy")
    # Team router/composition knobs
    team_router_enabled: bool = Field(False, description="If true, a fraction of tasks may be routed to 2-member teams")
    team_vote_enabled: bool = Field(True, description="If true, select best-of-two answer (majority vote for 2)")
    team_handoff_enabled: bool = Field(False, description="If true, allow solver→checker single revise step (not yet implemented)")
    team_handoff_cap_per_gen: int = Field(4, ge=0, description="Max handoff revisions allowed per generation")
    team_handoff_cost: float = Field(0.05, ge=0.0, description="Energy cost charged to the checker for a handoff revision")
    team_max_routes_per_gen: int = Field(8, ge=0, description="Max team episodes per generation across all pairs")
    team_min_power: float = Field(0.2, ge=0.0, le=1.0, description="Minimum power proxy required for team promotions")
    team_holdout_margin: float | None = Field(None, description="Team-specific holdout margin; falls back to holdout_margin if None")
    team_holdout_sample_size: int = Field(2, ge=1, description="How many holdout tasks to sample during team probes")
    team_block_diagonal_merges: bool = Field(
        True, description="Compose assimilation soups with block-diagonal structure when team/colony roles are known"
    )
    team_block_rank_cap: int = Field(
        64,
        ge=1,
        description="Maximum combined rank when building block-diagonal soups (clamped by host max rank)",
    )
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
    colony_variance_leash: float = Field(
        1.5,
        ge=0.0,
        description="Variance ratio threshold that triggers diversification guard.",
    )
    colony_bankrupt_tolerance: int = Field(
        2,
        ge=1,
        description="How many bankruptcy hits a colony tolerates before dissolution.",
    )
    colony_guard_interval: int = Field(
        8,
        ge=1,
        description="Generations between deception guard checks (0 disables).",
    )
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
    archive_cap: int = Field(
        256,
        ge=1,
        description="Maximum number of entries retained in the MAP-Elites archive.",
    )
    novelty_weight: float = Field(
        0.3,
        ge=0.0,
        description="Weight applied to novelty when ranking merge candidates (0 disables).",
    )
    novelty_min: float = Field(
        0.05,
        ge=0.0,
        description="Floor applied to novelty so rare cells always retain some influence.",
    )


class CommsConfig(BaseModel):
    enabled: bool = False
    post_cost: float = Field(0.2, ge=0.0)
    read_cost: float = Field(0.1, ge=0.0)
    credit_frac: float = Field(0.2, ge=0.0, le=1.0)
    ttl: int = Field(10, ge=1)
    post_gen_cap: int = Field(1, ge=0, description="Maximum text posts per generation")
    read_gen_cap: int = Field(4, ge=0, description="Maximum reads per generation across the population")
    credit_power_window: int = Field(6, ge=1, description="Generations to wait for reader power improvement before abandoning credit")
    credit_power_min_delta: float = Field(0.05, ge=0.0, description="Minimum ROI improvement required to award poster credit")
    # Cache-to-Cache (C2C) latent comms
    c2c_enabled: bool = Field(False, description="Enable cache-to-cache latent communications")
    c2c_post_cost: float = Field(0.1, ge=0.0)
    c2c_read_cost: float = Field(0.05, ge=0.0)
    c2c_ttl: int = Field(5, ge=1)
    c2c_mix: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor for latent_prefix vs current prompt latent")
    history_cap: int = Field(64, ge=1, description="Maximum number of text messages retained on the shared board")
    default_priority: float = Field(0.0, description="Baseline priority assigned to system-generated posts")


class KnowledgeConfig(BaseModel):
    enabled: bool = Field(False, description="Enable per-org memory cache for useful hints/solutions")
    write_cost: float = Field(0.25, ge=0.0, description="Energy cost charged when writing to the memory cache")
    read_cost: float = Field(0.05, ge=0.0, description="Energy cost charged when prepending memories to a prompt")
    ttl: int = Field(40, ge=1, description="Generations before a memory entry expires")
    max_items: int = Field(8, ge=1, description="Maximum cached entries per organelle")


class SurvivalConfig(BaseModel):
    enabled: bool = Field(False, description="Enable reserve/hazard survival dynamics")
    reserve_ratio: float = Field(0.6, ge=0.0, description="Fraction of ticket energy to keep as reserve")
    reserve_cost_beta: float = Field(1.2, ge=0.0, description="Multiplier for recent episode costs when computing reserve threshold")
    reserve_cost_window: int = Field(6, ge=1, description="Episodes considered when estimating reserve cost")
    reserve_batch_scale: float = Field(0.5, ge=0.0, le=1.0, description="Scale applied to batch size while reserve active")
    steps_cap_low_energy: int = Field(2, ge=1, description="Cap on per-org episodes when reserve active")
    cheap_cell_quantile: float = Field(0.3, ge=0.0, le=1.0, description="Quantile of cheapest cells to sample while in reserve/hazard")
    price_bias_low_energy: bool = Field(True, description="Bias sampling toward low-price cells when reserve/hazard active")
    hazard_window: int = Field(6, ge=2, description="Window for ROI z-score to detect hazard state")
    hazard_threshold: float = Field(-0.8, description="Enter hazard when ROI z-score falls below this value")
    hazard_exit_threshold: float = Field(-0.1, description="Leave hazard once ROI z-score rises above this value")
    hazard_cooldown_gens: int = Field(4, ge=0, description="Cooldown before re-entering hazard after exit")
    hazard_rank_downshift: int = Field(1, ge=0, description="Rank decrement applied when hazard triggers")
    hazard_probe_disable: bool = Field(True, description="Skip cross-cell probes while in hazard")
    hazard_roi_relief_boost: float = Field(0.1, ge=0.0, description="Additional ROI relief granted in hazard")
    hazard_topup_bonus: float = Field(0.3, ge=0.0, description="Reduction applied to ROI threshold for energy top-ups in hazard")
    hazard_holdout_margin: float = Field(0.01, ge=0.0, description="Additional margin required for merges while in hazard")
    min_power_recovery: float = Field(0.15, ge=0.0, le=1.0, description="Minimum power proxy to attempt merges after hazard")

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
    failure_penalty: float = Field(0.05, ge=0.0, description="Energy penalty when policy output is malformed")


def _default_few_shot_examples() -> dict[str, list[dict[str, str]]]:
    return {
        "word.count": [
            {
                "prompt": "Count the number of words in the sentence: 'Agents evolve together.' Respond with an integer.",
                "answer": "3",
            },
            {
                "prompt": "Count the number of words in the sentence: 'Evolution rewards careful cooperation.' Respond with an integer.",
                "answer": "4",
            },
        ],
        "logic.bool": [
            {
                "prompt": "Evaluate the logical expression and respond with 'True' or 'False': TRUE AND NOT FALSE",
                "answer": "True",
            },
            {
                "prompt": "Evaluate the logical expression and respond with 'True' or 'False': NOT FALSE OR FALSE",
                "answer": "True",
            },
        ],
        "math": [
            {
                "prompt": "Compute 4 plus 7. Respond with the number only.",
                "answer": "11",
            },
            {
                "prompt": "Multiply 5 by 6. Respond with the number only.",
                "answer": "30",
            },
        ],
        "math.sequence": [
            {
                "prompt": "Given the sequence 2, 5, 8, what is the next number? Respond with the number only.",
                "answer": "11",
            },
            {
                "prompt": "Given the sequence 3, 6, 9, what is the next number? Respond with the number only.",
                "answer": "12",
            },
        ],
    }


class PromptingConfig(BaseModel):
    few_shot_enabled: bool = Field(False, description="Enable few-shot scaffolds for selected task families.")
    few_shot_header: str = Field(
        "Examples:",
        description="Header inserted before few-shot examples.",
    )
    few_shot_footer: str = Field(
        "Now solve the task below.",
        description="Footer inserted before the actual task prompt.",
    )
    few_shot_separator: str = Field(
        "\n",
        description="Separator between example prompt/answer pairs.",
    )
    few_shot_examples: dict[str, list[dict[str, str]]] = Field(
        default_factory=_default_few_shot_examples,
        description="Mapping of task family to list of example prompt/answer pairs.",
    )


class ForagingConfig(BaseModel):
    enabled: bool = Field(False, description="Enable evolved foraging traits and Q-based routing.")
    beta_default: float = Field(1.5, ge=0.0, description="Default softmax temperature (higher = greedier).")
    q_decay_default: float = Field(0.3, ge=0.0, le=1.0, description="Fallback EMA decay for Q updates.")
    ucb_bonus_default: float = Field(0.2, ge=0.0, description="Fallback exploration bonus for rarely visited cells.")
    q_init: float = Field(0.0, description="Initial Q value for unseen cells.")
    policy_bias_cap: float = Field(
        0.5,
        ge=0.0,
        description="Additional probability mass (fractional) granted to policy-preferred cells.",
    )
    mutation_beta_scale: float = Field(
        0.3,
        ge=0.0,
        description="Std-dev multiplier (× mutation_rate) when mutating beta_exploit.",
    )
    mutation_decay_scale: float = Field(
        0.2,
        ge=0.0,
        description="Std-dev multiplier (× mutation_rate) when mutating q_decay.",
    )
    mutation_ucb_scale: float = Field(
        0.25,
        ge=0.0,
        description="Std-dev multiplier (× mutation_rate) when mutating ucb_bonus.",
    )
    mutation_budget_scale: float = Field(
        0.25,
        ge=0.0,
        description="Std-dev multiplier (× mutation_rate) when mutating budget_aggressiveness.",
    )
    telemetry_top_k: int = Field(3, ge=1, description="Number of top cells to surface per organelle in telemetry.")
    telemetry_max_orgs: int = Field(
        10,
        ge=1,
        description="Maximum number of organelles to report in foraging telemetry snapshot.",
    )


class WinterConfig(BaseModel):
    enabled: bool = Field(False, description="Enable periodic winter pulses that stress the economy.")
    winter_interval: int = Field(
        40,
        ge=1,
        description="Number of generations between the start of winter pulses.",
    )
    winter_duration: int = Field(
        4,
        ge=1,
        description="How many generations a winter pulse lasts once triggered.",
    )
    price_multiplier: float = Field(
        1.2,
        ge=0.0,
        description="Multiplier applied to task prices during winter.",
    )
    ticket_multiplier: float = Field(
        0.7,
        ge=0.0,
        description="Multiplier applied to ticket cost during winter (values <1.0 reduce tickets).",
    )
    post_winter_bonus: float = Field(
        0.25,
        ge=0.0,
        description="Fraction of ticket energy credited to survivors when winter ends.",
    )


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
    budget_enabled: bool = Field(True, description="Enable per-org budgeting driven by energy, traits, and policy signals")
    budget_energy_floor: float = Field(0.4, ge=0.0, description="Lower clamp for energy balance ratio when computing budgets")
    budget_energy_ceiling: float = Field(3.0, ge=0.0, description="Upper clamp for energy balance ratio when computing budgets")
    budget_trait_bonus: float = Field(1.0, ge=0.0, description="Strength of trait influence (explore_rate) on per-org budgets")
    budget_policy_floor: float = Field(0.3, ge=0.0, description="Minimum policy budget_frac multiplier")
    budget_policy_ceiling: float = Field(2.0, ge=0.0, description="Maximum policy budget_frac multiplier")
    global_episode_cap: int = Field(
        0,
        ge=0,
        description="Optional hard cap on total synthetic episodes per generation (0 disables the cap)",
    )


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
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)  # type: ignore[arg-type]
    policy: PolicyConfig = Field(default_factory=PolicyConfig)  # type: ignore[arg-type]
    prompting: PromptingConfig = Field(default_factory=PromptingConfig)  # type: ignore[arg-type]
    foraging: ForagingConfig = Field(default_factory=ForagingConfig)  # type: ignore[arg-type]
    winter: WinterConfig = Field(default_factory=WinterConfig)  # type: ignore[arg-type]
    survival: SurvivalConfig = Field(default_factory=SurvivalConfig)  # type: ignore[arg-type]


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
    "QDConfig",
    "CommsConfig",
    "KnowledgeConfig",
    "PolicyConfig",
    "PromptingConfig",
    "ForagingConfig",
    "WinterConfig",
    "SurvivalConfig",
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
def _default_few_shot_examples() -> dict[str, list[dict[str, str]]]:
    return {
        "word.count": [
            {
                "prompt": "Count the number of words in the sentence: 'Agents evolve together.' Respond with an integer.",
                "answer": "3",
            },
            {
                "prompt": "Count the number of words in the sentence: 'Evolution rewards careful cooperation.' Respond with an integer.",
                "answer": "4",
            },
        ],
        "logic.bool": [
            {
                "prompt": "Evaluate the logical expression and respond with 'True' or 'False': TRUE AND NOT FALSE",
                "answer": "True",
            },
            {
                "prompt": "Evaluate the logical expression and respond with 'True' or 'False': NOT FALSE OR FALSE",
                "answer": "True",
            },
        ],
        "math": [
            {
                "prompt": "Compute 4 plus 7. Respond with the number only.",
                "answer": "11",
            },
            {
                "prompt": "Multiply 5 by 6. Respond with the number only.",
                "answer": "30",
            },
        ],
        "math.sequence": [
            {
                "prompt": "Given the sequence 2, 5, 8, what is the next number? Respond with the number only.",
                "answer": "11",
            },
            {
                "prompt": "Given the sequence 3, 6, 9, what is the next number? Respond with the number only.",
                "answer": "12",
            },
        ],
    }


class PromptingConfig(BaseModel):
    few_shot_enabled: bool = Field(False, description="Enable few-shot scaffolds for selected task families.")
    few_shot_header: str = Field(
        "Examples:",
        description="Header inserted before few-shot examples.",
    )
    few_shot_footer: str = Field(
        "Now solve the task below.",
        description="Footer inserted before the actual task prompt.",
    )
    few_shot_separator: str = Field(
        "\n",
        description="Separator between example prompt/answer pairs.",
    )
    few_shot_examples: dict[str, list[dict[str, str]]] = Field(
        default_factory=_default_few_shot_examples,
        description="Mapping of task family to list of example prompt/answer pairs.",
    )
