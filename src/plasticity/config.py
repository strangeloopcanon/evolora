"""Pydantic configuration models for submask discovery experiments."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Controls task generation for calibration and evaluation."""

    families: List[str] = Field(
        default=["regex", "math", "word.count"],
        description="Task families to generate.",
    )
    calibration_per_family: int = Field(
        200, ge=1, description="Number of calibration tasks per family."
    )
    holdout_per_family: int = Field(200, ge=1, description="Number of holdout tasks per family.")
    calibration_seed: int = Field(42, description="RNG seed for calibration tasks.")
    holdout_seed: int = Field(
        7777, description="RNG seed for holdout tasks (must differ from calibration)."
    )
    max_sequence_length: int = Field(256, ge=32, description="Max prompt token length.")


class ModelConfig(BaseModel):
    """Model loading configuration."""

    model_id: str = Field("Qwen/Qwen3-1.7B", description="HuggingFace model identifier.")
    dtype: str = Field("bfloat16", description="Weight dtype.")
    device: str = Field("auto", description="Device placement.")
    batch_size: int = Field(8, ge=1, description="Inference batch size.")


class CalibrationConfig(BaseModel):
    """Phase 1: per-family importance recording."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    tasks: TaskConfig = Field(default_factory=TaskConfig)
    output_dir: str = Field(
        "artifacts_plasticity/calibration", description="Where to save importance tensors."
    )


class MaskConfig(BaseModel):
    """Phase 2: mask derivation from importance tensors."""

    importance_dir: str = Field(
        "artifacts_plasticity/calibration",
        description="Directory containing saved importance tensors.",
    )
    sparsity_levels: List[float] = Field(
        default=[0.3, 0.5, 0.7, 0.9],
        description="Target sparsity fractions (proportion of weights to zero out).",
    )
    output_dir: str = Field(
        "artifacts_plasticity/masks", description="Where to save derived masks."
    )
    random_seed: int = Field(999, description="Seed for random baseline masks.")


class EvalConfig(BaseModel):
    """Phase 3: 5-condition holdout evaluation."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    tasks: TaskConfig = Field(default_factory=TaskConfig)
    masks_dir: str = Field(
        "artifacts_plasticity/masks", description="Directory containing derived masks."
    )
    output_dir: str = Field(
        "artifacts_plasticity/eval", description="Where to save evaluation results."
    )
    conditions: List[str] = Field(
        default=["dense", "task_matched", "cross_task", "global", "random"],
        description="Evaluation conditions to run.",
    )


class AnalysisConfig(BaseModel):
    """Phase 4: structural analysis and plotting."""

    masks_dir: str = Field(
        "artifacts_plasticity/masks", description="Directory with masks for overlap analysis."
    )
    importance_dir: str = Field(
        "artifacts_plasticity/calibration",
        description="Directory with importance tensors for correlation analysis.",
    )
    eval_dir: str = Field(
        "artifacts_plasticity/eval", description="Directory with evaluation results."
    )
    output_dir: str = Field(
        "artifacts_plasticity/analysis", description="Where to save plots and tables."
    )
    families: List[str] = Field(default=["regex", "math", "word.count"])
    sparsity_levels: List[float] = Field(default=[0.3, 0.5, 0.7, 0.9])
    dpi: int = Field(150, ge=72, description="Plot resolution.")


class ExperimentConfig(BaseModel):
    """Top-level config aggregating all phases."""

    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    masks: MaskConfig = Field(default_factory=MaskConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    run_name: Optional[str] = Field(None, description="Optional experiment run identifier.")
