"""Tests for multi-task transfer evaluation."""

import json
import sys
import tempfile
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_transfer import (
    FamilyResult,
    TransferResult,
    compute_robustness_premium,
    load_holdout_tasks,
)


class TestFamilyResult:
    """Test FamilyResult dataclass."""

    def test_accuracy_zero_total(self) -> None:
        result = FamilyResult(family="test", correct=0, total=0)
        assert result.accuracy == 0.0

    def test_accuracy_calculation(self) -> None:
        result = FamilyResult(family="test", correct=7, total=10)
        assert result.accuracy == 0.7

    def test_is_source_flag(self) -> None:
        source = FamilyResult(family="regex", is_source=True)
        target = FamilyResult(family="math", is_source=False)
        assert source.is_source is True
        assert target.is_source is False


class TestTransferResult:
    """Test TransferResult aggregation logic."""

    def test_source_accuracy(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=5, total=10, is_source=False),
        }
        assert result.source_accuracy == 0.8

    def test_target_accuracies(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=5, total=10, is_source=False),
            "code": FamilyResult("code", correct=6, total=10, is_source=False),
        }
        targets = result.target_accuracies
        assert "regex" not in targets
        assert targets["math"] == 0.5
        assert targets["code"] == 0.6

    def test_mean_target_accuracy(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=4, total=10, is_source=False),
            "code": FamilyResult("code", correct=6, total=10, is_source=False),
        }
        assert result.mean_target_accuracy == 0.5  # (0.4 + 0.6) / 2

    def test_transfer_ratio(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=10, total=10, is_source=True),
            "math": FamilyResult("math", correct=5, total=10, is_source=False),
        }
        assert result.transfer_ratios["math"] == 0.5  # 0.5 / 1.0

    def test_transfer_gap(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=5, total=10, is_source=False),
        }
        assert abs(result.transfer_gaps["math"] - 0.3) < 0.001  # 0.8 - 0.5

    def test_geometric_mean(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=10, total=10, is_source=True),
            "math": FamilyResult("math", correct=10, total=10, is_source=False),
        }
        # Both 100% so geometric mean should be 1.0
        assert abs(result.geometric_mean_accuracy - 1.0) < 0.001

    def test_summary_format(self) -> None:
        result = TransferResult(model_name="test", source_family="regex")
        result.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
        }
        summary = result.summary()
        assert summary["model"] == "test"
        assert summary["source_family"] == "regex"
        assert "source_accuracy" in summary


class TestRobustnessPremium:
    """Test robustness premium calculation."""

    def test_evolution_more_robust(self) -> None:
        evo = TransferResult(model_name="evo", source_family="regex")
        evo.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=7, total=10, is_source=False),
        }

        sft = TransferResult(model_name="sft", source_family="regex")
        sft.family_results = {
            "regex": FamilyResult("regex", correct=9, total=10, is_source=True),
            "math": FamilyResult("math", correct=4, total=10, is_source=False),
        }

        premium = compute_robustness_premium(evo, sft)
        assert premium["mean_target_accuracy_delta"] > 0
        assert premium["interpretation"] == "evolution_more_robust"

    def test_sft_more_robust(self) -> None:
        evo = TransferResult(model_name="evo", source_family="regex")
        evo.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=3, total=10, is_source=False),
        }

        sft = TransferResult(model_name="sft", source_family="regex")
        sft.family_results = {
            "regex": FamilyResult("regex", correct=7, total=10, is_source=True),
            "math": FamilyResult("math", correct=6, total=10, is_source=False),
        }

        premium = compute_robustness_premium(evo, sft)
        assert premium["mean_target_accuracy_delta"] < 0
        assert premium["interpretation"] == "sft_more_robust"

    def test_roughly_equivalent(self) -> None:
        evo = TransferResult(model_name="evo", source_family="regex")
        evo.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=6, total=10, is_source=False),
        }

        sft = TransferResult(model_name="sft", source_family="regex")
        sft.family_results = {
            "regex": FamilyResult("regex", correct=8, total=10, is_source=True),
            "math": FamilyResult("math", correct=6, total=10, is_source=False),
        }

        premium = compute_robustness_premium(evo, sft)
        assert premium["interpretation"] == "roughly_equivalent"


class TestLoadHoldoutTasks:
    """Test holdout task loading and filtering."""

    def test_loads_source_and_target_families(self) -> None:
        tasks_data = [
            {"prompt": "p1", "target": "t1", "family": "regex"},
            {"prompt": "p2", "target": "t2", "family": "math.multi_step"},
            {"prompt": "p3", "target": "t3", "family": "code.format"},
            {"prompt": "p4", "target": "t4", "family": "unrelated"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for t in tasks_data:
                f.write(json.dumps(t) + "\n")
            f.flush()

            tasks = load_holdout_tasks(
                f.name,
                source_family="regex",
                target_families=["math.multi_step"],
            )

        # Should include regex (source) and math.multi_step (target)
        # Should exclude code.format and unrelated
        assert len(tasks) == 2
        families = {t["family"] for t in tasks}
        assert "regex" in families
        assert "math.multi_step" in families

    def test_marks_source_tasks(self) -> None:
        tasks_data = [
            {"prompt": "p1", "target": "t1", "family": "regex"},
            {"prompt": "p2", "target": "t2", "family": "math"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for t in tasks_data:
                f.write(json.dumps(t) + "\n")
            f.flush()

            tasks = load_holdout_tasks(
                f.name,
                source_family="regex",
                target_families=["math"],
            )

        source_task = next(t for t in tasks if t["family"] == "regex")
        target_task = next(t for t in tasks if t["family"] == "math")
        assert source_task["_is_source"] is True
        assert target_task["_is_source"] is False

    def test_samples_per_family(self) -> None:
        tasks_data = [
            {"prompt": f"p{i}", "target": f"t{i}", "family": "regex"} for i in range(20)
        ] + [{"prompt": f"m{i}", "target": f"mt{i}", "family": "math"} for i in range(20)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for t in tasks_data:
                f.write(json.dumps(t) + "\n")
            f.flush()

            tasks = load_holdout_tasks(
                f.name,
                source_family="regex",
                target_families=["math"],
                samples_per_family=5,
            )

        regex_count = sum(1 for t in tasks if t["family"] == "regex")
        math_count = sum(1 for t in tasks if t["family"] == "math")
        assert regex_count == 5
        assert math_count == 5
