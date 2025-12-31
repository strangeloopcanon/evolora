"""Tests for regex generalization evaluation framework."""

from __future__ import annotations

from pathlib import Path

import pytest

from symbiont_ecology.evaluation.regex_generalization import (
    CapabilityAxis,
    HoldOutType,
    RegexGeneralizationEvaluator,
    RegexMetrics,
    RegexTask,
    RegexTestCase,
    analyze_regex,
    check_regex_against_cases,
    compare_reports,
    evaluate_debugging_task,
    evaluate_explanation_task,
    evaluate_recognition_task,
    evaluate_refactoring_task,
    evaluate_synthesis_task,
    extract_regex_from_response,
)

# ---------------------------------------------------------------------------
# Test analyze_regex
# ---------------------------------------------------------------------------


class TestAnalyzeRegex:
    def test_simple_pattern(self):
        metrics = analyze_regex(r"\d+")
        assert metrics.char_length == 3
        assert metrics.quantifier_count >= 1
        assert metrics.nesting_depth == 0

    def test_complex_pattern(self):
        metrics = analyze_regex(r"^(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
        assert metrics.char_length > 0
        assert metrics.alternation_count >= 2
        assert metrics.nesting_depth >= 1
        assert metrics.group_count >= 1

    def test_nested_groups(self):
        metrics = analyze_regex(r"((a)(b)(c))")
        assert metrics.nesting_depth == 2
        assert metrics.group_count >= 4

    def test_backtracking_risk_detection(self):
        # Pattern with multiple greedy wildcards
        risky = analyze_regex(r".*foo.*bar.*")
        assert risky.has_backtracking_risk is True

        # Safe pattern
        safe = analyze_regex(r"^[a-z]+$")
        assert safe.has_backtracking_risk is False

    def test_complexity_score(self):
        simple = analyze_regex(r"\d+")
        complex_pattern = analyze_regex(r"^(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
        assert complex_pattern.complexity_score > simple.complexity_score


# ---------------------------------------------------------------------------
# Test extract_regex_from_response
# ---------------------------------------------------------------------------


class TestExtractRegexFromResponse:
    def test_plain_regex(self):
        assert extract_regex_from_response(r"\d+") == r"\d+"

    def test_code_block(self):
        response = "```regex\n^[a-z]+$\n```"
        assert extract_regex_from_response(response) == r"^[a-z]+$"

    def test_inline_backticks(self):
        response = "The pattern is `\\d{3}-\\d{4}`"
        assert extract_regex_from_response(response) == r"\d{3}-\d{4}"

    def test_multiline_response(self):
        response = "Here's the regex:\n^[A-Z][a-z]+$"
        assert extract_regex_from_response(response) == r"^[A-Z][a-z]+$"

    def test_skip_prefix_lines(self):
        response = "regex\n^test$"
        assert extract_regex_from_response(response) == r"^test$"


# ---------------------------------------------------------------------------
# Test test_regex_against_cases
# ---------------------------------------------------------------------------


class TestCheckRegexAgainstCases:
    def test_all_pass(self):
        cases = [
            RegexTestCase("abc", True),
            RegexTestCase("123", False),
        ]
        success, details = check_regex_against_cases(r"^[a-z]+$", cases)
        assert success is True
        assert len(details) == 2
        assert all(d["passed"] for d in details)

    def test_some_fail(self):
        cases = [
            RegexTestCase("abc", True),
            RegexTestCase("ABC", True),  # This will fail
        ]
        success, details = check_regex_against_cases(r"^[a-z]+$", cases)
        assert success is False

    def test_invalid_regex(self):
        cases = [RegexTestCase("test", True)]
        success, details = check_regex_against_cases(r"[invalid(", cases)
        assert success is False
        assert "error" in details[0]


# ---------------------------------------------------------------------------
# Test Task Evaluators
# ---------------------------------------------------------------------------


class TestRecognitionEvaluator:
    def test_yes_answer(self):
        task = RegexTask(
            task_id="test",
            prompt="Does it match?",
            capability=CapabilityAxis.RECOGNITION,
            expected_answer="yes",
        )
        success, _ = evaluate_recognition_task(task, "Yes, it matches because...")
        assert success is True

    def test_no_answer(self):
        task = RegexTask(
            task_id="test",
            prompt="Does it match?",
            capability=CapabilityAxis.RECOGNITION,
            expected_answer="no",
        )
        success, _ = evaluate_recognition_task(task, "No, it doesn't match because...")
        assert success is True

    def test_wrong_answer(self):
        task = RegexTask(
            task_id="test",
            prompt="Does it match?",
            capability=CapabilityAxis.RECOGNITION,
            expected_answer="yes",
        )
        success, _ = evaluate_recognition_task(task, "No, it does not match")
        assert success is False


class TestSynthesisEvaluator:
    def test_correct_synthesis(self):
        task = RegexTask(
            task_id="test",
            prompt="Write a regex for digits",
            capability=CapabilityAxis.SYNTHESIS,
            test_cases=[
                RegexTestCase("123", True),
                RegexTestCase("abc", False),
            ],
        )
        success, details = evaluate_synthesis_task(task, r"\d+")
        assert success is True
        assert "extracted_pattern" in details

    def test_incorrect_synthesis(self):
        task = RegexTask(
            task_id="test",
            prompt="Write a regex for digits only",
            capability=CapabilityAxis.SYNTHESIS,
            test_cases=[
                RegexTestCase("123", True),
                RegexTestCase("abc", False),
            ],
        )
        # Pattern that matches anything
        success, details = evaluate_synthesis_task(task, r".*")
        assert success is False


class TestExplanationEvaluator:
    def test_good_explanation(self):
        task = RegexTask(
            task_id="test",
            prompt="Explain this regex",
            capability=CapabilityAxis.EXPLANATION,
            metadata={"required_keywords": ["digit", "number", "match"]},
        )
        response = "This regex matches any digit or number in the string"
        success, details = evaluate_explanation_task(task, response)
        assert success is True
        assert details["coverage_score"] >= 0.7

    def test_poor_explanation(self):
        task = RegexTask(
            task_id="test",
            prompt="Explain this regex",
            capability=CapabilityAxis.EXPLANATION,
            metadata={"required_keywords": ["year", "month", "day", "timestamp"]},
        )
        response = "This regex matches stuff"
        success, details = evaluate_explanation_task(task, response)
        assert success is False


class TestDebuggingEvaluator:
    def test_successful_fix(self):
        task = RegexTask(
            task_id="test",
            prompt="Fix this regex",
            capability=CapabilityAxis.DEBUGGING,
            target_regex=r"\d",  # buggy
            test_cases=[
                RegexTestCase("123", True),
                RegexTestCase("abc", False),
            ],
        )
        success, _ = evaluate_debugging_task(task, r"^\d+$")
        assert success is True

    def test_failed_fix(self):
        task = RegexTask(
            task_id="test",
            prompt="Fix this regex",
            capability=CapabilityAxis.DEBUGGING,
            test_cases=[
                RegexTestCase("123", True),
                RegexTestCase("abc", False),
            ],
        )
        # Still matches abc
        success, _ = evaluate_debugging_task(task, r".*")
        assert success is False


class TestRefactoringEvaluator:
    def test_simpler_pattern(self):
        task = RegexTask(
            task_id="test",
            prompt="Simplify this regex",
            capability=CapabilityAxis.REFACTORING,
            target_regex=r"^[0-9][0-9][0-9]$",  # verbose
            test_cases=[
                RegexTestCase("123", True),
                RegexTestCase("12", False),
                RegexTestCase("1234", False),
            ],
        )
        success, details = evaluate_refactoring_task(task, r"^\d{3}$")
        assert success is True
        assert "improvement" in details
        assert details["improvement"] > 0


# ---------------------------------------------------------------------------
# Test Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_task_round_trip(self):
        task = RegexTask(
            task_id="test_001",
            prompt="Write a regex",
            capability=CapabilityAxis.SYNTHESIS,
            holdout_type=HoldOutType.OPERATOR,
            mutation_type=None,
            target_regex=r"^test$",
            test_cases=[
                RegexTestCase("test", True),
                RegexTestCase("TEST", False),
            ],
            expected_answer=None,
            metadata={"key": "value"},
        )

        # Serialize and deserialize
        data = task.to_dict()
        restored = RegexTask.from_dict(data)

        assert restored.task_id == task.task_id
        assert restored.capability == task.capability
        assert restored.holdout_type == task.holdout_type
        assert restored.target_regex == task.target_regex
        assert len(restored.test_cases) == len(task.test_cases)
        assert restored.metadata == task.metadata

    def test_load_from_jsonl(self):
        # Load from the existing eval suite in the repo
        path = Path(__file__).parent.parent / "config" / "evaluation" / "regex_generalization.jsonl"
        if path.exists():
            evaluator = RegexGeneralizationEvaluator.from_jsonl(path)
            assert len(evaluator.tasks) > 0
            # Verify tasks loaded correctly
            for task in evaluator.tasks:
                assert task.task_id is not None
                assert task.capability is not None


# ---------------------------------------------------------------------------
# Test Full Evaluator
# ---------------------------------------------------------------------------


class TestRegexGeneralizationEvaluator:
    def test_evaluate_single_success(self):
        task = RegexTask(
            task_id="synth_test",
            prompt="Write a regex for 3 digits",
            capability=CapabilityAxis.SYNTHESIS,
            test_cases=[
                RegexTestCase("123", True),
                RegexTestCase("12", False),
                RegexTestCase("1234", False),
            ],
        )
        evaluator = RegexGeneralizationEvaluator([task])
        result = evaluator.evaluate_single(task, r"^\d{3}$")

        assert result.success is True
        assert result.capability == CapabilityAxis.SYNTHESIS
        assert result.metrics is not None

    def test_evaluate_responses(self):
        tasks = [
            RegexTask(
                task_id="task_1",
                prompt="Write a regex for digits",
                capability=CapabilityAxis.SYNTHESIS,
                test_cases=[RegexTestCase("123", True), RegexTestCase("abc", False)],
            ),
            RegexTask(
                task_id="task_2",
                prompt="Does it match 'test'?",
                capability=CapabilityAxis.RECOGNITION,
                expected_answer="yes",
            ),
        ]

        evaluator = RegexGeneralizationEvaluator(tasks)
        responses = {
            "task_1": r"^\d+$",
            "task_2": "Yes, it matches",
        }

        report = evaluator.evaluate_responses(responses)

        assert report.total_tasks == 2
        assert report.total_correct == 2
        assert report.overall_accuracy == 1.0
        assert CapabilityAxis.SYNTHESIS.value in report.capability_breakdown
        assert CapabilityAxis.RECOGNITION.value in report.capability_breakdown

    def test_evaluate_all_with_runner(self):
        # Create a small set of tasks for testing
        tasks = [
            RegexTask(
                task_id="runner_test_1",
                prompt="Write a regex for digits",
                capability=CapabilityAxis.SYNTHESIS,
                test_cases=[RegexTestCase("123", True), RegexTestCase("abc", False)],
            ),
            RegexTask(
                task_id="runner_test_2",
                prompt="Does ^\\d+$ match '456'?",
                capability=CapabilityAxis.RECOGNITION,
                expected_answer="yes",
            ),
        ]

        def dummy_runner(prompt: str) -> str:
            if "match" in prompt.lower():
                return "Yes, it matches"
            return r"^\d+$"

        evaluator = RegexGeneralizationEvaluator(tasks)
        report = evaluator.evaluate_all(dummy_runner, verbose=False)

        assert report.total_tasks == 2
        assert 0 <= report.overall_accuracy <= 1


# ---------------------------------------------------------------------------
# Test Comparison
# ---------------------------------------------------------------------------


class TestCompareReports:
    def test_basic_comparison(self):
        from symbiont_ecology.evaluation.regex_generalization import EvalReport

        report_a = EvalReport(
            total_tasks=10,
            total_correct=7,
            overall_accuracy=0.7,
            capability_breakdown={"synthesis": {"accuracy": 0.8, "total": 5, "correct": 4}},
            holdout_breakdown={},
            mutation_breakdown={},
            simplicity_stats={"mean_complexity": 25.0},
            task_results=[],
        )

        report_b = EvalReport(
            total_tasks=10,
            total_correct=8,
            overall_accuracy=0.8,
            capability_breakdown={"synthesis": {"accuracy": 0.9, "total": 5, "correct": 4.5}},
            holdout_breakdown={},
            mutation_breakdown={},
            simplicity_stats={"mean_complexity": 20.0},
            task_results=[],
        )

        comparison = compare_reports(report_a, report_b, "SFT", "Evolved")

        assert comparison["overall"]["delta"] == pytest.approx(0.1)
        assert comparison["capability_comparison"]["synthesis"]["delta"] == pytest.approx(0.1)
        assert comparison["simplicity_comparison"]["delta"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Test Metrics Dataclass
# ---------------------------------------------------------------------------


class TestRegexMetrics:
    def test_complexity_score_computation(self):
        metrics = RegexMetrics(
            char_length=50,
            ast_node_count=20,
            nesting_depth=3,
            alternation_count=2,
            group_count=4,
            quantifier_count=5,
            has_backtracking_risk=True,
        )

        # Verify complexity score is computed correctly
        expected = (
            50 * 0.1  # char_length
            + 20 * 1.0  # ast_node_count
            + 3 * 2.0  # nesting_depth
            + 2 * 1.5  # alternation_count
            + 4 * 0.5  # group_count
            + 5 * 0.3  # quantifier_count
            + 5.0  # backtracking risk
        )
        assert metrics.complexity_score == pytest.approx(expected)
