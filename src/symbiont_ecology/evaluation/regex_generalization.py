"""Regex Generalization Evaluation Framework.

This module measures the generalizability of regex-related skills in language models,
distinguishing conceptual understanding from template memorization.

Evaluation Dimensions:
1. Capability Axes - Different regex task types (Recognition, Synthesis, etc.)
2. Hold-Out Structure - Novel regex structures at test time
3. Simplicity/Minimality - Solution quality metrics
4. Mutation/Counterfactual - Local semantic reasoning tests

See docs/regex_generalization.md for framework documentation.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from symbiont_ecology.utils.regex_extract import pick_best_regex_candidate

# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class CapabilityAxis(str, Enum):
    """Types of regex skills being evaluated."""

    RECOGNITION = "recognition"  # Does this regex match X?
    SYNTHESIS = "synthesis"  # Write a regex that matches...
    EXPLANATION = "explanation"  # Explain what this regex matches
    DEBUGGING = "debugging"  # Fix this regex
    REFACTORING = "refactoring"  # Simplify this regex


class HoldOutType(str, Enum):
    """Types of structure being held out at test time."""

    OPERATOR = "operator"  # Novel operators (e.g., alternation)
    COMPOSITION = "composition"  # Combined operators
    SEMANTIC_COUPLING = "semantic_coupling"  # Joint constraints
    SURFACE_FORM = "surface_form"  # Different delimiters/formats


class MutationType(str, Enum):
    """Types of regex mutations for counterfactual tests."""

    WIDEN_RANGE = "widen_range"  # e.g., [0-3] -> [0-9]
    REMOVE_GROUPING = "remove_grouping"  # e.g., (?:...) -> ...
    CHANGE_QUANTIFIER = "change_quantifier"  # e.g., {3} -> +
    REMOVE_ANCHOR = "remove_anchor"  # e.g., ^...$ -> ...
    FLIP_NEGATION = "flip_negation"  # e.g., [^a] -> [a]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RegexTestCase:
    """A single test case for regex evaluation."""

    string: str
    should_match: bool


@dataclass
class RegexTask:
    """A regex evaluation task."""

    task_id: str
    prompt: str
    capability: CapabilityAxis
    holdout_type: HoldOutType | None = None
    mutation_type: MutationType | None = None
    target_regex: str | None = None
    test_cases: list[RegexTestCase] = field(default_factory=list)
    expected_answer: str | None = None  # For non-synthesis tasks
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSONL output."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "capability": self.capability.value,
            "holdout_type": self.holdout_type.value if self.holdout_type else None,
            "mutation_type": self.mutation_type.value if self.mutation_type else None,
            "target_regex": self.target_regex,
            "test_cases": [
                {"string": tc.string, "should_match": tc.should_match} for tc in self.test_cases
            ],
            "expected_answer": self.expected_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegexTask:
        """Deserialize from dictionary."""
        test_cases = [
            RegexTestCase(tc["string"], tc["should_match"]) for tc in data.get("test_cases", [])
        ]
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            capability=CapabilityAxis(data["capability"]),
            holdout_type=HoldOutType(data["holdout_type"]) if data.get("holdout_type") else None,
            mutation_type=(
                MutationType(data["mutation_type"]) if data.get("mutation_type") else None
            ),
            target_regex=data.get("target_regex"),
            test_cases=test_cases,
            expected_answer=data.get("expected_answer"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegexMetrics:
    """Simplicity and quality metrics for a regex pattern."""

    char_length: int
    ast_node_count: int
    nesting_depth: int
    alternation_count: int
    group_count: int
    quantifier_count: int
    has_backtracking_risk: bool

    @property
    def complexity_score(self) -> float:
        """Compute a weighted complexity score (lower is simpler)."""
        return (
            self.char_length * 0.1
            + self.ast_node_count * 1.0
            + self.nesting_depth * 2.0
            + self.alternation_count * 1.5
            + self.group_count * 0.5
            + self.quantifier_count * 0.3
            + (5.0 if self.has_backtracking_risk else 0.0)
        )


@dataclass
class TaskResult:
    """Result of evaluating a single task."""

    task_id: str
    capability: CapabilityAxis
    holdout_type: HoldOutType | None
    mutation_type: MutationType | None
    success: bool
    response: str
    metrics: RegexMetrics | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Comprehensive evaluation report."""

    total_tasks: int
    total_correct: int
    overall_accuracy: float
    capability_breakdown: dict[str, dict[str, float]]
    holdout_breakdown: dict[str, dict[str, float]]
    mutation_breakdown: dict[str, dict[str, float]]
    simplicity_stats: dict[str, float]
    task_results: list[TaskResult]

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "total_correct": self.total_correct,
                "overall_accuracy": self.overall_accuracy,
            },
            "capability_breakdown": self.capability_breakdown,
            "holdout_breakdown": self.holdout_breakdown,
            "mutation_breakdown": self.mutation_breakdown,
            "simplicity_stats": self.simplicity_stats,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "capability": r.capability.value,
                    "holdout_type": r.holdout_type.value if r.holdout_type else None,
                    "mutation_type": r.mutation_type.value if r.mutation_type else None,
                    "success": r.success,
                    "response": r.response[:200],  # Truncate for legibility
                    "details": r.details,
                }
                for r in self.task_results
            ],
        }


# ---------------------------------------------------------------------------
# Regex Analysis Utilities
# ---------------------------------------------------------------------------


def analyze_regex(pattern: str) -> RegexMetrics:
    """Analyze a regex pattern and compute metrics."""
    # Basic metrics
    char_length = len(pattern)
    alternation_count = pattern.count("|")
    group_count = len(re.findall(r"\((?!\?:)", pattern))  # Capturing groups
    group_count += len(re.findall(r"\(\?:", pattern))  # Non-capturing groups
    quantifier_count = len(re.findall(r"[+*?]|\{\d+(?:,\d*)?\}", pattern))

    # AST node count (simplified estimate)
    # Count significant tokens
    node_patterns = [
        r"\[(?:\^)?[^\]]+\]",  # Character classes
        r"\\[dDwWsS]",  # Character class shortcuts
        r"\\.",  # Escaped characters
        r"[+*?]",  # Quantifiers
        r"\{\d+(?:,\d*)?\}",  # Range quantifiers
        r"\|",  # Alternation
        r"\((?:\?:)?",  # Group starts
        r"\)",  # Group ends
        r"\^|\$",  # Anchors
        r"[^\\[\]{}()+*?|^$]",  # Literal characters
    ]
    ast_node_count = sum(len(re.findall(p, pattern)) for p in node_patterns)

    # Nesting depth
    max_depth = 0
    current_depth = 0
    for char in pattern:
        if char == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ")":
            current_depth = max(0, current_depth - 1)

    # Backtracking risk (simplified heuristic)
    # Nested quantifiers or .* patterns are risky
    has_backtracking_risk = bool(
        re.search(r"\.\*.*\.\*", pattern)  # Multiple greedy wildcards
        or re.search(r"\([^)]*[+*][^)]*\)[+*]", pattern)  # Nested quantifiers
        or re.search(r"(?<!\?)\*\*", pattern)  # Consecutive quantifiers
    )

    return RegexMetrics(
        char_length=char_length,
        ast_node_count=ast_node_count,
        nesting_depth=max_depth,
        alternation_count=alternation_count,
        group_count=group_count,
        quantifier_count=quantifier_count,
        has_backtracking_risk=has_backtracking_risk,
    )


def check_regex_against_cases(
    pattern: str, test_cases: list[RegexTestCase]
) -> tuple[bool, list[dict[str, Any]]]:
    """Check a regex pattern against test cases.

    Returns (all_passed, details) where details is a list of per-case results.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return False, [{"error": f"Invalid regex: {e}"}]

    details = []
    all_passed = True

    for tc in test_cases:
        match = compiled.fullmatch(tc.string)
        actual_match = match is not None
        passed = actual_match == tc.should_match
        if not passed:
            all_passed = False
        details.append(
            {
                "string": tc.string,
                "expected": tc.should_match,
                "actual": actual_match,
                "passed": passed,
            }
        )

    return all_passed, details


def extract_regex_from_response(response: str) -> str:
    """Extract a regex pattern from model response."""
    pattern, _details = pick_best_regex_candidate(response, test_cases=None)
    if pattern:
        return pattern
    return response.strip()


# ---------------------------------------------------------------------------
# Task Evaluators
# ---------------------------------------------------------------------------


class ModelRunner(Protocol):
    """Protocol for running a model on a prompt."""

    def __call__(self, prompt: str) -> str:
        """Run the model and return response text."""
        ...


def evaluate_recognition_task(task: RegexTask, response: str) -> tuple[bool, dict[str, Any]]:
    """Evaluate a recognition task (does this regex match X?)."""
    response_lower = response.strip().lower()

    # Extract yes/no/true/false
    if task.expected_answer:
        expected = task.expected_answer.lower()
        # Check for explicit answer
        if expected in ["yes", "true", "matches"]:
            success = any(w in response_lower for w in ["yes", "true", "match", "correct", "valid"])
            success = success and not any(
                w in response_lower for w in ["no", "false", "not match", "invalid"]
            )
        else:
            success = any(
                w in response_lower
                for w in ["no", "false", "not match", "doesn't match", "invalid"]
            )
        return success, {"expected": expected, "response_snippet": response_lower[:100]}

    # Some recognition-style tasks (e.g. mutation-effect questions) do not have a strict yes/no
    # answer, but specify required_keywords for a free-text explanation.
    keywords = task.metadata.get("required_keywords", [])
    if keywords:
        found_keywords = [kw for kw in keywords if str(kw).lower() in response_lower]
        score = len(found_keywords) / len(keywords) if keywords else 1.0
        success = score >= 0.7
        return success, {
            "required_keywords": keywords,
            "found_keywords": found_keywords,
            "coverage_score": score,
        }

    return False, {"error": "No expected answer provided"}


def evaluate_synthesis_task(task: RegexTask, response: str) -> tuple[bool, dict[str, Any]]:
    """Evaluate a synthesis task (write a regex)."""
    pattern, pick_details = pick_best_regex_candidate(response, test_cases=task.test_cases)

    if not pattern:
        return False, {"error": "Could not extract regex pattern", "pick": pick_details}

    # Test against provided test cases
    if task.test_cases:
        success, case_details = check_regex_against_cases(pattern, task.test_cases)
        return success, {
            "extracted_pattern": pattern,
            "test_results": case_details,
            "pick": pick_details,
        }

    # If no test cases, check if it compiles
    try:
        re.compile(pattern)
        return True, {
            "extracted_pattern": pattern,
            "note": "No test cases, only checked compilation",
        }
    except re.error as e:
        return False, {"extracted_pattern": pattern, "error": str(e)}


def evaluate_explanation_task(task: RegexTask, response: str) -> tuple[bool, dict[str, Any]]:
    """Evaluate an explanation task (explain what this regex matches)."""
    # Check for key concepts that should be mentioned
    keywords = task.metadata.get("required_keywords", [])
    response_lower = response.lower()

    found_keywords = [kw for kw in keywords if kw.lower() in response_lower]
    score = len(found_keywords) / len(keywords) if keywords else 1.0

    # Success if >= 70% of keywords mentioned
    success = score >= 0.7

    return success, {
        "required_keywords": keywords,
        "found_keywords": found_keywords,
        "coverage_score": score,
    }


def evaluate_debugging_task(task: RegexTask, response: str) -> tuple[bool, dict[str, Any]]:
    """Evaluate a debugging task (fix this regex)."""
    pattern, pick_details = pick_best_regex_candidate(response, test_cases=task.test_cases)

    if not pattern:
        return False, {"error": "Could not extract fixed regex pattern", "pick": pick_details}

    # Test the fixed pattern against test cases
    if task.test_cases:
        success, case_details = check_regex_against_cases(pattern, task.test_cases)
        return success, {
            "fixed_pattern": pattern,
            "test_results": case_details,
            "pick": pick_details,
        }

    return False, {"error": "No test cases provided for debugging task"}


def evaluate_refactoring_task(task: RegexTask, response: str) -> tuple[bool, dict[str, Any]]:
    """Evaluate a refactoring task (simplify without changing behavior)."""
    pattern, pick_details = pick_best_regex_candidate(response, test_cases=task.test_cases)

    if not pattern:
        return False, {"error": "Could not extract refactored regex pattern", "pick": pick_details}

    # Must pass all test cases
    if task.test_cases:
        success, case_details = check_regex_against_cases(pattern, task.test_cases)
        if not success:
            return False, {"refactored_pattern": pattern, "test_results": case_details}

    # Compute complexity metrics
    new_metrics = analyze_regex(pattern)
    details: dict[str, Any] = {"refactored_pattern": pattern, "pick": pick_details}

    if task.target_regex:
        old_metrics = analyze_regex(task.target_regex)
        details["original_complexity"] = old_metrics.complexity_score
        details["new_complexity"] = new_metrics.complexity_score
        details["improvement"] = old_metrics.complexity_score - new_metrics.complexity_score

        # Success if complexity is reduced or equal (and still passes tests)
        return new_metrics.complexity_score <= old_metrics.complexity_score, details

    return True, details


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

EVALUATORS: dict[CapabilityAxis, Callable[[RegexTask, str], tuple[bool, dict[str, Any]]]] = {
    CapabilityAxis.RECOGNITION: evaluate_recognition_task,
    CapabilityAxis.SYNTHESIS: evaluate_synthesis_task,
    CapabilityAxis.EXPLANATION: evaluate_explanation_task,
    CapabilityAxis.DEBUGGING: evaluate_debugging_task,
    CapabilityAxis.REFACTORING: evaluate_refactoring_task,
}


class RegexGeneralizationEvaluator:
    """Main evaluator for regex generalization tasks."""

    def __init__(self, tasks: list[RegexTask]):
        self.tasks = tasks

    @classmethod
    def from_jsonl(cls, path: Path) -> RegexGeneralizationEvaluator:
        """Load tasks from JSONL file."""
        tasks = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tasks.append(RegexTask.from_dict(data))
        return cls(tasks)

    def evaluate_single(self, task: RegexTask, response: str) -> TaskResult:
        """Evaluate a single task."""
        evaluator = EVALUATORS.get(task.capability)
        if not evaluator:
            return TaskResult(
                task_id=task.task_id,
                capability=task.capability,
                holdout_type=task.holdout_type,
                mutation_type=task.mutation_type,
                success=False,
                response=response,
                details={"error": f"No evaluator for capability {task.capability}"},
            )

        success, details = evaluator(task, response)

        # Compute regex metrics if synthesis/debugging/refactoring
        metrics = None
        if task.capability in [
            CapabilityAxis.SYNTHESIS,
            CapabilityAxis.DEBUGGING,
            CapabilityAxis.REFACTORING,
        ]:
            pattern, _ = pick_best_regex_candidate(response, test_cases=task.test_cases)
            if pattern:
                try:
                    re.compile(pattern)
                    metrics = analyze_regex(pattern)
                except re.error:
                    pass

        return TaskResult(
            task_id=task.task_id,
            capability=task.capability,
            holdout_type=task.holdout_type,
            mutation_type=task.mutation_type,
            success=success,
            response=response,
            metrics=metrics,
            details=details,
        )

    def evaluate_all(self, model_runner: ModelRunner, verbose: bool = False) -> EvalReport:
        """Evaluate all tasks using the provided model runner."""
        results: list[TaskResult] = []

        for i, task in enumerate(self.tasks):
            if verbose:
                print(f"Evaluating task {i + 1}/{len(self.tasks)}: {task.task_id}")

            response = model_runner(task.prompt)
            result = self.evaluate_single(task, response)
            results.append(result)

            if verbose:
                status = "✓" if result.success else "✗"
                print(f"  {status} {task.capability.value}")

        return self._compile_report(results)

    def evaluate_responses(self, responses: dict[str, str]) -> EvalReport:
        """Evaluate pre-computed responses (task_id -> response)."""
        results: list[TaskResult] = []

        for task in self.tasks:
            response = responses.get(task.task_id, "")
            result = self.evaluate_single(task, response)
            results.append(result)

        return self._compile_report(results)

    def _compile_report(self, results: list[TaskResult]) -> EvalReport:
        """Compile results into a comprehensive report."""
        total = len(results)
        correct = sum(1 for r in results if r.success)

        # Capability breakdown
        capability_stats: dict[str, dict[str, float]] = {}
        for cap in CapabilityAxis:
            cap_results = [r for r in results if r.capability == cap]
            if cap_results:
                cap_correct = sum(1 for r in cap_results if r.success)
                capability_stats[cap.value] = {
                    "total": len(cap_results),
                    "correct": cap_correct,
                    "accuracy": cap_correct / len(cap_results),
                }

        # Hold-out breakdown
        holdout_stats: dict[str, dict[str, float]] = {}
        for ho in HoldOutType:
            ho_results = [r for r in results if r.holdout_type == ho]
            if ho_results:
                ho_correct = sum(1 for r in ho_results if r.success)
                holdout_stats[ho.value] = {
                    "total": len(ho_results),
                    "correct": ho_correct,
                    "accuracy": ho_correct / len(ho_results),
                }

        # Mutation breakdown
        mutation_stats: dict[str, dict[str, float]] = {}
        for mt in MutationType:
            mt_results = [r for r in results if r.mutation_type == mt]
            if mt_results:
                mt_correct = sum(1 for r in mt_results if r.success)
                mutation_stats[mt.value] = {
                    "total": len(mt_results),
                    "correct": mt_correct,
                    "accuracy": mt_correct / len(mt_results),
                }

        # Simplicity stats (for successful synthesis tasks)
        complexity_scores = []
        for r in results:
            if r.success and r.metrics:
                complexity_scores.append(r.metrics.complexity_score)

        simplicity_stats: dict[str, float] = {}
        if complexity_scores:
            simplicity_stats = {
                "mean_complexity": statistics.mean(complexity_scores),
                "median_complexity": statistics.median(complexity_scores),
                "min_complexity": min(complexity_scores),
                "max_complexity": max(complexity_scores),
                "stdev_complexity": (
                    statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0.0
                ),
            }

        return EvalReport(
            total_tasks=total,
            total_correct=correct,
            overall_accuracy=correct / total if total else 0.0,
            capability_breakdown=capability_stats,
            holdout_breakdown=holdout_stats,
            mutation_breakdown=mutation_stats,
            simplicity_stats=simplicity_stats,
            task_results=results,
        )


# ---------------------------------------------------------------------------
# Comparison Utilities
# ---------------------------------------------------------------------------


def compare_reports(
    report_a: EvalReport, report_b: EvalReport, label_a: str = "A", label_b: str = "B"
) -> dict[str, Any]:
    """Compare two evaluation reports (e.g., SFT vs Evolved LoRA)."""
    comparison = {
        "overall": {
            label_a: {"accuracy": report_a.overall_accuracy, "total": report_a.total_tasks},
            label_b: {"accuracy": report_b.overall_accuracy, "total": report_b.total_tasks},
            "delta": report_b.overall_accuracy - report_a.overall_accuracy,
        },
        "capability_comparison": {},
        "holdout_comparison": {},
        "mutation_comparison": {},
        "simplicity_comparison": {},
    }

    # Compare capabilities
    all_caps = set(report_a.capability_breakdown.keys()) | set(report_b.capability_breakdown.keys())
    for cap in all_caps:
        acc_a = report_a.capability_breakdown.get(cap, {}).get("accuracy", 0.0)
        acc_b = report_b.capability_breakdown.get(cap, {}).get("accuracy", 0.0)
        comparison["capability_comparison"][cap] = {
            label_a: acc_a,
            label_b: acc_b,
            "delta": acc_b - acc_a,
        }

    # Compare hold-out types
    all_holdouts = set(report_a.holdout_breakdown.keys()) | set(report_b.holdout_breakdown.keys())
    for ho in all_holdouts:
        acc_a = report_a.holdout_breakdown.get(ho, {}).get("accuracy", 0.0)
        acc_b = report_b.holdout_breakdown.get(ho, {}).get("accuracy", 0.0)
        comparison["holdout_comparison"][ho] = {
            label_a: acc_a,
            label_b: acc_b,
            "delta": acc_b - acc_a,
        }

    # Compare mutation types
    all_mutations = set(report_a.mutation_breakdown.keys()) | set(
        report_b.mutation_breakdown.keys()
    )
    for mt in all_mutations:
        acc_a = report_a.mutation_breakdown.get(mt, {}).get("accuracy", 0.0)
        acc_b = report_b.mutation_breakdown.get(mt, {}).get("accuracy", 0.0)
        comparison["mutation_comparison"][mt] = {
            label_a: acc_a,
            label_b: acc_b,
            "delta": acc_b - acc_a,
        }

    # Compare simplicity
    if report_a.simplicity_stats and report_b.simplicity_stats:
        comparison["simplicity_comparison"] = {
            label_a: report_a.simplicity_stats.get("mean_complexity", 0.0),
            label_b: report_b.simplicity_stats.get("mean_complexity", 0.0),
            "delta": report_b.simplicity_stats.get("mean_complexity", 0.0)
            - report_a.simplicity_stats.get("mean_complexity", 0.0),
            "note": "Lower complexity is better",
        }

    return comparison
