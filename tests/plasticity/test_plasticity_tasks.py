"""CI-safe tests for plasticity task generators."""

from __future__ import annotations

import re

import pytest

from plasticity.tasks import (
    MathTaskGen,
    RegexTaskGen,
    WordCountTaskGen,
    generate_tasks,
)


class TestRegexTaskGen:
    def test_generates_correct_count(self) -> None:
        tasks = RegexTaskGen().generate(10, seed=42)
        assert len(tasks) == 10

    def test_all_tasks_are_regex_family(self) -> None:
        tasks = RegexTaskGen().generate(5, seed=1)
        for t in tasks:
            assert t.family == "regex"

    def test_target_pattern_is_valid_regex(self) -> None:
        tasks = RegexTaskGen().generate(20, seed=99)
        for t in tasks:
            re.compile(t.target)

    def test_target_pattern_passes_own_evaluator(self) -> None:
        tasks = RegexTaskGen().generate(20, seed=99)
        for t in tasks:
            assert t.evaluate(t.target), f"Target pattern failed its own evaluator: {t.target}"

    def test_deterministic_with_same_seed(self) -> None:
        a = RegexTaskGen().generate(5, seed=42)
        b = RegexTaskGen().generate(5, seed=42)
        for ta, tb in zip(a, b):
            assert ta.prompt == tb.prompt
            assert ta.target == tb.target

    def test_different_seeds_give_different_tasks(self) -> None:
        a = RegexTaskGen().generate(10, seed=1)
        b = RegexTaskGen().generate(10, seed=2)
        prompts_a = {t.prompt for t in a}
        prompts_b = {t.prompt for t in b}
        assert prompts_a != prompts_b

    def test_evaluator_rejects_garbage(self) -> None:
        tasks = RegexTaskGen().generate(5, seed=42)
        for t in tasks:
            assert not t.evaluate("")
            assert not t.evaluate("[invalid regex (((")


class TestMathTaskGen:
    def test_generates_correct_count(self) -> None:
        tasks = MathTaskGen().generate(10, seed=42)
        assert len(tasks) == 10

    def test_all_tasks_are_math_family(self) -> None:
        for t in MathTaskGen().generate(5, seed=1):
            assert t.family == "math"

    def test_correct_answer_passes(self) -> None:
        tasks = MathTaskGen().generate(20, seed=42)
        for t in tasks:
            assert t.evaluate(str(t.target)), f"Correct answer failed: {t.target}"

    def test_wrong_answer_fails(self) -> None:
        tasks = MathTaskGen().generate(5, seed=42)
        for t in tasks:
            wrong = float(t.target) + 9999
            assert not t.evaluate(str(wrong))

    def test_evaluator_extracts_number_from_text(self) -> None:
        tasks = MathTaskGen().generate(3, seed=42)
        for t in tasks:
            padded = f"The answer is {t.target} I think"
            assert t.evaluate(padded)

    def test_deterministic(self) -> None:
        a = MathTaskGen().generate(5, seed=7)
        b = MathTaskGen().generate(5, seed=7)
        for ta, tb in zip(a, b):
            assert ta.target == tb.target


class TestWordCountTaskGen:
    def test_generates_correct_count(self) -> None:
        tasks = WordCountTaskGen().generate(10, seed=42)
        assert len(tasks) == 10

    def test_all_tasks_are_word_count_family(self) -> None:
        for t in WordCountTaskGen().generate(5, seed=1):
            assert t.family == "word.count"

    def test_correct_answer_passes(self) -> None:
        tasks = WordCountTaskGen().generate(20, seed=42)
        for t in tasks:
            assert t.evaluate(str(t.target)), f"Correct answer {t.target} failed for: {t.prompt}"

    def test_wrong_answer_fails(self) -> None:
        tasks = WordCountTaskGen().generate(5, seed=42)
        for t in tasks:
            wrong = int(t.target) + 100
            assert not t.evaluate(str(wrong))


class TestGenerateTasks:
    def test_dispatches_to_correct_generator(self) -> None:
        for family in ["regex", "math", "word.count"]:
            tasks = generate_tasks(family, 3, seed=42)
            assert len(tasks) == 3
            for t in tasks:
                assert t.family == family

    def test_unknown_family_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task family"):
            generate_tasks("nonexistent", 1)
