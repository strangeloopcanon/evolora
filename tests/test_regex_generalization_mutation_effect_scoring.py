from symbiont_ecology.evaluation.regex_generalization import (
    CapabilityAxis,
    RegexGeneralizationEvaluator,
    RegexTask,
)


def test_recognition_falls_back_to_keyword_scoring_when_expected_answer_missing() -> None:
    task = RegexTask(
        task_id="mutation_effect_smoke",
        prompt="What new strings will now match?",
        capability=CapabilityAxis.RECOGNITION,
        expected_answer=None,
        metadata={"required_keywords": ["24", "25", "26", "27", "28", "29"]},
    )
    evaluator = RegexGeneralizationEvaluator([task])

    result = evaluator.evaluate_single(task, "New matches: 24 25 26 27 28")
    assert result.success is True
    assert result.details.get("coverage_score") is not None

    result2 = evaluator.evaluate_single(task, "24 25 26")
    assert result2.success is False
