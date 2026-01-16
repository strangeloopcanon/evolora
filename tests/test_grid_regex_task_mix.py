from symbiont_ecology.environment.grid import GridTask


def test_regex_recognition_evaluator() -> None:
    task = GridTask(
        task_id="r1",
        cell=("regex.recognition", "short"),
        prompt='Does "^a+$" match "aaa"? Answer yes or no.',
        price=1.0,
        target={"expected": True},
        family="regex.recognition",
        depth="short",
        difficulty=0.5,
    )
    ok, _reward = task.evaluate("yes")
    assert ok is True
    ok2, _reward2 = task.evaluate("no")
    assert ok2 is False


def test_regex_explanation_evaluator_keyword_threshold() -> None:
    task = GridTask(
        task_id="e1",
        cell=("regex.explanation", "short"),
        prompt="Explain regex",
        price=1.0,
        target={"required_keywords": ["year", "month", "day", "hour"]},
        family="regex.explanation",
        depth="short",
        difficulty=0.5,
    )
    ok, _reward = task.evaluate("Matches year month day but not minutes.")
    assert ok is True  # 3/4 >= 0.7
    ok2, _reward2 = task.evaluate("Matches year and month.")
    assert ok2 is False  # 2/4 < 0.7


def test_regex_refactoring_requires_not_more_complex() -> None:
    test_strings = [
        {"string": "123", "should_match": True},
        {"string": "12", "should_match": False},
        {"string": "1234", "should_match": False},
    ]
    task = GridTask(
        task_id="f1",
        cell=("regex.refactoring", "short"),
        prompt="Simplify regex",
        price=1.0,
        target={"original_pattern": "^\\d{3}$", "test_strings": test_strings},
        family="regex.refactoring",
        depth="short",
        difficulty=0.5,
    )
    ok, _reward = task.evaluate("^(?:[0-9][0-9][0-9])$")
    assert ok is False
