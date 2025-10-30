from symbiont_ecology.environment.grid import GridTask


def test_math_parsing_relaxed() -> None:
    task = GridTask(
        task_id="t1",
        cell=("math", "short"),
        prompt="Add 10 and 2. Respond with the number only.",
        price=1.0,
        target=12.0,
        family="math",
        depth="short",
        difficulty=0.3,
    )
    # Answer contains prose; evaluator should extract the first numeric token
    success, _ = task.evaluate("The answer is 12.")
    assert success is True


def test_json_repair_bracket_extraction() -> None:
    task = GridTask(
        task_id="t2",
        cell=("json_repair", "short"),
        prompt="Produce a valid JSON array.",
        price=1.0,
        target=[1, 2, 3],
        family="json_repair",
        depth="short",
        difficulty=0.3,
    )
    # Answer wraps the JSON; evaluator should extract bracketed array and parse
    success, _ = task.evaluate("Here you go: [1, 2, 3]")
    assert success is True

