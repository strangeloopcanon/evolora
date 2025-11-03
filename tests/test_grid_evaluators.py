from symbiont_ecology.environment.grid import GridTask


def test_word_count_spelled_number():
    task = GridTask(
        task_id="t1",
        cell=("word.count", "short"),
        prompt="Count words",
        price=1.0,
        target=13,
        family="word.count",
        depth="short",
        difficulty=0.5,
    )
    ok, _ = task.evaluate("I think the answer is thirteen")
    assert ok is True


def test_json_repair_bracket_extraction():
    task = GridTask(
        task_id="t2",
        cell=("json_repair", "short"),
        prompt="Make JSON",
        price=1.0,
        target=[1, 2, 3],
        family="json_repair",
        depth="short",
        difficulty=0.5,
    )
    ok, _ = task.evaluate("Final output: [1, 2, 3] thanks!")
    assert ok is True


def test_logic_bool_accepts_yes_no():
    task = GridTask(
        task_id="t3",
        cell=("logic.bool", "short"),
        prompt="True/False",
        price=1.0,
        target=True,
        family="logic.bool",
        depth="short",
        difficulty=0.5,
    )
    ok, _ = task.evaluate("Answer: yes")
    assert ok is True

    task2 = GridTask(
        task_id="t5",
        cell=("logic.bool", "short"),
        prompt="True/False",
        price=1.0,
        target=False,
        family="logic.bool",
        depth="short",
        difficulty=0.5,
    )
    ok2, _ = task2.evaluate("no")
    assert ok2 is True


def test_math_parses_scientific_notation():
    task = GridTask(
        task_id="t4",
        cell=("math", "short"),
        prompt="Number",
        price=1.0,
        target=12.5,
        family="math",
        depth="short",
        difficulty=0.5,
    )
    ok, _ = task.evaluate("The result is 1.25e1")
    assert ok is True
