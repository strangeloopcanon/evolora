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


def test_math_multi_step_evaluator():
    task = GridTask(
        task_id="m1",
        cell=("math.multi_step", "medium"),
        prompt="Compute (4 + 5) * 3 - 6.",
        price=1.0,
        target=21.0,
        family="math.multi_step",
        depth="medium",
        difficulty=0.4,
    )
    ok, _ = task.evaluate("21")
    assert ok is True


def test_code_format_equivalence():
    task = GridTask(
        task_id="c1",
        cell=("code.format", "medium"),
        prompt="Convert name",
        price=1.0,
        target="adaptive_signal_flow",
        family="code.format",
        depth="medium",
        difficulty=0.4,
    )
    ok, _ = task.evaluate("adaptive_signal_flow")
    assert ok is True
    ok_code_block, _ = task.evaluate("```adaptive_signal_flow```")
    assert ok_code_block is True


def test_string_sort_accepts_json_array_targets():
    task = GridTask(
        task_id="s1",
        cell=("string.sort", "short"),
        prompt="Sort list",
        price=1.0,
        target=["alpha", "bravo"],
        family="string.sort",
        depth="short",
        difficulty=0.3,
    )
    ok, _ = task.evaluate('["alpha", "bravo"]')
    assert ok is True
    ok_py_list, _ = task.evaluate("['alpha', 'bravo']")
    assert ok_py_list is False


def test_supervised_completion_supports_multiobjective_families():
    import json

    math_task = GridTask(
        task_id="sc_math",
        cell=("math", "short"),
        prompt="Add",
        price=1.0,
        target=12.0,
        family="math",
        depth="short",
        difficulty=0.1,
    )
    assert math_task.supervised_completion() == "12"

    json_task = GridTask(
        task_id="sc_json",
        cell=("json_repair", "short"),
        prompt="JSON",
        price=1.0,
        target=[1, 2, 3],
        family="json_repair",
        depth="short",
        difficulty=0.1,
    )
    assert json_task.supervised_completion() == json.dumps([1, 2, 3], ensure_ascii=False)

    bool_task = GridTask(
        task_id="sc_bool",
        cell=("logic.bool", "short"),
        prompt="Bool",
        price=1.0,
        target=True,
        family="logic.bool",
        depth="short",
        difficulty=0.1,
    )
    assert bool_task.supervised_completion() == "True"

    sort_task = GridTask(
        task_id="sc_sort",
        cell=("string.sort", "short"),
        prompt="Sort letters",
        price=1.0,
        target="cba",
        family="string.sort",
        depth="short",
        difficulty=0.1,
    )
    assert sort_task.supervised_completion() == "abc"
