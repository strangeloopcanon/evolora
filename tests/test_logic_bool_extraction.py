from symbiont_ecology.environment.grid import GridTask


def test_logic_bool_markdown_wrapped() -> None:
    task = GridTask(
        task_id="b1",
        cell=("logic.bool", "short"),
        prompt="Eval",
        price=1.0,
        target=False,
        family="logic.bool",
        depth="short",
        difficulty=0.1,
    )
    success, _ = task.evaluate("**Answer:** FALSE\n**Explanation:** ...")
    assert success is True

