from symbiont_ecology.environment.grid import GridTask


def test_word_count_spelled_number() -> None:
    task = GridTask(
        task_id="w1",
        cell=("word.count", "short"),
        prompt="Count words",
        price=1.0,
        target=3,
        family="word.count",
        depth="short",
        difficulty=0.1,
    )
    # Model might emit a spelled number
    success, _ = task.evaluate("There are three words.")
    assert success is True
