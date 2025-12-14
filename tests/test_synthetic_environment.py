from symbiont_ecology.environment.synthetic import (
    SyntheticTask,
    TaskFactory,
    _is_prime,
    evaluate_population_responses,
)


def test_synthetic_factory_and_scoring() -> None:
    factory = TaskFactory(seed=3)
    tasks = factory.sample(4)
    assert {task.kind for task in tasks}
    responses = {}
    for idx, task in enumerate(tasks):
        answer = task.expected_answer()
        responses[f"org_{idx}"] = (answer, 0.2 * idx)
        reward = task.score(answer, energy_spent=0.2 * idx)
        assert reward.task_reward == 1.0
    rewards = evaluate_population_responses(tasks, responses)
    assert len(rewards) == len(responses)
    # Advance phase and ensure difficulty increases for sorting task
    factory.advance_phase()
    advanced_tasks = factory.sample(10)
    sort_tasks = [task for task in advanced_tasks if task.kind == "string.sort"]
    if sort_tasks:
        assert sort_tasks[0].difficulty >= 0.45  # phase-adjusted difficulty


def test_synthetic_expected_answers_cover_all_branches() -> None:
    cases = [
        ("math.add", {"a": 3, "b": 4}, "7"),
        ("math.mul", {"a": 2, "b": 5}, "10"),
        ("string.reverse", {"text": "abc"}, "cba"),
        ("string.sort", {"letters": "c b a"}, "abc"),
        ("word.count", {"sentence": "one two three"}, "3"),
        ("math.prime", {"upper": 7}, "2,3,5,7"),
    ]
    for kind, payload, expected in cases:
        task = SyntheticTask(task_id="t", kind=kind, prompt="", payload=payload, difficulty=0.2)
        assert task.expected_answer() == expected
        reward = task.score("wrong", energy_spent=0.5)
        assert reward.task_reward in {0.0, 1.0}


def test_synthetic_helpers_edge_cases() -> None:
    assert _is_prime(2) is True
    assert _is_prime(1) is False
    assert _is_prime(9) is False
    empty_rewards = evaluate_population_responses([], {})
    assert empty_rewards == {}
