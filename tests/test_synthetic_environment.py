from symbiont_ecology.environment.synthetic import TaskFactory, evaluate_population_responses


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
