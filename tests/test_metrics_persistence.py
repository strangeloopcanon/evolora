from symbiont_ecology.metrics.persistence import TelemetrySink
from symbiont_ecology.metrics.telemetry import AssimilationEvent, EpisodeLog, RewardBreakdown


def test_telemetry_sink_writes(tmp_path) -> None:
    sink = TelemetrySink(root=tmp_path, episodes_file="episodes.jsonl", assimilation_file="assim.jsonl")
    episode = EpisodeLog(
        episode_id="epi_1",
        task_id="task_1",
        organelles=["orgA"],
        rewards=RewardBreakdown(
            task_reward=1.0,
            novelty_bonus=0.1,
            competence_bonus=0.2,
            helper_bonus=0.0,
            risk_penalty=0.0,
            cost_penalty=0.05,
        ),
        energy_spent=0.5,
        observations={"prompt": "Add 2 and 2", "answer": "4"},
    )
    sink.log_episode(episode)
    event = AssimilationEvent(
        organelle_id="orgA",
        uplift=0.05,
        p_value=0.01,
        passed=True,
        energy_cost=0.2,
        safety_hits=0,
        cell={"family": "math", "depth": "short"},
    )
    sink.log_assimilation(event, decision=True)
    episodes_path = tmp_path / "episodes.jsonl"
    assim_path = tmp_path / "assim.jsonl"
    assert episodes_path.exists()
    assert assim_path.exists()
    import json

    episode_record = json.loads(episodes_path.read_text().strip())
    assert episode_record["type"] == "episode"
    assimilation_record = json.loads(assim_path.read_text().strip())
    assert assimilation_record["type"] == "assimilation"
    assert assimilation_record["cell"] == {"family": "math", "depth": "short"}
    assert assimilation_record["probes"] == []
    assert assimilation_record["soup"] == []
