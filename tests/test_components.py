import torch

from symbiont_ecology import (
    ATPLedger,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    PopulationManager,
    TelemetrySink,
)
from symbiont_ecology.environment.bridge import EchoTool, ToolRegistry
from symbiont_ecology.environment.human import HumanBandit
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.morphogenesis import MorphogenesisController
from symbiont_ecology.evolution.population import Genome
from symbiont_ecology.metrics.telemetry import AssimilationEvent, EpisodeLog, RewardBreakdown
from symbiont_ecology.utils.torch_utils import clamp_norm, ensure_dtype, no_grad, resolve_device


def test_assimilation_tester_passes_when_uplift_high() -> None:
    tester = AssimilationTester(uplift_threshold=0.05, p_value_threshold=0.05, safety_budget=0)
    result = tester.evaluate(
        organelle_id="org_a",
        control_scores=[0.7, 0.72, 0.68],
        treatment_scores=[0.8, 0.82, 0.81],
        safety_hits=0,
        energy_cost=1.0,
    )
    assert result.decision is True
    assert result.event.uplift > 0


def test_morphogenesis_triggers_resize_and_merge() -> None:
    config = EcologyConfig()
    config.limits.lora_budget_frac = 1.0
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    organelle_id = host.spawn_organelle(rank=2)
    population = PopulationManager(config.evolution, config.foraging)
    genome = Genome(organelle_id=organelle_id, drive_weights={}, gate_bias=0.0, rank=2)
    population.register(genome)
    population.record_roi(organelle_id, 1.5)
    controller = MorphogenesisController(config=config, host=host)
    controller.apply([genome], population)
    assert host.get_organelle(organelle_id).get_rank() >= 3  # type: ignore[attr-defined]


def test_tool_registry_and_human_bandit() -> None:
    registry = ToolRegistry({"echo": EchoTool()})
    assert registry.call("echo", text="hello") == "hello"
    bandit = HumanBandit()
    feedback = bandit.solicit("prompt", "Answer with Thanks")
    assert feedback.reward.helper_bonus >= 0.0
    batch_feedback = bandit.batch("prompt", ["Thanks", ""])
    assert len(batch_feedback) == 2


def test_torch_utils_helpers() -> None:
    vector = torch.tensor([3.0, 4.0])
    clamped = clamp_norm(vector.clone(), max_norm=5.0)
    assert torch.allclose(clamped, vector)
    half = ensure_dtype(vector.double(), torch.float32)
    assert half.dtype == torch.float32
    with no_grad():
        vector.add_(1.0)
    assert resolve_device("cpu").type == "cpu"


def test_telemetry_sink_writes(tmp_path) -> None:
    sink = TelemetrySink(tmp_path, "episodes.jsonl", "assimilation.jsonl")
    episode = EpisodeLog(
        episode_id="epi_1",
        task_id="task-1",
        organelles=["org_a"],
        rewards=RewardBreakdown(
            task_reward=1.0,
            novelty_bonus=0.0,
            competence_bonus=0.0,
            helper_bonus=0.0,
            risk_penalty=0.0,
            cost_penalty=0.0,
        ),
        energy_spent=0.5,
        observations={"prompt": "Add 1 and 2", "answer": "3"},
    )
    sink.log_episode(episode)
    event = AssimilationEvent(
        organelle_id="org_a",
        uplift=0.1,
        p_value=0.001,
        passed=True,
        energy_cost=1.0,
        safety_hits=0,
        cell={"family": "math", "depth": "short"},
    )
    sink.log_assimilation(event, decision=True)
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "assimilation.jsonl").exists()
