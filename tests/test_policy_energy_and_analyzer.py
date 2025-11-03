import json
from types import SimpleNamespace

from symbiont_ecology import ATPLedger, EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import Genome, PopulationManager
import importlib.util
from pathlib import Path

_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ecology_run.py"
spec = importlib.util.spec_from_file_location("_analyzer", _ANALYZER_PATH)
assert spec and spec.loader
_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_analyzer)
summarise_generations = _analyzer.summarise_generations


def _build_loop_with_fake_host() -> tuple[EcologyLoop, str, ATPLedger]:
    cfg = EcologyConfig()
    cfg.policy.enabled = True
    cfg.policy.energy_cost = 0.1
    cfg.policy.charge_tokens = False
    # Minimal population with one genome
    pop = PopulationManager(cfg.evolution)
    organelle_id = "org_test"
    pop.register(
        Genome(organelle_id=organelle_id, drive_weights={}, gate_bias=0.0, rank=2)
    )
    # Fake host that returns a one-line JSON answer and minimal metrics
    ledger = ATPLedger()
    ledger.ensure_energy(organelle_id, 1.0)

    def fake_step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids: list[str]):  # type: ignore[override]
        policy = {"budget_frac": 1.2, "reserve_ratio": 0.3}
        answer = json.dumps(policy)
        metrics = SimpleNamespace(tokens=32)
        envelope = SimpleNamespace(observation=SimpleNamespace(state={"answer": answer}))
        return SimpleNamespace(envelope=envelope, responses={organelle_id: metrics})

    host = SimpleNamespace(step=fake_step, ledger=ledger)
    loop = EcologyLoop(
        config=cfg,
        host=host,  # type: ignore[arg-type]
        environment=SimpleNamespace(),  # not used in policy path
        population=pop,
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    return loop, organelle_id, ledger


def test_policy_energy_is_charged_and_recorded() -> None:
    loop, oid, ledger = _build_loop_with_fake_host()
    before = ledger.energy_balance(oid)
    loop._request_and_apply_policy(oid)
    after = ledger.energy_balance(oid)
    # Fixed micro-cost should be deducted
    assert after < before
    assert abs((before - after) - loop.config.policy.energy_cost) < 1e-6
    # Internal accounting updated
    assert getattr(loop, "_policy_cost_total", 0.0) >= loop.config.policy.energy_cost
    # Policy stored
    assert oid in loop._active_policies


def test_policy_energy_scaled_by_tokens_when_enabled() -> None:
    # Rebuild with token-scaled charging enabled
    loop, oid, ledger = _build_loop_with_fake_host()
    loop.config.policy.charge_tokens = True
    loop.config.policy.energy_cost = 0.1
    loop.config.policy.token_cap = 64
    before = ledger.energy_balance(oid)
    loop._request_and_apply_policy(oid)
    after = ledger.energy_balance(oid)
    # metrics.tokens=32 in the fake, so expect half the micro-cost
    assert abs((before - after) - 0.05) < 1e-6


def test_sample_task_with_policy_bias() -> None:
    # Minimal loop with environment stub to exercise policy routing bias
    cfg = EcologyConfig()
    cfg.policy.enabled = True
    cfg.policy.bias_strength = 1.0  # always honor policy
    pop = PopulationManager(cfg.evolution)
    oid = "org_test2"
    pop.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2))
    env = SimpleNamespace(
        rng=SimpleNamespace(random=lambda: 0.0),
        sample_task_from_cell=lambda cell, canary=False: SimpleNamespace(prompt="p", family=cell[0], depth=cell[1], cell=cell),
        controller=SimpleNamespace(),
    )
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),  # unused in this path
        environment=env,  # type: ignore[arg-type]
        population=pop,
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    loop._active_policies[oid] = {"cell_pref": {"family": "word.count", "depth": "short"}}
    task = loop._sample_task_with_policy(lp_mix=0.0, organelle_id=oid)
    assert task.family == "word.count"
    assert task.depth == "short"


def test_resolve_lp_mix_auto_tune_bounds() -> None:
    cfg = EcologyConfig()
    cfg.curriculum.alp_auto_mix = True
    cfg.curriculum.lp_mix = 0.2
    cfg.curriculum.lp_mix_min = 0.1
    cfg.curriculum.lp_mix_max = 0.3
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),  # unused
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={"a": 0.1, "b": 0.4})),  # type: ignore[arg-type]
        population=PopulationManager(cfg.evolution),
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    val = loop._resolve_lp_mix(cfg.curriculum.lp_mix)
    assert cfg.curriculum.lp_mix_min <= val <= cfg.curriculum.lp_mix_max


def test_compute_batch_size_branches() -> None:
    cfg = EcologyConfig()
    cfg.environment.auto_batch = True
    cfg.environment.batch_min = 1
    cfg.environment.batch_max = 4
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={})),  # type: ignore[arg-type]
        population=PopulationManager(cfg.evolution),
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    # ROI low -> min
    loop.population.aggregate_roi = lambda limit=5: 0.4  # type: ignore[assignment]
    assert loop._compute_batch_size(3) == cfg.environment.batch_min
    # ROI high -> max
    loop.population.aggregate_roi = lambda limit=5: 1.6  # type: ignore[assignment]
    assert loop._compute_batch_size(3) == cfg.environment.batch_max
    # ROI middle -> between
    loop.population.aggregate_roi = lambda limit=5: 1.0  # type: ignore[assignment]
    mid = loop._compute_batch_size(3)
    assert cfg.environment.batch_min <= mid <= cfg.environment.batch_max


def test_auto_nudge_evidence_adjusts_knobs() -> None:
    cfg = EcologyConfig()
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={})),  # type: ignore[arg-type]
        population=PopulationManager(cfg.evolution),
        assimilation=AssimilationTester(
            uplift_threshold=cfg.evolution.assimilation_threshold,
            p_value_threshold=cfg.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    # Force a stall and many gate hits
    loop.assim_fail_streak = 12
    before = cfg.assimilation_tuning.min_window
    summary = {"assimilation_gating": {"low_power": 3, "uplift_below_threshold": 1, "insufficient_scores": 60, "topup_roi_blocked": 6}, "promotions": 0, "merges": 0}
    loop._auto_nudge_evidence(summary)
    after = cfg.assimilation_tuning.min_window
    assert after >= before


def test_decay_assimilation_thresholds_on_fail_streak() -> None:
    cfg = EcologyConfig()
    # Ensure deterministic starting threshold
    cfg.evolution.assimilation_threshold = 0.02
    class DummyAssim:
        def __init__(self):
            self.p_value_threshold = cfg.evolution.assimilation_p_value
            self.calls = []
        def update_thresholds(self, uplift_threshold: float, p_value_threshold: float | None = None) -> None:
            self.calls.append((uplift_threshold, p_value_threshold))
    dummy = DummyAssim()
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={})),  # type: ignore[arg-type]
        population=PopulationManager(cfg.evolution),
        assimilation=dummy,  # type: ignore[arg-type]
        human_bandit=None,
        sink=None,
    )
    loop.assim_fail_streak = 10
    before = cfg.evolution.assimilation_threshold
    loop._maybe_decay_assimilation_thresholds()
    after = cfg.evolution.assimilation_threshold
    assert after <= before
    assert dummy.calls, "expected assimilation.update_thresholds to be called"


def test_sanitize_telemetry_handles_various_types() -> None:
    loop = EcologyLoop(
        config=EcologyConfig(),
        host=SimpleNamespace(),
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={})),  # type: ignore[arg-type]
        population=PopulationManager(EcologyConfig().evolution),
        assimilation=AssimilationTester(
            uplift_threshold=0.01,
            p_value_threshold=0.05,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )
    class Weird:
        def __str__(self) -> str:
            return "weird"
    data = {
        "a": float("inf"),
        "b": [1, 2.0, False, (3, 4.5)],
        "c": {"x": True, "y": None},
        "d": Weird(),
    }
    out = loop._sanitize_telemetry(data)
    # ensure it became JSON-serializable-ish
    assert out["a"] == 0.0
    assert out["b"][3][1] == 4.5
    assert out["c"]["x"] is True
    assert out["d"] == "weird"


def test_analyzer_policy_conditioning_and_fields_aggregate() -> None:
    records = [
        {"generation": 1, "avg_roi": 1.0, "policy_applied": 1, "policy_fields_used": {"budget_frac": 1}, "policy_budget_frac_avg": 1.2, "policy_attempts": 2, "policy_parsed": 1},
        {"generation": 2, "avg_roi": 0.5, "policy_applied": 0, "policy_attempts": 1, "policy_parsed": 0},
    ]
    summary = summarise_generations(records)
    assert summary["policy_applied_total"] == 1
    assert summary["policy_roi_mean_when_applied"] == 1.0
    assert summary["policy_roi_mean_when_not"] == 0.5
    assert summary["policy_fields_used_total"].get("budget_frac") == 1
    assert summary["policy_parse_attempts_total"] == 3
    assert summary["policy_parse_parsed_total"] == 1


def test_config_loader_fallback_parser(tmp_path) -> None:
    # Force fallback by simulating missing OmegaConf
    import sys
    from symbiont_ecology.config import load_ecology_config

    original = sys.modules.get("omegaconf")
    sys.modules["omegaconf"] = None  # type: ignore[assignment]
    try:
        yaml_text = """
host:
  backbone_model: google/gemma-3-270m-it
policy:
  enabled: true
"""
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml_text)
        cfg = load_ecology_config(p)
        assert cfg.host.backbone_model == "google/gemma-3-270m-it"
        assert cfg.policy.enabled is True
    finally:
        if original is not None:
            sys.modules["omegaconf"] = original
        else:
            sys.modules.pop("omegaconf", None)


def test_config_loader_omegaconf_parse_failure_fallback(tmp_path) -> None:
    # Simulate OmegaConf import succeeds but .load raises, exercising inner except branch
    import sys
    from types import SimpleNamespace
    from symbiont_ecology.config import load_ecology_config

    class _DummyOC:
        @staticmethod
        def load(_path):
            raise ValueError("parse error")

        @staticmethod
        def to_container(_conf, resolve=True):  # pragma: no cover - not reached
            return {}

    sys.modules["omegaconf"] = SimpleNamespace(OmegaConf=_DummyOC)  # type: ignore[assignment]
    try:
        p = tmp_path / "bad.yaml"
        p.write_text("policy:\n  enabled: true\n")
        cfg = load_ecology_config(p)
        assert cfg.policy.enabled is True
    finally:
        sys.modules.pop("omegaconf", None)
