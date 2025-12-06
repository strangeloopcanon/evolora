from types import SimpleNamespace

import pytest

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager


class DummyLedger:
    def __init__(self, balances: dict[str, float]) -> None:
        self._balances = dict(balances)
        self.energy_cap = 10.0

    def energy_balance(self, organelle_id: str) -> float:
        return self._balances.get(organelle_id, 0.0)

    def credit_energy(self, organelle_id: str, amount: float) -> float:
        new_val = min(self.energy_cap, self._balances.get(organelle_id, 0.0) + amount)
        self._balances[organelle_id] = new_val
        return new_val

    def consume_energy(self, organelle_id: str, amount: float) -> bool:
        if self._balances.get(organelle_id, 0.0) < amount:
            return False
        self._balances[organelle_id] -= amount
        return True


class DummyHost:
    def __init__(self, balances: dict[str, float]) -> None:
        self.ledger = DummyLedger(balances)

    def list_organelle_ids(self) -> list[str]:
        return list(self.ledger._balances.keys())

    def get_organelle(self, organelle_id: str):  # pragma: no cover - unused in tests
        return SimpleNamespace(rank=2)

    def retire_organelle(self, organelle_id: str) -> None:
        self.ledger._balances.pop(organelle_id, None)


def _make_loop(balances: dict[str, float]) -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.assimilation_tuning.colonies_enabled = True
    cfg.assimilation_tuning.colony_expand_windows = 1
    cfg.assimilation_tuning.colony_expand_delta = 0.1
    cfg.assimilation_tuning.colony_variance_improve = 0.2
    cfg.assimilation_tuning.colony_hazard_bandwidth_scale = 0.25
    host = DummyHost(balances)
    env = SimpleNamespace(controller=SimpleNamespace(cells={}), rng=SimpleNamespace())
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    loop = EcologyLoop(cfg, host, env, pop, assimilation=SimpleNamespace())
    loop._sample_holdout_tasks = lambda: ["task"]
    return loop


def test_colony_expands_when_thresholds_met():
    loop = _make_loop({"a": 2.0, "b": 2.0, "c": 2.0})
    loop.config.assimilation_tuning.colony_reserve_ticket_multiplier = 0.0
    loop.config.assimilation_tuning.colony_reserve_ratio = 0.0
    for oid in ("a", "b", "c"):
        loop.population.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2))
        loop.population.energy[oid] = [0.1, 0.2]
        loop.population.roi[oid] = [1.0]

    stats_map: dict[tuple[str, ...], dict[str, float]] = {
        ("a",): {"mean": 1.0, "variance": 0.1},
        ("b",): {"mean": 1.05, "variance": 0.08},
        ("c",): {"mean": 0.95, "variance": 0.09},
        ("a", "b"): {"mean": 1.22, "variance": 0.05},
        ("a", "b", "c"): {"mean": 1.38, "variance": 0.03},
    }

    def fake_stats(member_ids: list[str], _tasks) -> dict[str, object]:
        key = tuple(sorted(member_ids))
        base = stats_map.get(key, {"mean": 0.0, "variance": 1.0})
        return {
            "mean": float(base["mean"]),
            "variance": float(base["variance"]),
            "series": [float(base["mean"])] * 3,
        }

    loop._team_holdout_stats = fake_stats  # type: ignore[assignment]
    loop.generation_index = 10
    loop.colonies["col_alpha"] = loop._create_colony_meta("col_alpha", ["a", "b"], pot=1.0)
    loop.colonies["col_alpha"].update(
        {
            "reserve_ratio": 0.25,
            "review_interval": 1,
            "required_passes": 1,
            "holdout_passes": 1,
            "holdout_failures": 0,
            "expand_history": [],
            "last_review": loop.generation_index - 1,
        }
    )

    loop._tick_colonies()

    meta = loop.colonies["col_alpha"]
    assert set(meta["members"]) == {"a", "b", "c"}
    assert meta.get("expand_history") == []
    assert meta.get("last_delta") > 0.0


def test_colony_reserve_guard_triggers():
    loop = _make_loop({"a": 0.2, "b": 0.2})
    loop.config.assimilation_tuning.colony_reserve_ticket_multiplier = 3.0
    loop.config.assimilation_tuning.colony_reserve_ratio = 0.5
    loop.config.assimilation_tuning.colony_reserve_cost_window = 2
    for oid in ("a", "b"):
        loop.population.energy[oid] = [0.6, 0.55]
        loop.population.roi[oid] = [1.0, 0.9]
    loop.colonies["col_reserve"] = loop._create_colony_meta("col_reserve", ["a", "b"], pot=0.1)
    loop.colonies["col_reserve"].update(
        {
            "reserve_ratio": 0.25,
            "review_interval": 10,
            "required_passes": 1,
            "holdout_passes": 0,
            "holdout_failures": 0,
        }
    )
    loop._sample_holdout_tasks = lambda: []  # avoid evaluation
    loop.generation_index = 12

    loop._tick_colonies()

    meta = loop.colonies["col_reserve"]
    assert meta["reserve_active"] is True
    assert meta["freeze_reproduction"] is True
    assert any(ev["type"] == "reserve_enter" for ev in meta.get("events", []))


def test_colony_winter_mode_triggers():
    loop = _make_loop({"a": 1.0, "b": 1.0})
    loop.config.assimilation_tuning.colony_winter_z_kappa = 0.2
    loop.config.assimilation_tuning.colony_winter_window = 4
    for oid in ("a", "b"):
        loop.population.roi[oid] = [1.0, 0.8, 0.3, -0.4]
        loop.population.energy[oid] = [0.6, 0.7, 0.5]
    loop.colonies["col_winter"] = loop._create_colony_meta("col_winter", ["a", "b"], pot=3.0)
    loop.colonies["col_winter"].update(
        {
            "reserve_ratio": 0.25,
            "review_interval": 10,
            "required_passes": 1,
            "holdout_passes": 0,
            "holdout_failures": 0,
        }
    )
    loop._sample_holdout_tasks = lambda: []  # avoid evaluation
    loop.generation_index = 9

    loop._tick_colonies()

    meta = loop.colonies["col_winter"]
    assert meta["winter_mode"] is True
    assert any(ev["type"] == "winter_enter" for ev in meta.get("events", []))


def test_team_probe_sustain_records_candidates():
    loop = _make_loop({"a": 2.0, "b": 1.5})
    loop.config.assimilation_tuning.team_probe_synergy_delta = 0.05
    loop.config.assimilation_tuning.team_probe_variance_nu = 0.1
    loop.config.assimilation_tuning.team_probe_sustain = 2
    for oid in ("a", "b"):
        loop.population.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2))
    loop.population.roi["a"] = [1.0, 1.1, 1.2]
    loop.population.roi["b"] = [0.8, 0.9, 1.0]
    loop._sample_holdout_tasks = lambda: ["task"]

    def fake_stats(member_ids, _tasks):
        if len(member_ids) == 2:
            return {"mean": 2.1, "variance": 0.05, "series": [2.1, 2.1]}
        if member_ids == ["a"]:
            return {"mean": 0.9, "variance": 0.08, "series": [0.9, 0.9]}
        if member_ids == ["b"]:
            return {"mean": 0.7, "variance": 0.07, "series": [0.7, 0.7]}
        return {"mean": 0.0, "variance": 1.0, "series": []}

    loop._team_holdout_stats = fake_stats  # type: ignore[assignment]

    loop._sample_team_synergy()
    assert not loop._team_probe_candidates_gen

    loop._sample_team_synergy()
    assert loop._team_probe_candidates_gen


def test_colony_tax_enriches_pot():
    loop = _make_loop({"a": 5.0})
    loop.config.assimilation_tuning.colony_tax_rate = 0.25
    loop.population.register(Genome(organelle_id="a", drive_weights={}, gate_bias=0.0, rank=2))
    settlement = {"delta": 1.6, "energy_before": 5.0, "energy_after": 5.0}
    meta = loop._create_colony_meta("col_tax", ["a"], pot=0.5)
    loop.colonies["col_tax"] = meta
    loop._apply_colony_tax("a", settlement)
    assert meta["pot"] > 0.5
    assert settlement["energy_after"] < 5.0


def test_colony_selection_pools_and_replication():
    loop = _make_loop({"a": 3.0, "b": 3.0, "c": 3.0, "d": 3.0})
    tune = loop.config.assimilation_tuning
    tune.colony_selection_enabled = True
    tune.colony_selection_interval = 1
    tune.colony_selection_margin = 0.0
    tune.colony_selection_reward_frac = 0.5
    tune.colony_selection_min_pool = 0.5
    tune.colony_min_size = 2
    loop._colony_selection_pool = {"members": [], "pot": 0.0, "events": []}

    for oid, roi in (("a", 2.0), ("b", 1.8), ("c", 0.6), ("d", 0.5)):
        genome = Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2)
        loop.population.register(genome)
        loop.population.roi[oid] = [roi, roi]
        loop.population.energy[oid] = [1.0, 1.1]

    loop.generation_index = 5
    loop.colonies["col_best"] = loop._create_colony_meta("col_best", ["a", "b"], pot=4.0)
    loop.colonies["col_best"].update({"fitness": 1.5, "reserve_floor": 1.0})
    loop.colonies["col_worst"] = loop._create_colony_meta("col_worst", ["c", "d"], pot=2.0)
    loop.colonies["col_worst"].update({"fitness": 0.2, "reserve_floor": 1.0})

    loop._colony_selection_step()

    assert "col_worst" not in loop.colonies
    new_id = f"col_best_c{loop.generation_index}"
    assert new_id in loop.colonies
    assert set(loop.colonies[new_id]["members"]) == {"c", "d"}
    # Parent colony should receive reward and retain expected pot after share
    assert loop.colonies["col_best"]["pot"] == pytest.approx(3.75, rel=1e-6)
    # Child colony inherits pool funds + parental share
    assert loop.colonies[new_id]["pot"] == pytest.approx(1.75, rel=1e-6)
    # Selection stats updated
    assert loop._colony_selection_stats.get("dissolved") == 1
    assert loop._colony_selection_stats.get("replicated") == 1
    # Pool members consumed
    assert not loop._colony_selection_pool.get("members")
    assert loop._colony_selection_pool.get("pot") == pytest.approx(0.5, rel=1e-6)


def test_colony_selection_replication():
    loop = _make_loop({"a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0})
    cfg = loop.config.assimilation_tuning
    cfg.colony_selection_enabled = True
    cfg.colony_selection_interval = 1
    cfg.colony_selection_margin = 0.01
    loop.generation_index = 7
    for oid in ("a", "b", "c", "d"):
        loop.population.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=2))
        loop.population.roi[oid] = [1.0, 1.1, 1.2]
    best_meta = loop._create_colony_meta("col_best", ["a", "b"], pot=1.0)
    best_meta["fitness"] = 1.5
    best_meta["roi_mean"] = 1.2
    loop.colonies["col_best"] = best_meta
    worst_meta = loop._create_colony_meta("col_worst", ["c", "d"], pot=0.4)
    worst_meta["fitness"] = 0.2
    worst_meta["roi_mean"] = 0.3
    loop.colonies["col_worst"] = worst_meta
    loop._colony_selection_step()
    assert "col_worst" not in loop.colonies
    offspring = [cid for cid in loop.colonies if cid.startswith("col_best_c")]
    assert offspring
    assert any(ev["type"] == "dissolve" for ev in loop._colony_events_archive)


def test_colony_tier_migration_promote_and_demote():
    loop = _make_loop({"a": 2.0, "b": 2.0})
    tune = loop.config.assimilation_tuning
    tune.colony_tier_count = 3
    tune.colony_tier_promote_passes = 1
    tune.colony_tier_promote_delta = 0.05
    tune.colony_tier_demote_failures = 1
    tune.colony_tier_demote_delta = -0.01
    tune.colony_tier_hazard_floor = -0.5
    loop.colonies["col_test"] = loop._create_colony_meta("col_test", ["a", "b"], pot=1.0)
    meta = loop.colonies["col_test"]
    loop.generation_index = 3
    meta["holdout_passes"] = 1
    meta["last_delta"] = 0.2
    meta["tier_cooldown"] = 0
    loop._colony_tier_migration()
    assert meta["tier"] == 1
    assert any(ev["type"] == "tier_promote" for ev in meta.get("events", []))
    # Demote on failures
    meta["holdout_failures"] = 1
    meta["last_delta"] = -0.2
    meta["tier_cooldown"] = 0
    loop.generation_index += 1
    loop._colony_tier_migration()
    assert meta["tier"] == 0
    assert any(ev["type"] == "tier_demote" for ev in meta.get("events", []))


def test_apply_lora_soup_merge_uses_block_roles():
    loop = _make_loop({"a": 2.0, "b": 2.0})
    loop.config.assimilation_tuning.team_block_diagonal_merges = True
    loop.config.assimilation_tuning.team_block_rank_cap = 16
    loop.config.host.max_lora_rank = 16
    genome_a = Genome(organelle_id="a", drive_weights={}, gate_bias=0.0, rank=2)
    genome_b = Genome(organelle_id="b", drive_weights={}, gate_bias=0.0, rank=2)
    loop.population.register(genome_a)
    loop.population.register(genome_b)
    loop.population.population["a"].rank_noise = {"attn": 0.5}
    loop.population.population["a"].adapter_dropout = {"attn"}
    loop.population.population["a"].duplication_factors = {"mlp": 0.5}
    loop.colonies["col_test"] = loop._create_colony_meta("col_test", ["a", "b"], pot=1.0)
    captured: dict[str, object] = {}

    def fake_merge(soup, rank, *, roles=None, mode=None, mutation_meta=None):  # type: ignore[no-redef]
        captured["soup"] = dict(soup)
        captured["rank"] = rank
        captured["roles"] = roles
        captured["mode"] = mode
        captured["mutation_meta"] = mutation_meta

    loop.host.merge_lora_soup = fake_merge  # type: ignore[assignment]
    loop.host.get_organelle = lambda oid: SimpleNamespace(rank=2)
    stats_map = {"a": {"roi": 1.0, "ema": 1.0}, "b": {"roi": 0.8, "ema": 0.9}}
    summary = loop._apply_lora_soup_merge(("word", "easy"), "a", ["a", "b"], stats_map, [])
    assert captured.get("mode") == "block"
    assert captured.get("roles") == {"a": 0, "b": 1}
    assert captured.get("rank") == 4
    assert any(entry.get("role") == 0 for entry in summary if entry["organelle_id"] == "a")
    mutation_meta = captured.get("mutation_meta") or {}
    assert "a" in mutation_meta and mutation_meta["a"]["dropout"] == ["attn"]
    record_a = next(item for item in summary if item["organelle_id"] == "a")
    assert "rank_noise" in record_a or "duplication" in record_a
