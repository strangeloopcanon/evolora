from unittest.mock import patch

from symbiont_ecology.config import EvolutionConfig
from symbiont_ecology.evolution.population import Genome, PopulationManager


def test_population_register_and_metrics_and_selection():
    pm = PopulationManager(EvolutionConfig())
    g1 = Genome("o1", {"novelty": 0.1}, gate_bias=0.0, rank=4)
    g2 = Genome("o2", {"novelty": 0.2}, gate_bias=0.0, rank=4)
    g3 = Genome("o3", {"novelty": 0.3}, gate_bias=0.0, rank=4)
    for g in (g1, g2, g3):
        pm.register(g)
    # record some history
    pm.record_score("o1", 0.5)
    pm.record_score("o1", 0.6)
    pm.record_score("o2", 0.2)
    pm.record_score("o3", 0.9)
    pm.record_energy("o1", 1.0)
    pm.record_energy("o2", 2.0)
    pm.record_roi("o1", 0.5)
    pm.record_roi("o2", 0.1)
    pm.record_roi("o3", 0.9)
    records = pm.recent_score_records("o1", limit=2)
    assert records and all("score" in rec for rec in records)
    pm.record_score("o1", 0.8, meta={"family": "math"})
    last = pm.recent_score_records("o1", limit=1)[0]
    assert last.get("family") == "math"
    assert pm.average_score("o1") > 0
    assert pm.aggregate_roi() > 0
    assert pm.aggregate_energy() > 0
    # selection ranking should favor viable and higher ROI/score
    viability = {"o1": True, "o2": False, "o3": True}
    ranked = pm.rank_for_selection(viability)
    assert ranked[0].organelle_id in {"o1", "o3"}


def test_population_prune_excess_and_assimilation_records():
    cfg = EvolutionConfig(max_population=2)
    pm = PopulationManager(cfg)
    for i in range(3):
        oid = f"o{i}"
        pm.register(Genome(oid, {"novelty": 0.0}, gate_bias=0.0, rank=2))
        pm.record_score(oid, float(i) * 0.1)
    removed = pm.prune_excess()
    # Should propose removing exactly one org
    assert len(removed) == 1
    # Assimilation record bookkeeping
    pm.record_assimilation("o0", ("word.count", "short"), {"uplift": 0.1})
    recs = pm.assimilation_records("o0", ("word.count", "short"))
    assert recs and recs[-1]["uplift"] == 0.1


def test_cell_novelty_decreases_with_usage():
    cfg = EvolutionConfig()
    pm = PopulationManager(cfg)
    pm.register(Genome("o1", {}, gate_bias=0.0, rank=2))
    pm.register(Genome("o2", {}, gate_bias=0.0, rank=2))
    cell = ("word.count", "short")
    nov_initial = pm.cell_novelty("o1", cell)
    pm.update_cell_value("o1", cell, roi=0.6, decay=0.3)
    pm.update_cell_value("o2", cell, roi=0.7, decay=0.3)
    nov_after = pm.cell_novelty("o1", cell)
    assert nov_after < nov_initial
    assert pm.global_cell_counts[cell] >= 2


def test_mutation_injects_rank_noise_dropout_and_duplication():
    cfg = EvolutionConfig(
        mutation_rank_noise_prob=1.0,
        mutation_rank_noise_scale=0.5,
        mutation_dropout_prob=1.0,
        mutation_dropout_decay=0.0,
        mutation_duplication_prob=1.0,
        mutation_duplication_scale=0.5,
        mutation_layer_tags=["attn"],
    )
    pm = PopulationManager(cfg)
    genome = Genome("parent", {}, gate_bias=0.0, rank=2)
    random_random = [0.0] * 8  # ensure all probability checks pass
    gauss_values = [
        0.0,  # word_count_focus
        0.0,  # logic_focus
        0.05,  # beta_exploit
        -0.02,  # q_decay
        0.03,  # ucb_bonus
        -0.04,  # budget_aggressiveness
        0.01,  # post_rate
        -0.02,  # read_rate
        0.0,  # hint_weight
        0.6,  # rank noise delta
        0.5,  # duplication delta
        0.0,  # gate_bias jitter
    ]
    with patch("symbiont_ecology.evolution.population.random.random", side_effect=random_random), patch(
        "symbiont_ecology.evolution.population.random.choice", side_effect=[0, "attn", "attn", "attn"]
    ), patch("symbiont_ecology.evolution.population.random.gauss", side_effect=gauss_values):
        mutant = pm.mutate(genome)
    assert mutant.rank_noise.get("attn") is not None
    assert "attn" in mutant.adapter_dropout
    assert mutant.duplication_factors.get("attn", 0.0) > 0.0
