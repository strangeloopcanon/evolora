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
