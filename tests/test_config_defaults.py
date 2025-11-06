from symbiont_ecology.config import EcologyConfig


def test_assimilation_tuning_new_team_fields_present():
    cfg = EcologyConfig()
    at = cfg.assimilation_tuning
    assert hasattr(at, "team_probe_per_gen") and isinstance(at.team_probe_per_gen, int)
    assert hasattr(at, "team_min_tasks") and isinstance(at.team_min_tasks, int)
    assert hasattr(at, "team_routing_probe_per_gen") and isinstance(at.team_routing_probe_per_gen, int)
    assert hasattr(at, "team_block_diagonal_merges") and isinstance(at.team_block_diagonal_merges, bool)
    assert hasattr(at, "team_block_rank_cap") and isinstance(at.team_block_rank_cap, int)
    assert hasattr(at, "assimilation_history_limit")
    assert hasattr(at, "assimilation_history_summary")


def test_evolution_config_mutation_fields_present():
    cfg = EcologyConfig()
    evo = cfg.evolution
    assert hasattr(evo, "mutation_rank_noise_prob")
    assert hasattr(evo, "mutation_dropout_prob")
    assert hasattr(evo, "mutation_duplication_prob")
    assert hasattr(evo, "mutation_layer_tags")


def test_knowledge_config_defaults_present():
    cfg = EcologyConfig()
    knowledge = cfg.knowledge
    assert hasattr(knowledge, "enabled")
    assert hasattr(knowledge, "write_cost")
    assert hasattr(knowledge, "read_cost")
    assert hasattr(knowledge, "ttl")
    assert hasattr(knowledge, "max_items")
