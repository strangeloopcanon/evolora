from symbiont_ecology import (
    ATPLedger,
    AssimilationTester,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    HumanBandit,
    PopulationManager,
)
from symbiont_ecology.environment.grid import GridConfig, GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def test_assimilation_uses_global_probe_and_soup(tmp_path) -> None:
    config = EcologyConfig()
    config.grid = GridConfig(families=["math"], depths=["short"])
    config.evolution.assimilation_threshold = 0.01
    config.evolution.assimilation_p_value = 0.2
    config.assimilation_tuning.per_cell_interval = 1
    config.assimilation_tuning.max_merges_per_cell = 2
    config.assimilation_tuning.soup_size = 3
    config.assimilation_tuning.hf_prompts = []
    config.metrics.root = tmp_path

    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    population = PopulationManager(config.evolution)
    candidate_id = host.spawn_organelle(rank=2)
    mate_id = host.spawn_organelle(rank=2)
    population.register(Genome(organelle_id=candidate_id, drive_weights={}, gate_bias=0.0, rank=2))
    population.register(Genome(organelle_id=mate_id, drive_weights={}, gate_bias=0.0, rank=2))

    assimilation = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )

    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=5,
    )

    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=assimilation,
        human_bandit=None,
        sink=None,
    )

    cell = ("math", "short")
    loop.environment.organism_stats[candidate_id] = {cell: 0.8}
    loop.environment.organism_stats[mate_id] = {cell: 0.7}
    loop.environment.organism_canary_fail = {}

    population.history[candidate_id] = [0.1, 0.2, 0.25, 0.35]
    population.energy[candidate_id] = [0.1, 0.1, 0.1, 0.1]
    population.roi[candidate_id] = [1.4, 1.5]
    population.roi[mate_id] = [1.1, 1.2]

    merges = loop._attempt_assimilation(capped=1)
    assert merges == 1
    assert candidate_id not in population.population
    assert len(population.population) == 2
    history = population.assimilation_records(candidate_id, cell)
    assert history, "assimilation history should be recorded"
    assert history[-1]["passed"] is True


def test_assimilation_records_multi_soup_and_hf(tmp_path) -> None:
    config = EcologyConfig()
    config.grid = GridConfig(families=["math"], depths=["short"])
    config.evolution.assimilation_threshold = 0.01
    config.evolution.assimilation_p_value = 0.4
    config.assimilation_tuning.per_cell_interval = 1
    config.assimilation_tuning.max_merges_per_cell = 3
    config.assimilation_tuning.soup_size = 3
    config.assimilation_tuning.hf_prompts = ["Describe your approach."]
    config.metrics.root = tmp_path

    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    population = PopulationManager(config.evolution)
    candidate_id = host.spawn_organelle(rank=2)
    mate_a = host.spawn_organelle(rank=2)
    mate_b = host.spawn_organelle(rank=2)
    population.register(Genome(organelle_id=candidate_id, drive_weights={}, gate_bias=0.0, rank=2))
    population.register(Genome(organelle_id=mate_a, drive_weights={}, gate_bias=0.0, rank=2))
    population.register(Genome(organelle_id=mate_b, drive_weights={}, gate_bias=0.0, rank=2))

    assimilation = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )

    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=11,
    )

    class StubSink:
        def __init__(self) -> None:
            self.events: list[tuple] = []

        def log_episode(self, *_args, **_kwargs) -> None:
            return

        def log_assimilation(self, event, decision) -> None:
            self.events.append((event, decision))

    sink = StubSink()

    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=assimilation,
        human_bandit=HumanBandit(),
        sink=sink,  # type: ignore[arg-type]
    )

    cell = ("math", "short")
    loop.environment.organism_stats[candidate_id] = {cell: 0.85}
    loop.environment.organism_stats[mate_a] = {cell: 0.8}
    loop.environment.organism_stats[mate_b] = {cell: 0.78}
    loop.environment.organism_canary_fail = {}

    population.history[candidate_id] = [0.1, 0.2, 0.3, 0.45]
    population.energy[candidate_id] = [0.05, 0.05, 0.05, 0.05]
    population.roi[candidate_id] = [1.6, 1.7]
    population.roi[mate_a] = [1.4, 1.5]
    population.roi[mate_b] = [1.3, 1.45]

    merges = loop._attempt_assimilation(capped=1)
    assert merges == 1
    assert sink.events, "expected assimilation event to be logged"
    event, decision = sink.events[-1]
    assert decision is True
    assert event.soup, "soup summary should be populated"
    member_ids = {entry["organelle_id"] for entry in event.soup}
    assert candidate_id in member_ids
    assert len(member_ids) >= 2
    assert event.probes, "human-feedback probes should be recorded"
    assert isinstance(event.method, str)
    assert isinstance(event.dr_used, bool)


def test_population_tracks_assimilation_history() -> None:
    config = EcologyConfig()
    population = PopulationManager(config.evolution)
    genome = Genome(organelle_id="org-test", drive_weights={}, gate_bias=0.0, rank=2)
    population.register(genome)
    record = {"generation": 1, "passed": True, "uplift": 0.2}
    population.record_assimilation(genome.organelle_id, ("math", "short"), record)
    history = population.assimilation_records(genome.organelle_id, ("math", "short"))
    assert history
    assert history[-1]["generation"] == 1


def test_assimilation_dr_alignment() -> None:
    tester = AssimilationTester(0.0, 0.5, 0)
    tester.dr_enabled = True
    tester.dr_strata = ["family"]
    tester.dr_min_stratum = 1
    control = [0.1, 0.2, 0.4]
    treatment = [0.2, 0.3, 0.5]
    control_meta = [{"family": "math"}, {"family": "logic"}, {"family": "math"}]
    treatment_meta = [{"family": "math"}, {"family": "logic"}, {"family": "math"}]
    result = tester.evaluate(
        "org",
        control,
        treatment,
        safety_hits=0,
        energy_cost=0.1,
        energy_balance=1.0,
        energy_top_up=None,
        control_meta=control_meta,
        treatment_meta=treatment_meta,
    )
    assert result.event.dr_used is True
    assert "dr" in result.event.method
    assert result.event.sample_size == 3
    assert result.event.dr_sample_sizes
