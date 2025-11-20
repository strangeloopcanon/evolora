from types import SimpleNamespace
from symbiont_ecology.environment.loops import EcologyLoop


def test_update_qd_archive_populates_bins() -> None:
    class Dummy:
        pass

    cfg = SimpleNamespace(qd=SimpleNamespace(enabled=True, cost_bins=3))
    loop = Dummy()
    loop.config = cfg
    # population with two organelles and average stats
    loop.population = SimpleNamespace(
        population={"o1": object(), "o2": object()},
        average_energy=lambda oid: 0.5 if oid == "o1" else 1.0,
        average_roi=lambda oid, limit=5: 1.2 if oid == "o1" else 0.8,
        cell_novelty=lambda oid, cell, scale=0.3, floor=0.05: 0.4 if oid == "o1" else 0.2,
    )
    # environment returns best cell score key and ema
    loop.environment = SimpleNamespace(
        best_cell_score=lambda oid: (("math", "short"), 0.6),
    )
    loop._qd_archive = {}
    # Call unbound method with our dummy
    size = EcologyLoop._update_qd_archive(loop)  # type: ignore[arg-type]
    assert size >= 1
