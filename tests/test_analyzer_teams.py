import importlib.util
from pathlib import Path


def _load_analyzer():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "analyze_ecology_run.py"
    spec = importlib.util.spec_from_file_location("analyze_ecology_run", mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_summarise_generations_team_and_corouting():
    records = [
        {
            "generation": 1,
            "avg_roi": 1.0,
            "avg_total": 0.1,
            "avg_energy_cost": 0.5,
            "active": 2,
            "bankrupt": 0,
            "merges": 0,
            "culled_bankrupt": 0,
            "mean_energy_balance": 1.0,
            "lp_mix_base": 0.2,
            "lp_mix_active": 0.2,
            "team_routes": 1,
            "team_promotions": 0,
            "co_routing_top": {"A:B": 2},
        },
        {
            "generation": 2,
            "avg_roi": 2.0,
            "avg_total": 0.2,
            "avg_energy_cost": 0.6,
            "active": 2,
            "bankrupt": 0,
            "merges": 1,
            "culled_bankrupt": 0,
            "mean_energy_balance": 1.1,
            "lp_mix_base": 0.2,
            "lp_mix_active": 0.3,
            "team_routes": 2,
            "team_promotions": 1,
            "co_routing_top": {"A:B": 1, "B:C": 3},
        },
    ]
    analyzer = _load_analyzer()
    summary = analyzer.summarise_generations(records)
    assert summary["team_routes_total"] == 3
    assert summary["team_promotions_total"] == 1
    # Aggregate co-routing should sum counts across gens
    assert summary["co_routing_totals"]["A:B"] == 3
