import torch

from symbiont_ecology.config import EvolutionConfig
from symbiont_ecology.evolution.population import Genome, PopulationManager
from symbiont_ecology.utils.torch_utils import clamp_norm, ensure_dtype, no_grad, resolve_device


def test_torch_utils_basic_paths():
    dev = resolve_device("cpu")
    assert isinstance(dev, torch.device)
    dev2 = resolve_device(torch.device("cpu"))
    assert isinstance(dev2, torch.device)
    x = torch.ones(2, 2, dtype=torch.float32)
    y = clamp_norm(x * 100.0, max_norm=1.0)
    assert torch.linalg.norm(y) <= 1.01
    z = ensure_dtype(x, torch.float16)
    assert z.dtype == torch.float16
    # no-op branches
    y2 = clamp_norm(torch.zeros(2, 2), max_norm=1.0)
    assert torch.allclose(y2, torch.zeros(2, 2))
    z2 = ensure_dtype(x, torch.float32)
    assert z2 is not None and z2.dtype == torch.float32
    with no_grad():
        w = x + 1.0
        assert torch.all(w == 2.0)


def test_population_adapter_utilisation_and_remove():
    pm = PopulationManager(EvolutionConfig())
    g = Genome("oX", {"novelty": 0.0}, gate_bias=0.0, rank=2)
    pm.register(g)
    pm.record_adapter_usage("oX", {"q_proj": 1, "total": 1, "rank": 2}, tokens=10)
    util = pm.average_adapter_usage("oX", "q_proj")
    assert util > 0
    mods = pm.module_utilisation()
    assert "q_proj" in mods
    pm.increment_ages()
    pm.remove("oX")
    assert "oX" not in pm.population
