from symbiont_ecology.evolution.morphogenesis import MorphogenesisController
from symbiont_ecology.evolution.population import Genome


def test_tweak_gate_bias_increases_and_decreases():
    g = Genome(organelle_id="org_x", drive_weights={}, gate_bias=0.0, rank=2)
    start = g.gate_bias
    MorphogenesisController._tweak_gate_bias(g, roi=1.2)
    assert g.gate_bias > start
    MorphogenesisController._tweak_gate_bias(g, roi=0.4)
    # should move back down
    assert g.gate_bias < start + 0.05 + 1e-9
