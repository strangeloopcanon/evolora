from symbiont_ecology.evolution.morphogenesis import MorphogenesisController


def test_tweak_gate_bias_direction():
    class G:
        def __init__(self):
            self.organelle_id = "o"
            self.gate_bias = 0.0
            self.rank = 1

    dummy = G()
    MorphogenesisController._tweak_gate_bias(dummy, roi=1.1)
    assert dummy.gate_bias > 0
    MorphogenesisController._tweak_gate_bias(dummy, roi=0.5)
    assert dummy.gate_bias < 0.1  # moved down
