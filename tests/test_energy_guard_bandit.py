from symbiont_ecology.environment.loops import EnergyGuardBandit


def test_energy_guard_bandit_basic_flow():
    bandit = EnergyGuardBandit([0.1, 0.2], seed=123)
    # First selections should try each arm once
    idx0 = bandit.select()
    assert idx0 in (0, 1)
    bandit.record_reward(1.0)
    assert bandit.arms[idx0].pulls == 1
    assert bandit.arms[idx0].reward == 1.0
    # Next selection should consider UCB; just ensure it returns a valid arm
    idx1 = bandit.select()
    assert idx1 in (0, 1)

