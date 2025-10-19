from symbiont_ecology.evolution.ledger import ATPLedger


def test_ledger_energy_and_atp() -> None:
    ledger = ATPLedger()
    ledger.ensure("orgA", initial=5.0)
    assert ledger.charge("orgA", 2.0) == 2.0
    ledger.credit("orgA", 1.5)
    ledger.ensure_energy("orgA", 3.0)
    assert ledger.consume_energy("orgA", 2.0)
    assert not ledger.consume_energy("orgA", 5.0)
    ledger.credit_energy("orgA", 10.0)
    ledger.configure_energy_cap(6.0)
    ledger.credit_energy("orgA", 10.0)
    snapshot = ledger.snapshot()
    assert snapshot.accounts["orgA"] > 0
    assert snapshot.energy["orgA"] == ledger.energy_cap
