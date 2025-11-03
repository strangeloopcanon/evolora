from symbiont_ecology.evolution.ledger import ATPLedger


def test_atp_and_energy_flows():
    ledger = ATPLedger()
    oid = "org_test"
    ledger.ensure(oid, 1.0)
    ledger.credit(oid, 2.0)
    spent = ledger.charge(oid, 2.5)
    assert 0.0 <= spent <= 3.0
    ledger.configure_energy_cap(5.0)
    ledger.ensure_energy(oid, 1.0)
    assert ledger.consume_energy(oid, 0.5) is True
    after = ledger.credit_energy(oid, 10.0)
    assert after == 5.0
    snap = ledger.snapshot()
    assert isinstance(snap.accounts, dict) and isinstance(snap.energy, dict)

