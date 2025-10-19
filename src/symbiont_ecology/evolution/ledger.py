"""ATP credit ledger for organelles."""

from __future__ import annotations

from dataclasses import dataclass, field

from symbiont_ecology.metrics.telemetry import LedgerSnapshot


@dataclass
class ATPAccount:
    balance: float = 0.0

    def deposit(self, amount: float) -> None:
        self.balance += max(amount, 0.0)

    def withdraw(self, amount: float) -> float:
        spend = min(self.balance, max(amount, 0.0))
        self.balance -= spend
        return spend


@dataclass
class ATPLedger:
    accounts: dict[str, ATPAccount] = field(default_factory=dict)
    energy_accounts: dict[str, float] = field(default_factory=dict)
    energy_cap: float = 5.0

    def ensure(self, organelle_id: str, initial: float) -> None:
        if organelle_id not in self.accounts:
            self.accounts[organelle_id] = ATPAccount(initial)

    def charge(self, organelle_id: str, amount: float) -> float:
        self.ensure(organelle_id, 0.0)
        return self.accounts[organelle_id].withdraw(amount)

    def credit(self, organelle_id: str, amount: float) -> None:
        self.ensure(organelle_id, 0.0)
        self.accounts[organelle_id].deposit(amount)

    # Energy management -------------------------------------------------
    def configure_energy_cap(self, cap: float) -> None:
        self.energy_cap = cap

    def ensure_energy(self, organelle_id: str, initial: float) -> None:
        if organelle_id not in self.energy_accounts:
            self.energy_accounts[organelle_id] = max(initial, 0.0)

    def energy_balance(self, organelle_id: str) -> float:
        return self.energy_accounts.get(organelle_id, 0.0)

    def consume_energy(self, organelle_id: str, amount: float) -> bool:
        self.ensure_energy(organelle_id, 0.0)
        if self.energy_accounts[organelle_id] < amount:
            return False
        self.energy_accounts[organelle_id] -= amount
        return True

    def credit_energy(self, organelle_id: str, amount: float) -> float:
        self.ensure_energy(organelle_id, 0.0)
        self.energy_accounts[organelle_id] = min(
            self.energy_cap, self.energy_accounts[organelle_id] + amount
        )
        return self.energy_accounts[organelle_id]

    def set_energy(self, organelle_id: str, value: float) -> None:
        self.energy_accounts[organelle_id] = max(0.0, min(self.energy_cap, value))

    def snapshot(self) -> LedgerSnapshot:
        balances = {key: acc.balance for key, acc in self.accounts.items()}
        energy = dict(self.energy_accounts)
        return LedgerSnapshot(
            accounts=balances,
            total_atp=sum(balances.values()),
            energy=energy,
        )


__all__ = ["ATPAccount", "ATPLedger"]
