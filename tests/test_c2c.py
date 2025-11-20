from types import SimpleNamespace

from symbiont_ecology import ATPLedger
from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.routing.router import BanditRouter


def test_cache_bus_post_and_read():
    cfg = EcologyConfig()
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary)
    # Post one latent vector and read it back
    ok = env.post_cache("org_A", [0.1, 0.2, 0.3], ttl=2)
    assert ok is True
    items = env.read_caches(max_items=1)
    assert items and items[0]["organelle_id"] == "org_A"
    assert items[0]["latent"] == [0.1, 0.2, 0.3]
    # TTL decrements; another read should still produce until TTL exhausts
    items2 = env.read_caches(max_items=1)
    assert items2 and items2[0]["organelle_id"] == "org_A"


def test_host_step_latent_prefix_blend():
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    # Build a latent prefix matching backbone hidden size
    vec = host.backbone.encode_text(["seed"], device=host.device).squeeze(0).tolist()
    res = host.step(
        prompt="p",
        intent="test",
        max_routes=0,
        allowed_organelle_ids=[],
        latent_prefix=vec,
        latent_mix=1.0,
    )
    latent = res.envelope.observation.state.get("latent")
    assert isinstance(latent, list) and len(latent) == len(vec)
    # With mix=1.0, latent should equal the prefix
    assert latent == vec


def test_colony_c2c_debit_helper():
    meta = {"pot": 1.0, "c2c_bandwidth_left": 0.6, "c2c_reads_left": 2}
    ok = EcologyLoop._colony_c2c_debit(meta, 0.3, "c2c_reads_left")
    assert ok is True
    assert meta["pot"] == 0.7
    assert meta["c2c_bandwidth_left"] == 0.3
    assert meta["c2c_reads_left"] == 1
    # Insufficient bandwidth stops the debit
    fail = EcologyLoop._colony_c2c_debit(meta, 0.5, "c2c_reads_left")
    assert fail is False


class _FakeLedger:
    def __init__(self, balance: float = 1.0) -> None:
        self._balance = {"org": balance}

    def energy_balance(self, organelle_id: str) -> float:
        return self._balance.get(organelle_id, 0.0)

    def consume_energy(self, organelle_id: str, amount: float) -> None:
        self._balance[organelle_id] = self.energy_balance(organelle_id) - amount


def test_consume_c2c_latents_and_post():
    loop = object.__new__(EcologyLoop)
    ledger = _FakeLedger(balance=1.0)
    posted: list[list[float]] = []

    def _read_caches(max_items: int = 1):
        return [{"latent": [0.4, 0.5]}]

    loop.host = SimpleNamespace(ledger=ledger)
    loop.environment = SimpleNamespace(
        read_caches=_read_caches,
        post_cache=lambda oid, latent, ttl: posted.append(latent),
    )
    loop._pending_latents = {}
    loop.colonies = {}

    loop._consume_c2c_latents("org", None, 0.2)
    assert loop._pending_latents["org"] == [[0.4, 0.5]]
    assert ledger.energy_balance("org") == 0.8

    # Colony-funded path
    meta = {"pot": 1.0, "c2c_bandwidth_left": 0.5, "c2c_posts_left": 1}
    loop._post_c2c_latent("org", [0.7, 0.8], 0.3, 5, ("col", meta))
    assert posted == [[0.7, 0.8]]
    assert meta["pot"] == 0.7
    assert meta["c2c_posts_left"] == 0
    assert meta["c2c_bandwidth_left"] == 0.2


def test_log_colony_event_helper():
    meta: dict[str, object] = {}
    dummy = SimpleNamespace(_colony_events_archive=[])
    EcologyLoop._log_colony_event(dummy, meta, 7, "create", members=["org_a"])
    assert dummy._colony_events_archive[0]["type"] == "create"
    events = meta.get("events")
    assert isinstance(events, list)
    assert events[0]["gen"] == 7
    assert events[0]["type"] == "create"
