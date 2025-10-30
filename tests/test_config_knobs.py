from symbiont_ecology.config import EcologyConfig


def test_metrics_in_memory_log_limit_default() -> None:
    cfg = EcologyConfig()
    assert isinstance(cfg.metrics.in_memory_log_limit, int)
    assert cfg.metrics.in_memory_log_limit == 256


def test_host_gen_max_new_tokens_default() -> None:
    cfg = EcologyConfig()
    assert isinstance(cfg.host.gen_max_new_tokens, int)
    assert 1 <= cfg.host.gen_max_new_tokens <= 512
