import time

from app.services.ai_analyzer import _retry_after_seconds, record_ai_failure
from app.services.circuit_breaker import CircuitBreaker, ai_circuit_breaker


class _Err(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


def test_trip_opens_immediately_and_recovers_after_its_cooldown():
    cb = CircuitBreaker(recovery_timeout_sec=60)
    cb.trip("m", cooldown_sec=0.05)
    assert not cb.can_execute("m")
    time.sleep(0.08)
    assert cb.can_execute("m")  # half-open after the custom cooldown, not after 60s


def test_trip_cooldown_is_cleared_by_reset():
    cb = CircuitBreaker(recovery_timeout_sec=60)
    cb.trip("m", cooldown_sec=3600)
    cb.reset("m")
    assert cb.can_execute("m")
    cb.record_failure("m"); cb.record_failure("m"); cb.record_failure("m")
    assert not cb.can_execute("m")  # back to the default threshold and cooldown


def test_missing_credentials_trip_the_model_at_once():
    record_ai_failure("gemini/x", _Err("No active credentials for provider: gemini", 404))
    assert not ai_circuit_breaker.can_execute("gemini/x")


def test_quota_exhaustion_trips_using_the_retry_hint():
    msg = '[groq] [429]: Rate limit reached ... Please try again in 17m40.1s. Need more tokens?'
    record_ai_failure("groq/y", _Err(msg, 503))
    assert not ai_circuit_breaker.can_execute("groq/y")
    assert 1000 < _retry_after_seconds(msg) < 1100


def test_retry_after_defaults_and_caps():
    assert _retry_after_seconds("no hint here") == 300.0
    assert _retry_after_seconds("try again in 9h") == 3600.0
    assert _retry_after_seconds("try again in 2s") == 60.0


def test_plain_server_errors_still_need_three_failures():
    for _ in range(2):
        record_ai_failure("nexo/z", _Err("502 bad gateway", 502))
    assert ai_circuit_breaker.can_execute("nexo/z")
    record_ai_failure("nexo/z", _Err("502 bad gateway", 502))
    assert not ai_circuit_breaker.can_execute("nexo/z")
