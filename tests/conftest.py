import pytest
from app.services.circuit_breaker import ai_circuit_breaker
from app.services.ai_analyzer import _ai_clients


@pytest.fixture(autouse=True)
def reset_circuit_breaker_isolation():
    """
    Autouse fixture to prevent Circuit Breaker State Poisoning across tests.
    Resets the global singleton ai_circuit_breaker before and after each test case.
    """
    ai_circuit_breaker.reset()
    yield
    ai_circuit_breaker.reset()


@pytest.fixture(autouse=True)
def reset_ai_client_cache():
    """Clear cached AI clients so tests patching ai_analyzer.AsyncOpenAI get a fresh mock."""
    _ai_clients.clear()
    yield
    _ai_clients.clear()


@pytest.fixture(autouse=True)
def reset_urgent_alert_rate_limit():
    """Clear the in-memory urgent-alert rate limiter so earlier tests don't exhaust the daily cap."""
    from app.services.telegram_service import _urgent_alert_timestamps

    _urgent_alert_timestamps.clear()
    yield
    _urgent_alert_timestamps.clear()


@pytest.fixture(autouse=True)
def use_9router_model_names_in_tests(monkeypatch):
    """Production defaults to agy/... models; tests mock AsyncOpenAI and must never spawn the real CLI."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "AI_MODEL", "test-model")
    monkeypatch.setattr(settings, "AI_FALLBACK_MODELS", "test-fallback-1,test-fallback-2")
