import pytest
from app.services.circuit_breaker import ai_circuit_breaker


@pytest.fixture(autouse=True)
def reset_circuit_breaker_isolation():
    """
    Autouse fixture to prevent Circuit Breaker State Poisoning across tests.
    Resets the global singleton ai_circuit_breaker before and after each test case.
    """
    ai_circuit_breaker.reset()
    yield
    ai_circuit_breaker.reset()
