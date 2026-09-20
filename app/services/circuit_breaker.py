import time
from typing import Dict
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, fast reject/failover
    HALF_OPEN = "HALF_OPEN"  # Testing canary requests


class CircuitBreaker:
    """
    In-memory Circuit Breaker to protect downstream AI providers
    from cascading timeouts and rate limit (429/500/504) storms.
    """
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 60.0,
        half_open_success_threshold: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_success_threshold = half_open_success_threshold

        self._states: Dict[str, CircuitState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}
        self._last_state_change: Dict[str, float] = {}

    def _get_state(self, service_key: str) -> CircuitState:
        if service_key not in self._states:
            self._states[service_key] = CircuitState.CLOSED
            self._failure_counts[service_key] = 0
            self._success_counts[service_key] = 0
            self._last_state_change[service_key] = time.time()

        current_state = self._states[service_key]
        now = time.time()

        # If OPEN and cooldown period expired -> transition to HALF_OPEN
        if current_state == CircuitState.OPEN:
            if now - self._last_state_change[service_key] >= self.recovery_timeout_sec:
                self._states[service_key] = CircuitState.HALF_OPEN
                self._success_counts[service_key] = 0
                self._last_state_change[service_key] = now
                return CircuitState.HALF_OPEN

        return self._states[service_key]

    def can_execute(self, service_key: str) -> bool:
        state = self._get_state(service_key)
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self, service_key: str):
        state = self._get_state(service_key)
        if state == CircuitState.HALF_OPEN:
            self._success_counts[service_key] = self._success_counts.get(service_key, 0) + 1
            if self._success_counts[service_key] >= self.half_open_success_threshold:
                self._states[service_key] = CircuitState.CLOSED
                self._failure_counts[service_key] = 0
                self._last_state_change[service_key] = time.time()
        elif state == CircuitState.CLOSED:
            self._failure_counts[service_key] = 0

    def record_failure(self, service_key: str):
        self._get_state(service_key)
        self._failure_counts[service_key] = self._failure_counts.get(service_key, 0) + 1
        now = time.time()

        if self._failure_counts[service_key] >= self.failure_threshold:
            self._states[service_key] = CircuitState.OPEN
            self._last_state_change[service_key] = now

    def reset(self, service_key: str = None):
        if service_key:
            self._states[service_key] = CircuitState.CLOSED
            self._failure_counts[service_key] = 0
            self._success_counts[service_key] = 0
            self._last_state_change[service_key] = time.time()
        else:
            self._states.clear()
            self._failure_counts.clear()
            self._success_counts.clear()
            self._last_state_change.clear()

    def get_all_statuses(self) -> Dict[str, str]:
        return {k: self._get_state(k).value for k in self._states.keys()}


# Global singleton instance
ai_circuit_breaker = CircuitBreaker()
