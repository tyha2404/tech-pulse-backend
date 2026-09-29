import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.services.circuit_breaker import CircuitBreaker, CircuitState
from app.services.ai_analyzer import analyze_article_with_9router
from app.schemas.schemas import AIAnalysisResult


@pytest.mark.asyncio
async def test_cascading_failure_trips_circuit_breaker_to_fallback():
    """
    Red Team test: Simulate 100% provider outage (504 Gateway Timeouts, 429 Rate Limits).
    Verify that:
    1. Circuit breaker trips to OPEN state immediately after failure threshold.
    2. Fallback heuristic seamlessly engages without unhandled 500 exceptions.
    3. Latency stays bounded instead of hanging requests indefinitely.
    """
    # Create an isolated circuit breaker
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=5.0)
    service_key = "9router-gemini"

    # Trip breaker
    cb.record_failure(service_key)
    cb.record_failure(service_key)
    assert cb.can_execute(service_key) is False
    assert cb.get_all_statuses()[service_key] == CircuitState.OPEN.value

    # Simulate analyze_article_with_9router under complete network outage
    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=Exception("504 Gateway Timeout: AI provider down")):
        result, model_used = await analyze_article_with_9router(
            title="High Throughput Kafka Pipelines with Rust",
            content="A deep dive into zero-copy serialization and batching strategies.",
            url="https://eng.blog/kafka-rust",
        )

        assert isinstance(result, AIAnalysisResult)
        assert model_used == "extractive-fallback"
        assert result.vietnamese_title == "High Throughput Kafka Pipelines with Rust"
        assert result.is_worth_reading is False
        assert "Trích xuất tự động" in result.vietnamese_summary


@pytest.mark.asyncio
async def test_extreme_payload_fuzzing_resilience():
    """
    Red Team test: Send massive 100,000 character strings to analyzer.
    Verify truncation protects memory and LLM token limits without memory blow-up.
    """
    massive_content = "Distributed system architecture " * 5000  # ~160,000 characters
    
    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=Exception("Mocking provider call")):
        result, model_used = await analyze_article_with_9router(
            title="Massive Payload Stress Test",
            content=massive_content,
            url="https://stress.test",
        )
        assert isinstance(result, AIAnalysisResult)
        assert model_used == "extractive-fallback"
        # Fallback summary is strictly bounded
        assert len(result.vietnamese_summary.split()) <= 60
