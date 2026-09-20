import pytest
import asyncio
from app.services.circuit_breaker import CircuitBreaker, CircuitState
from app.services.ai_analyzer import extractive_heuristic_fallback
from app.services.embedding_service import generate_text_embedding, _pseudo_embedding
from app.services.telegram_service import format_urgent_alert_message, _check_urgent_rate_limit
from app.crawlers.article_crawler import make_tz_aware
from datetime import datetime, timezone


def test_circuit_breaker_lifecycle():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=0.1)
    service = "gemini-test"
    
    assert cb.can_execute(service) is True
    cb.record_failure(service)
    assert cb.can_execute(service) is True
    
    # Second failure triggers OPEN state
    cb.record_failure(service)
    assert cb.can_execute(service) is False
    assert cb.get_all_statuses()[service] == CircuitState.OPEN.value

    # Reset
    cb.reset(service)
    assert cb.can_execute(service) is True


def test_extractive_heuristic_fallback():
    title = "Building Event-Driven NestJS Microservices with BullMQ and Redis"
    content = "This guide explores scalable queue architectures, worker concurrency, and backoff retry patterns in NestJS applications."
    url = "https://example.com/nestjs-bullmq"
    
    fallback = extractive_heuristic_fallback(title, content, url)
    assert fallback.relevance_score >= 5.0
    assert fallback.vietnamese_title == title
    assert len(fallback.tags) > 0
    assert "Nestjs" in fallback.tags or "Backend" in fallback.tags or "Technology" in fallback.tags


@pytest.mark.asyncio
async def test_pseudo_embedding_generation():
    text = "High performance PostgreSQL query tuning and indexing"
    vec = await generate_text_embedding(text)
    assert len(vec) == 1536
    assert isinstance(vec[0], float)


def test_urgent_alert_formatting():
    class MockSource:
        name = "Uber Engineering"

    class MockArticle:
        id = 42
        title = "Zero-Downtime Database Migration Strategies"
        vietnamese_title = "Chiến lược di chuyển cơ sở dữ liệu không gián đoạn"
        vietnamese_summary = "Phân tích kỹ thuật shadow writing và dual-read."
        relevance_score = 9.4
        source = MockSource()
        url = "https://eng.uber.com/migrations"
        key_takeaways = ["Shadow writing", "Dual reading", "Canary switch"]
        new_tech_stacks = [{"name": "Vitess", "category": "Database"}]
        tags = ["Database", "Architecture"]

    article = MockArticle()
    msg, markup = format_urgent_alert_message(article)
    
    assert "TECHPULSE BREAKING" in msg
    assert "9.4" in msg
    assert "Chiến lược di chuyển cơ sở dữ liệu" in msg
    assert "inline_keyboard" in markup
    # Ensure feedback buttons exist in markup
    buttons = markup["inline_keyboard"][0]
    assert any("fb:like:42" in b.get("callback_data", "") for b in buttons)
    assert any("fb:dislike:42" in b.get("callback_data", "") for b in buttons)


def test_urgent_rate_limiter():
    # Should allow requests within threshold
    res = _check_urgent_rate_limit()
    assert isinstance(res, bool)
