import pytest
from unittest.mock import AsyncMock, patch
from app.services.triage_service import fast_triage_article
from app.schemas.schemas import FastTriageResult


@pytest.mark.asyncio
async def test_fast_triage_tech_article_escalates_to_full_ai():
    """High value tech articles should be routed to full AI"""
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_choice = AsyncMock()
        mock_choice.message.content = '{"is_relevant_tech": true, "is_spam_or_marketing": false, "confidence": 0.92, "suggested_priority": "PROCESS_FULL_AI", "is_breaking_news": false, "urgency_level": "NORMAL", "tech_depth_score": 8.5, "reason": "Deep technical breakdown"}'
        mock_resp = AsyncMock()
        mock_resp.choices = [mock_choice]
        mock_create.return_value = mock_resp

        title = "DeepSeek Releases v3 Architecture with Multi-Head Latent Attention"
        snippet = "A deep technical breakdown of MLA and FP8 mixed precision training on distributed clusters."
        url = "https://deepseek.ai/blog/v3-architecture"

        res, model_used = await fast_triage_article(title=title, snippet=snippet, url=url)
        assert isinstance(res, FastTriageResult)
        assert res.is_relevant_tech is True
        assert res.suggested_priority in ["PROCESS_FULL_AI", "STORE_UNANALYZED"]
        assert model_used in ["typesafe-jev", "fast-llm-classifier"]


@pytest.mark.asyncio
async def test_fast_triage_spam_marketing_discarded():
    """Consumer gadget / coupon noise should be suggested to DISCARD"""
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_choice = AsyncMock()
        mock_choice.message.content = '{"is_relevant_tech": false, "is_spam_or_marketing": true, "confidence": 0.95, "suggested_priority": "DISCARD", "reason": "Consumer deals and gadget accessories"}'
        mock_resp = AsyncMock()
        mock_resp.choices = [mock_choice]
        mock_create.return_value = mock_resp

        title = "Best Black Friday Deals on iPhone 15 Cases and Screen Protectors"
        snippet = "Grab 50% discount codes on silicone covers for iPhone and Samsung accessories."
        url = "https://deals-affiliate.com/iphone-cases"

        res, model_used = await fast_triage_article(title=title, snippet=snippet, url=url)
        assert isinstance(res, FastTriageResult)
        assert res.is_spam_or_marketing is True
        assert res.is_relevant_tech is False
        assert res.suggested_priority == "DISCARD"



@pytest.mark.asyncio
async def test_fast_triage_typesafe_jev_mock():
    """Mock test specifically verifying TypeSafe Jev System One endpoint contract with primitives"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json = lambda: {
            "model": "jev-1.13.0",
            "answers": {
                "is_relevant_tech": {
                    "type": "noul",
                    "probability": 0.98
                },
                "is_spam_or_marketing": {
                    "type": "noul",
                    "probability": 0.02
                },
                "suggested_priority": {
                    "type": "choice",
                    "choice": "PROCESS_FULL_AI",
                    "confidence": 0.95
                },
                "tech_depth_score": {
                    "type": "score",
                    "score": 8.8,
                    "confidence": 0.92
                }
            },
            "usage": {"input_tokens": 150, "output_tokens": 30}
        }
        with patch("app.core.config.settings.TYPESAFE_API_KEY", "test-jev-key"):
            res, model_used = await fast_triage_article(
                title="PostgreSQL 17 Logical Replication Performance",
                snippet="Benchmarking streaming replication in high-throughput clusters.",
                url="https://postgres.org/news/17-replication"
            )
            assert model_used == "typesafe-jev"
            assert res.is_relevant_tech is True
            assert res.is_spam_or_marketing is False
            assert res.confidence == 0.95
            assert res.suggested_priority == "PROCESS_FULL_AI"
            assert "Jev System-1 Primitive" in res.reason
