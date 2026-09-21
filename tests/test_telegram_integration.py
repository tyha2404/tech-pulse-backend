import pytest
from unittest.mock import AsyncMock, patch
from app.models.models import Article
from app.services.crawl_service import trigger_smart_article_notifications


@pytest.mark.asyncio
async def test_trigger_elite_article_notification():
    article = Article(
        id=99,
        title="Deep Architecture in NestJS",
        vietnamese_title="Kiến trúc chuyên sâu NestJS",
        relevance_score=8.5,
        is_canonical=True,
        url="https://test.com",
    )
    with patch(
        "app.services.crawl_service.send_telegram_message", new_callable=AsyncMock
    ) as mock_send:
        await trigger_smart_article_notifications(article, cluster_sources=[])
        assert mock_send.called
        # Check that HTML text contains elite marker
        args, kwargs = mock_send.call_args
        assert "TECHPULSE ELITE" in args[0]


@pytest.mark.asyncio
async def test_trigger_cluster_breaking_notification():
    article = Article(
        id=100,
        title="Claude 3.7 Release",
        vietnamese_title="Anthropic ra mắt Claude 3.7",
        relevance_score=8.0,  # Meets >= 7.5 threshold
        is_canonical=True,
        url="https://test.com",
    )
    sources = ["Tinh Tế", "Tuổi Trẻ", "Dân Trí"]
    with patch(
        "app.services.crawl_service.send_telegram_message", new_callable=AsyncMock
    ) as mock_send:
        await trigger_smart_article_notifications(article, cluster_sources=sources)
        assert mock_send.called
        args, kwargs = mock_send.call_args
        assert "XU HƯỚNG CÔNG NGHỆ ĐANG NÓNG" in args[0]

