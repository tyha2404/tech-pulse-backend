import pytest
from unittest.mock import AsyncMock, patch
from app.services.telegram_service import (
    format_flash_alert_message,
    send_instant_flash_alert,
)
from app.services.crawl_service import trigger_smart_article_notifications
from app.models.models import Article, Source


def test_format_flash_alert_message():
    title = "Zero-Day Vulnerability Found in Linux Kernel eBPF Subsystem"
    url = "https://security.org/cve-2026-9999"
    source = "SecurityWeek"
    snippet = "A critical CVSS 9.8 flaw allows arbitrary code execution in root space via eBPF."

    msg, markup = format_flash_alert_message(
        title=title,
        url=url,
        source_name=source,
        snippet=snippet,
        depth_score=9.5,
    )

    assert "FLASH BREAKING" in msg
    assert "9.5/10" in msg
    assert "Zero-Day Vulnerability" in msg
    assert "SecurityWeek" in msg
    assert "inline_keyboard" in markup
    assert markup["inline_keyboard"][0][0]["url"] == url


@pytest.mark.asyncio
async def test_send_instant_flash_alert_mock():
    with patch("app.services.telegram_service.send_telegram_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        success = await send_instant_flash_alert(
            title="OpenAI Releases GPT-5 Reasoning Engine",
            url="https://openai.com/gpt-5",
            source_name="OpenAI Blog",
            snippet="GPT-5 sets new state-of-the-art across all engineering benchmarks.",
            depth_score=9.8,
        )
        assert success is True
        assert mock_send.called


@pytest.mark.asyncio
async def test_skip_duplicate_urgent_alert_if_flashed():
    article = Article(
        id=999,
        title="Major Breakthrough",
        vietnamese_title="Đột phá công nghệ lớn",
        relevance_score=9.5,
        url="https://example.com/breaking",
    )

    with patch("app.services.crawl_service.send_urgent_alert", new_callable=AsyncMock) as mock_urgent:
        # If skip_urgent_if_flashed=True, send_urgent_alert should NOT be called
        await trigger_smart_article_notifications(article, skip_urgent_if_flashed=True)
        assert not mock_urgent.called

        # If skip_urgent_if_flashed=False, send_urgent_alert SHOULD be called
        await trigger_smart_article_notifications(article, skip_urgent_if_flashed=False)
        assert mock_urgent.called

