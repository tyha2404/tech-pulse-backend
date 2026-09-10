import pytest
from app.crawlers.feed_discoverer import discover_feed_url
from app.crawlers.article_crawler import fetch_rss_feed


@pytest.mark.asyncio
async def test_feed_discovery_vnexpress():
    detected_type, feed_url = await discover_feed_url("https://vnexpress.net/so-hoa")
    assert detected_type in ["rss", "scraper"]
    assert feed_url is not None


@pytest.mark.asyncio
async def test_rss_fetch():
    # Test with standard vnexpress so-hoa rss
    items = await fetch_rss_feed("https://vnexpress.net/rss/so-hoa.rss")
    assert len(items) > 0
    assert "title" in items[0]
    assert "url" in items[0]
