import pytest
from app.crawlers.feed_discoverer import discover_feed_url
from app.crawlers.article_crawler import fetch_rss_feed


@pytest.mark.asyncio
async def test_feed_discovery_vnexpress():
    detected_type, feed_url = await discover_feed_url("https://vnexpress.net/so-hoa")
    assert detected_type in ["rss", "scraper", "sitemap"]
    assert feed_url is not None



@pytest.mark.asyncio
async def test_rss_fetch():
    # Test with standard vnexpress so-hoa rss
    items = await fetch_rss_feed("https://vnexpress.net/rss/so-hoa.rss")
    assert len(items) > 0
    assert "title" in items[0]
    assert "url" in items[0]


@pytest.mark.asyncio
async def test_crawl_all_active_sources_semaphore():
    from unittest.mock import AsyncMock, patch, MagicMock
    from app.services.crawl_service import crawl_all_active_sources
    from app.models.models import Source

    # Create dummy sources
    mock_sources = [
        MagicMock(spec=Source, id=1, name="Source 1", is_active=True),
        MagicMock(spec=Source, id=2, name="Source 2", is_active=True),
        MagicMock(spec=Source, id=3, name="Source 3", is_active=True),
    ]

    # Mock AsyncSessionLocal and crawl_single_source
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_sources
    mock_session.execute.return_value = mock_result

    
    async def mock_get(model, obj_id):
        for s in mock_sources:
            if s.id == obj_id:
                return s
        return None
    mock_session.get.side_effect = mock_get

    progress_messages = []
    def on_progress(msg):
        progress_messages.append(msg)

    with patch("app.core.database.AsyncSessionLocal") as mock_db_cls, \
         patch("app.services.crawl_service.crawl_single_source", new_callable=AsyncMock) as mock_crawl_single:
        
        mock_db_cls.return_value.__aenter__.return_value = mock_session
        mock_crawl_single.side_effect = [2, 3, 1]

        total = await crawl_all_active_sources(
            concurrency_limit=2,
            run_ai=False,
            progress_callback=on_progress,
        )

        assert total == 6
        assert mock_crawl_single.call_count == 3
        assert len(progress_messages) == 3


@pytest.mark.asyncio
async def test_crawl_single_source_http_error_graceful():
    from unittest.mock import AsyncMock, patch, MagicMock
    import httpx
    from app.services.crawl_service import crawl_single_source
    from app.models.models import Source

    mock_db = AsyncMock()
    source = Source(id=46, name="24h Tech", url="https://www.24h.com.vn", feed_url="https://www.24h.com.vn/rss.xml", source_type="rss")

    request = httpx.Request("GET", source.feed_url)
    response = httpx.Response(403, request=request)
    http_error = httpx.HTTPStatusError("Client error '403 Forbidden'", request=request, response=response)

    with patch("app.services.crawl_service.fetch_rss_feed", side_effect=http_error):
        # Should NOT raise HTTPStatusError or crash
        articles_added = await crawl_single_source(source, mock_db, run_ai=False)
        assert articles_added == 0
        assert source.status == "error"
        assert "403 Forbidden" in (source.last_error or "")
        mock_db.rollback.assert_awaited()
        # Verify crawl_run was added and committed with http_status 403
        assert mock_db.add.called
        crawl_run = mock_db.add.call_args[0][0]
        assert crawl_run.status == "failed"
        assert crawl_run.http_status == 403
        mock_db.commit.assert_awaited()


