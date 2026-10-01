import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import scheduled_crawl_job, lifespan, app
from app.services.crawl_service import crawl_all_active_sources, crawl_single_source
from app.models.models import Source


@pytest.mark.asyncio
async def test_scheduled_crawl_job_catches_cancelled_error():
    """Verify scheduled_crawl_job catches CancelledError gracefully without crashing"""
    with patch(
        "app.main.crawl_all_active_sources",
        side_effect=asyncio.CancelledError("Job cancelled during PM2 restart"),
    ):
        # Should NOT raise CancelledError
        await scheduled_crawl_job()


@pytest.mark.asyncio
async def test_crawl_single_source_cancels_and_rolls_back():
    """Verify crawl_single_source rolls back session when cancelled"""
    mock_db = AsyncMock()
    source = Source(id=1, name="Test Source", url="https://test.com", source_type="rss")

    with patch(
        "app.services.crawl_service.fetch_rss_feed",
        side_effect=asyncio.CancelledError("Cancellation token triggered"),
    ):
        with pytest.raises(asyncio.CancelledError):
            await crawl_single_source(source, mock_db)

        # Ensure rollback was attempted upon cancellation
        mock_db.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_crawl_all_active_sources_cancels_subtasks_cleanly():
    """Verify crawl_all_active_sources properly cancels workers on CancelledError"""
    with patch("app.core.database.AsyncSessionLocal") as mock_session_maker:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_source = MagicMock()
        mock_source.id = 1
        mock_result.scalars.return_value.all.return_value = [mock_source]
        mock_db.execute.return_value = mock_result
        mock_session_maker.return_value.__aenter__.return_value = mock_db

        async def slow_crawl(*args, **kwargs):
            await asyncio.sleep(10)
            return 1

        with patch("app.services.crawl_service.crawl_single_source", side_effect=slow_crawl):
            task = asyncio.create_task(crawl_all_active_sources(concurrency_limit=2))
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_lifespan_graceful_shutdown():
    """Verify lifespan shutdown invokes scheduler.shutdown(wait=False) and engine.dispose()"""
    with patch("app.main.scheduler") as mock_scheduler, \
         patch("app.main.engine") as mock_engine, \
         patch("app.main.AsyncSessionLocal") as mock_session_maker:

        mock_scheduler.running = True
        mock_engine.dispose = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        mock_db = AsyncMock()
        mock_res = MagicMock()
        mock_res.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_res
        mock_session_maker.return_value.__aenter__.return_value = mock_db

        async with lifespan(app):
            pass

        mock_scheduler.shutdown.assert_called_with(wait=False)
        mock_engine.dispose.assert_awaited_once()
