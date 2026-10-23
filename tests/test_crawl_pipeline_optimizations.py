import datetime
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.models.models import Base, Article, Source, CrawlRun
from app.services import crawl_service
from app.services.ai_analyzer import extractive_heuristic_fallback


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


def _feed_item(url: str, title: str) -> dict:
    return {
        "title": title,
        "url": url,
        "author": None,
        "published_at": datetime.datetime.now(datetime.timezone.utc),
        "raw_content": "word " * 600,
    }


async def _fake_analysis(title, content, url):
    analysis = extractive_heuristic_fallback(title, content, url)
    analysis.relevance_score = 8.0
    analysis.cluster_topic_key = None
    return analysis, "test-model"


@pytest.mark.asyncio
async def test_crawl_dedupes_and_notifies_with_source_name(session):
    """End-to-end crawl on a real AsyncSession: batch dedupe (DB + within feed),
    precomputed reading time, no fake embedding, and the alert carries the source name."""
    source = Source(name="Real Session Source", url="https://src.test", feed_url="https://src.test/rss", source_type="rss")
    session.add(source)
    await session.flush()
    session.add(Article(source_id=source.id, title="Old", url="https://src.test/old"))
    await session.commit()

    items = [
        _feed_item("https://src.test/old", "Already stored"),
        _feed_item("https://src.test/new", "Brand new NestJS release"),
        _feed_item("https://src.test/new", "Same URL twice in one feed"),
    ]

    with patch.object(crawl_service, "fetch_rss_feed", AsyncMock(return_value=items)), \
         patch.object(crawl_service, "analyze_article_with_9router", side_effect=_fake_analysis), \
         patch.object(crawl_service, "generate_article_embedding", AsyncMock(return_value=None)), \
         patch("app.services.telegram_service.send_telegram_message", new_callable=AsyncMock) as mock_send:
        added = await crawl_service.crawl_single_source(source, session, run_ai=True)

    assert added == 1
    urls = (await session.execute(select(Article.url))).scalars().all()
    assert sorted(urls) == ["https://src.test/new", "https://src.test/old"]

    new_article = (await session.execute(select(Article).where(Article.url == "https://src.test/new"))).scalar_one()
    assert new_article.reading_time_minutes == 3
    assert new_article.embedding is None  # left for backfill, not a fake vector

    assert mock_send.await_count == 1
    assert "Real Session Source" in mock_send.call_args[0][0]

    run = (await session.execute(select(CrawlRun))).scalar_one()
    assert run.status == "success"
    assert run.articles_found == 3
    assert run.articles_new == 1


@pytest.mark.asyncio
async def test_crawl_failure_mid_transaction_is_recorded(session):
    """Regression: after rollback() `source` is expired; reading source.name in the error
    handler raised MissingGreenlet, so no failed CrawlRun was written and the source
    stayed stuck in 'pending'."""
    source = Source(name="Flaky", url="https://flaky.test", feed_url="https://flaky.test/rss", source_type="rss")
    session.add(source)
    await session.commit()

    with patch.object(crawl_service, "fetch_rss_feed", AsyncMock(return_value=[_feed_item("https://flaky.test/a", "A")])), \
         patch.object(crawl_service, "assign_article_cluster", side_effect=RuntimeError("boom")):
        added = await crawl_service.crawl_single_source(source, session, run_ai=False)

    assert added == 0
    run = (await session.execute(select(CrawlRun))).scalar_one()
    assert run.status == "failed"
    assert run.error_message == "boom"
    assert source.status == "error"


@pytest.mark.asyncio
async def test_crawl_all_skips_when_crawl_already_running():
    with patch("app.core.database.AsyncSessionLocal") as mock_session_maker:
        async with crawl_service.crawl_lock:
            total = await crawl_service.crawl_all_active_sources(run_ai=False)
    assert total == 0
    mock_session_maker.assert_not_called()


def test_estimate_reading_time():
    assert crawl_service.estimate_reading_time(None) == 1
    assert crawl_service.estimate_reading_time("word " * 1000) == 5
