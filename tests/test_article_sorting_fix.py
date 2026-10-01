import pytest
import datetime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func
from app.models.models import Base, Article
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI, APIRouter, Depends
from typing import List, Optional
from app.schemas.schemas import ArticleResponse


@pytest.mark.asyncio
async def test_sorting_coalesce_newest_oldest_score():
    """Verify that coalesce(Article.published_at, Article.created_at) properly handles null published_at."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        # Article 1: Recent created_at, published_at is None
        art_null_pub = Article(
            id=1,
            title="Article Without Published At",
            url="https://example.com/1",
            published_at=None,
            created_at=datetime.datetime(2026, 10, 1, 12, 0),
            relevance_score=6.0,
            is_read=False,
            is_hidden=False,
        )
        # Article 2: Older created_at, published_at yesterday
        art_yesterday = Article(
            id=2,
            title="Article Published Yesterday",
            url="https://example.com/2",
            published_at=datetime.datetime(2026, 9, 30, 12, 0),
            created_at=datetime.datetime(2026, 9, 30, 8, 0),
            relevance_score=9.5,
            is_read=False,
            is_hidden=False,
        )
        # Article 3: Oldest article
        art_oldest = Article(
            id=3,
            title="Article Published Last Month",
            url="https://example.com/3",
            published_at=datetime.datetime(2026, 9, 1, 12, 0),
            created_at=datetime.datetime(2026, 9, 1, 8, 0),
            relevance_score=8.0,
            is_read=False,
            is_hidden=False,
        )
        session.add_all([art_null_pub, art_yesterday, art_oldest])
        await session.commit()

        effective_date = func.coalesce(Article.published_at, Article.created_at)

        # 1. Test sort_by = newest (effective_date desc)
        stmt_newest = select(Article).order_by(effective_date.desc().nulls_last(), Article.id.desc())
        items_newest = (await session.execute(stmt_newest)).scalars().all()
        assert [a.id for a in items_newest] == [1, 2, 3]

        # 2. Test sort_by = oldest (effective_date asc)
        stmt_oldest = select(Article).order_by(effective_date.asc().nulls_last(), Article.id.asc())
        items_oldest = (await session.execute(stmt_oldest)).scalars().all()
        assert [a.id for a in items_oldest] == [3, 2, 1]

        # 3. Test sort_by = score (score desc, effective_date desc)
        stmt_score = select(Article).order_by(
            Article.relevance_score.desc(),
            effective_date.desc().nulls_last(),
            Article.id.desc(),
        )
        items_score = (await session.execute(stmt_score)).scalars().all()
        assert [a.id for a in items_score] == [2, 3, 1]  # 9.5 -> 8.0 -> 6.0


@pytest.mark.asyncio
async def test_crawler_pub_date_fallbacks():
    """Verify crawler parse_date supports published_parsed, updated_parsed, and raw string parsing."""
    from unittest.mock import MagicMock, patch
    from app.crawlers.article_crawler import fetch_rss_feed

    # Sub-case A: updated_parsed fallback
    entry_updated = MagicMock()
    entry_updated.title = "Updated Fallback"
    entry_updated.link = "http://test.com/up"
    entry_updated.published_parsed = None
    entry_updated.updated_parsed = (2026, 10, 1, 15, 30, 0, 3, 274, 0)
    entry_updated.summary = "summary"

    # Sub-case B: raw string dateutil fallback
    entry_raw_str = MagicMock()
    entry_raw_str.title = "Raw String Fallback"
    entry_raw_str.link = "http://test.com/raw"
    entry_raw_str.published_parsed = None
    entry_raw_str.updated_parsed = None
    entry_raw_str.published = "2026-10-01T14:00:00Z"
    entry_raw_str.summary = "summary"

    with patch("feedparser.parse") as mock_fp:
        mock_fp.return_value = MagicMock(entries=[entry_updated, entry_raw_str])
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = MagicMock(text="mock_xml", raise_for_status=MagicMock())
            items = await fetch_rss_feed("http://fake-feed")

            assert len(items) == 2
            assert items[0]["published_at"] == datetime.datetime(2026, 10, 1, 15, 30, tzinfo=datetime.timezone.utc)
            assert items[1]["published_at"] is not None
            assert items[1]["published_at"].year == 2026
            assert items[1]["published_at"].month == 10
            assert items[1]["published_at"].day == 1
