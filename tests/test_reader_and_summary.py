import pytest
import datetime
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.models import Article
from app.core.database import AsyncSessionLocal
from app.schemas.schemas import AIAnalysisResult, TechStackItem


@pytest.mark.asyncio
async def test_get_article_reader_mode():
    unique_key = str(int(datetime.datetime.now().timestamp()))
    async with AsyncSessionLocal() as db:
        art = Article(
            title=f"Reader Test Article {unique_key}",
            url=f"https://test.com/reader-{unique_key}",
            raw_content="This is a test article content designed to test reader mode in TechPulse.",
            published_at=datetime.datetime(2026, 9, 12),
            relevance_score=8.0,
            is_read=False,
            is_bookmarked=False,
        )
        db.add(art)
        await db.commit()
        await db.refresh(art)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get(f"/api/articles/{art.id}/reader")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == art.id
        assert data["title"] == art.title
        assert "This is a test article content" in data["content"]
        assert data["reading_time_minutes"] >= 1
        assert data["is_bookmarked"] is False


@pytest.mark.asyncio
async def test_summarize_article_on_demand():
    unique_key = str(int(datetime.datetime.now().timestamp()))
    async with AsyncSessionLocal() as db:
        art = Article(
            title=f"Summarize Test Article {unique_key}",
            url=f"https://test.com/summarize-{unique_key}",
            raw_content="PostgreSQL 18 introduces async I/O improvements for high concurrency databases.",
            published_at=datetime.datetime(2026, 9, 12),
            is_processed=False,
        )
        db.add(art)
        await db.commit()
        await db.refresh(art)

    mock_analysis = AIAnalysisResult(
        relevance_score=8.8,
        is_worth_reading=True,
        target_audience=["Backend Engineer"],
        vietnamese_title="PostgreSQL 18 Cải tiến Async I/O",
        vietnamese_summary="Bài viết phân tích các nâng cấp Async I/O trong PostgreSQL 18.",
        key_takeaways=[
            "Async I/O giúp tăng throughput lên 30%",
            "Giảm contention buffer pool",
        ],
        new_tech_stack=[
            TechStackItem(
                name="PostgreSQL 18", category="Database", desc="Async I/O engine"
            )
        ],
        tags=["PostgreSQL", "Database"],
    )

    with patch(
        "app.api.endpoints.analyze_article_with_9router",
        return_value=(mock_analysis, "gemini-2.5-flash"),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/articles/{art.id}/summarize")
            assert r.status_code == 200
            data = r.json()
            assert data["id"] == art.id
            assert data["is_processed"] is True
            assert data["vietnamese_title"] == "PostgreSQL 18 Cải tiến Async I/O"
            assert len(data["key_takeaways"]) == 2
