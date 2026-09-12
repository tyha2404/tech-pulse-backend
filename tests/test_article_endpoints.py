import pytest
from app.schemas.schemas import ArticleResponse, ArticleUpdate


def test_article_response_fields():
    data = {
        "id": 1,
        "title": "Test Title",
        "url": "https://example.com/test",
        "is_processed": True,
        "is_worth_reading": True,
        "relevance_score": 8.5,
        "key_takeaways": [],
        "new_tech_stacks": [],
        "tags": [],
        "target_audience": [],
        "created_at": "2026-09-12T10:00:00",
        "is_read": True,
        "is_hidden": False,
        "is_bookmarked": True,
        "reading_time_minutes": 3,
    }
    resp = ArticleResponse.model_validate(data)
    assert resp.is_read is True
    assert resp.is_hidden is False
    assert resp.is_bookmarked is True
    assert resp.reading_time_minutes == 3


def test_article_update_schema():
    payload = ArticleUpdate(is_read=True, is_hidden=False, is_bookmarked=True)
    assert payload.is_read is True
    assert payload.is_hidden is False
    assert payload.is_bookmarked is True

    partial = ArticleUpdate(is_bookmarked=False)
    assert partial.is_read is None
    assert partial.is_hidden is None
    assert partial.is_bookmarked is False


@pytest.mark.asyncio
async def test_get_articles_sorting_and_filtering():
    import datetime
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    from app.models.models import Article
    from app.core.database import AsyncSessionLocal
    from sqlalchemy.future import select

    unique_key = str(int(datetime.datetime.now().timestamp()))
    async with AsyncSessionLocal() as db:
        # Create test records
        art1 = Article(
            title=f"Old Article Test {unique_key}",
            url=f"https://test.com/old-{unique_key}",
            published_at=datetime.datetime(2025, 1, 1),
            relevance_score=5.0,
            is_read=False,
            is_hidden=False,
        )
        art2 = Article(
            title=f"New Top Article Test {unique_key}",
            url=f"https://test.com/new-{unique_key}",
            published_at=datetime.datetime(2026, 9, 1),
            relevance_score=9.0,
            is_read=True,
            is_hidden=False,
        )
        art3 = Article(
            title=f"Hidden Article Test {unique_key}",
            url=f"https://test.com/hidden-{unique_key}",
            published_at=datetime.datetime(2026, 9, 2),
            relevance_score=6.0,
            is_read=False,
            is_hidden=True,
        )
        db.add_all([art1, art2, art3])
        await db.commit()
        await db.refresh(art1)
        await db.refresh(art2)
        await db.refresh(art3)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test default sorting: hidden articles should be excluded
        r = await ac.get(f"/api/articles?query={unique_key}&sort_by=newest")
        assert r.status_code == 200
        items = r.json()
        ids = [i["id"] for i in items]
        assert art3.id not in ids
        assert art2.id in ids

        # Test include_hidden=true
        r_hidden = await ac.get(f"/api/articles?query={unique_key}&include_hidden=true")
        assert r_hidden.status_code == 200
        hidden_ids = [i["id"] for i in r_hidden.json()]
        assert art3.id in hidden_ids
        assert art2.id not in hidden_ids

        # Test read_status=unread
        r_unread = await ac.get(f"/api/articles?query={unique_key}&read_status=unread")
        assert r_unread.status_code == 200
        unread_ids = [i["id"] for i in r_unread.json()]
        assert art1.id in unread_ids
        assert art2.id not in unread_ids

        # Test read_status=read
        r_read = await ac.get(f"/api/articles?query={unique_key}&read_status=read")
        assert r_read.status_code == 200
        read_ids = [i["id"] for i in r_read.json()]
        assert art2.id in read_ids
        assert art1.id not in read_ids

        # Test sort_by=score
        r_score = await ac.get(f"/api/articles?query={unique_key}&sort_by=score")
        assert r_score.status_code == 200
        score_items = r_score.json()
        assert len(score_items) == 2
        assert score_items[0]["id"] == art2.id  # score 9.0 > 5.0
        assert score_items[1]["id"] == art1.id

        # Test sort_by=oldest
        r_oldest = await ac.get(f"/api/articles?query={unique_key}&sort_by=oldest")
        assert r_oldest.status_code == 200
        oldest_items = r_oldest.json()
        assert oldest_items[0]["id"] == art1.id  # 2025 comes before 2026

        # Test source_id filter
        r_source = await ac.get(f"/api/articles?query={unique_key}&source_id=999999")
        assert r_source.status_code == 200
        assert len(r_source.json()) == 0

        # Test PATCH article status (read, hidden, bookmarked)
        r_patch = await ac.patch(
            f"/api/articles/{art1.id}",
            json={"is_read": True, "is_hidden": True, "is_bookmarked": True},
        )
        assert r_patch.status_code == 200
        assert r_patch.json()["is_read"] is True
        assert r_patch.json()["is_hidden"] is True
        assert r_patch.json()["is_bookmarked"] is True

        # Test bookmarked_only=true
        r_bm = await ac.get(f"/api/articles?bookmarked_only=true&include_hidden=true")
        assert r_bm.status_code == 200
        bm_ids = [i["id"] for i in r_bm.json()]
        assert art1.id in bm_ids
