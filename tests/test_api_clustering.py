import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.database import AsyncSessionLocal, Base, engine
from app.models.models import Article, Source


@pytest.mark.asyncio
async def test_get_articles_group_duplicates():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import uuid

    rand_id = str(uuid.uuid4())[:8]
    import datetime

    now = datetime.datetime.utcnow()
    async with AsyncSessionLocal() as db:
        # Create test source
        src = Source(
            name=f"Test Source {rand_id}", url=f"https://source-{rand_id}.com/rss"
        )
        db.add(src)
        await db.commit()
        await db.refresh(src)

        # Create canonical article
        a1 = Article(
            source_id=src.id,
            title=f"Sự kiện ra mắt sản phẩm A {rand_id}",
            url=f"https://source-{rand_id}.com/news/1",
            cluster_id=f"cluster-{rand_id}",
            is_canonical=True,
            is_worth_reading=True,
            relevance_score=8.5,
            published_at=now,
        )
        # Create duplicate article from another source
        a2 = Article(
            source_id=src.id,
            title=f"Tin vắn: Đã ra mắt sản phẩm A {rand_id}",
            url=f"https://source-{rand_id}.com/news/2",
            cluster_id=f"cluster-{rand_id}",
            is_canonical=False,
            is_worth_reading=True,
            relevance_score=6.0,
            published_at=now,
        )
        db.add_all([a1, a2])
        await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test 1: group_duplicates=true (default)
        res = await ac.get("/api/articles?group_duplicates=true")
        assert res.status_code == 200
        articles = res.json()
        cluster_articles = [
            a for a in articles if a.get("cluster_id") == f"cluster-{rand_id}"
        ]
        assert len(cluster_articles) == 1
        assert cluster_articles[0]["is_canonical"] is True
        assert len(cluster_articles[0]["related_articles"]) == 1
        assert (
            cluster_articles[0]["related_articles"][0]["title"]
            == f"Tin vắn: Đã ra mắt sản phẩm A {rand_id}"
        )

        # Test 2: group_duplicates=false
        res_all = await ac.get("/api/articles?group_duplicates=false")
        assert res_all.status_code == 200
        all_articles = res_all.json()
        cluster_all = [
            a for a in all_articles if a.get("cluster_id") == f"cluster-{rand_id}"
        ]
        assert len(cluster_all) == 2
