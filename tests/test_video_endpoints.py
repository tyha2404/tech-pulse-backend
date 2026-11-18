from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app
from app.models.models import Article, NewsVideo, Source
from app.services import video_service as vs


@pytest.fixture
async def client(tmp_path, monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override():
        async with Session() as db:
            yield db

    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(settings, "VIDEO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "")
    vs.video_job.reset()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        c.Session, c.tmp = Session, tmp_path
        yield c
    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


async def _video(client, article_ids, with_file=True, telegram=None):
    async with client.Session() as db:
        db.add(NewsVideo(file="techpulse-20260101-000000.mp4", article_ids=article_ids, caption="Cap #tech",
                         duration_sec=31.2, telegram_sent=telegram))
        await db.commit()
    if with_file:
        (client.tmp / "techpulse-20260101-000000.mp4").write_bytes(b"mp4")


async def _articles(client):
    async with client.Session() as db:
        src = Source(name="HN", url="https://hn.test")
        db.add(src)
        await db.flush()
        for i in (1, 2):
            db.add(Article(id=i, title=f"T{i}", vietnamese_title=f"Tin {i}", url=f"https://e/{i}", source_id=src.id))
        await db.commit()


async def test_status_is_idle_by_default(client):
    r = await client.get("/api/video/status")
    assert r.status_code == 200 and r.json()["status"] == "idle"


async def test_list_videos_includes_urls_and_the_articles_used(client):
    await _articles(client)
    await _video(client, [2, 1], telegram=True)
    r = await client.get("/api/videos")
    (v,) = r.json()
    assert v["url"] == "/video/files/techpulse-20260101-000000.mp4" and v["telegram_sent"] is True
    assert [a["title"] for a in v["articles"]] == ["Tin 2", "Tin 1"]  # keeps the countdown order
    assert (await client.get("/api" + v["url"])).status_code == 200


async def test_resend_requires_telegram_config_and_an_existing_file(client, monkeypatch):
    await _articles(client)
    await _video(client, [1, 2])
    assert (await client.post("/api/videos/1/telegram")).status_code == 400

    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "1")
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(vs, "send_video_to_telegram", send)
    r = await client.post("/api/videos/1/telegram")
    assert r.json()["success"] is True and send.await_count == 1
    assert (await client.get("/api/videos")).json()[0]["telegram_sent"] is True

    assert (await client.post("/api/videos/99/telegram")).status_code == 404
    (client.tmp / "techpulse-20260101-000000.mp4").unlink()
    assert (await client.post("/api/videos/1/telegram")).status_code == 404


async def test_generate_returns_immediately_with_a_reason_when_nothing_to_build(client, monkeypatch):
    from unittest.mock import patch

    with patch("app.core.database.AsyncSessionLocal", client.Session):
        r = await client.post("/api/video/generate")
    assert r.status_code == 200
    body = r.json()
    assert body["started"] is False and body["skipped"] == "not_enough_new_articles"
