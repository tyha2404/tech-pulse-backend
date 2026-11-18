import asyncio
import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.database import Base
from app.models.models import Article, NewsVideo, Source
from app.services import video_service as vs


@pytest.fixture
async def Session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    vs.video_job.reset()
    # Never reach Telegram or the real TTS/renderer from these tests
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "")
    monkeypatch.setattr(settings, "VIDEO_OUTPUT_DIR", str(tmp_path))

    async def fake_script(articles):
        return vs._fallback_script(articles), "test"

    async def fake_render(segments, out: Path, on_step=None):
        if on_step:
            on_step("Dựng video (1/1)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return 30.0

    monkeypatch.setattr(vs, "generate_script", fake_script)
    monkeypatch.setattr(vs, "render_video", fake_render)


async def _seed(Session, n=4):
    async with Session() as db:
        src = Source(name="HN", url="https://hn.test")
        db.add(src)
        await db.flush()
        for i in range(1, n + 1):
            db.add(Article(
                title=f"T{i}", url=f"https://e/{i}", source_id=src.id, is_processed=True,
                relevance_score=9.0 - i * 0.1, vietnamese_title=f"Tin {i}", vietnamese_summary="Câu một.",
                tags=["AI"], cluster_topic_key=f"k{i}",
            ))
        await db.commit()


async def test_second_run_skips_articles_already_in_a_video(Session):
    await _seed(Session, n=4)
    async with Session() as db:
        first = await vs.generate_news_video(db, limit=3)
        assert first["success"] and len(first["article_ids"]) == 3
        # Only 1 unused article is left: not enough for another video
        second = await vs.generate_news_video(db, limit=3)
        assert second["success"] is False and second["skipped"] == "not_enough_new_articles"
        assert "1 bài mới" in second["message"]
        assert (await db.execute(vs.select(NewsVideo))).scalars().all().__len__() == 1


async def test_new_articles_make_a_new_video_without_repeating_old_ones(Session):
    await _seed(Session, n=3)
    async with Session() as db:
        first = await vs.generate_news_video(db, limit=3)
        async with Session() as db2:
            src = (await db2.execute(vs.select(Source))).scalars().first()
            for i in range(10, 13):
                db2.add(Article(title=f"N{i}", url=f"https://e/{i}", source_id=src.id, is_processed=True,
                                relevance_score=8.5, vietnamese_summary="S", tags=["AI"], cluster_topic_key=f"n{i}"))
            await db2.commit()
        second = await vs.generate_news_video(db, limit=3)
        assert second["success"]
        assert not set(first["article_ids"]) & set(second["article_ids"])


async def test_force_reuses_used_articles(Session):
    await _seed(Session, n=3)
    async with Session() as db:
        assert (await vs.generate_news_video(db, limit=3))["success"]
        assert (await vs.generate_news_video(db, limit=3))["success"] is False
        assert (await vs.generate_news_video(db, limit=3, reuse=True))["success"] is True


async def test_telegram_failure_still_records_the_video(Session, monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "1")
    monkeypatch.setattr(vs, "send_video_to_telegram", AsyncMock(return_value=False))
    await _seed(Session, n=3)
    async with Session() as db:
        r = await vs.generate_news_video(db, limit=3)
        assert r["success"] and r["telegram_sent"] is False
        # The failed upload must not cause the same articles to be built again
        assert (await vs.generate_news_video(db, limit=3))["skipped"] == "not_enough_new_articles"


async def test_busy_lock_blocks_parallel_runs(Session):
    await _seed(Session, n=3)
    async with Session() as db:
        async with vs.video_lock:
            r = await vs.generate_news_video(db, limit=3)
        assert r["skipped"] == "busy"


async def _wait_done(timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if vs.video_job.status != "running":
            return
        await asyncio.sleep(0.02)
    raise AssertionError("video job did not finish")


async def test_job_starts_in_background_and_reports_the_result(Session):
    await _seed(Session, n=3)
    assert vs.video_job.snapshot()["status"] == "idle"
    with patch("app.core.database.AsyncSessionLocal", Session):
        started = await vs.start_video_job(limit=3)
        assert started == {"started": True}
        assert vs.video_job.snapshot()["status"] == "running"
        await _wait_done()
    snap = vs.video_job.snapshot()
    assert snap["status"] == "done" and snap["result"]["success"] and snap["error"] is None
    assert len(snap["result"]["article_ids"]) == 3


async def test_job_refuses_when_not_enough_new_articles_or_already_running(Session):
    await _seed(Session, n=2)
    with patch("app.core.database.AsyncSessionLocal", Session):
        r = await vs.start_video_job(limit=3)
        assert r["started"] is False and r["skipped"] == "not_enough_new_articles" and "2 bài mới" in r["message"]
        assert vs.video_job.snapshot()["status"] == "idle"  # a refusal leaves no job behind

        await _seed_more(Session)
        assert (await vs.start_video_job(limit=3))["started"] is True
        assert (await vs.start_video_job(limit=3))["skipped"] == "busy"
        await _wait_done()


async def _seed_more(Session):
    async with Session() as db:
        src = (await db.execute(vs.select(Source))).scalars().first()
        db.add(Article(title="X", url="https://e/99", source_id=src.id, is_processed=True,
                       relevance_score=8.0, vietnamese_summary="S", tags=["AI"], cluster_topic_key="x"))
        await db.commit()


async def test_job_failure_is_reported_not_raised(Session, monkeypatch):
    async def boom(segments, out, on_step=None):
        raise RuntimeError("tts down")

    monkeypatch.setattr(vs, "render_video", boom)
    await _seed(Session, n=3)
    with patch("app.core.database.AsyncSessionLocal", Session):
        assert (await vs.start_video_job(limit=3))["started"] is True
        await _wait_done()
    snap = vs.video_job.snapshot()
    assert snap["status"] == "failed" and "tts down" in snap["error"]
    async with Session() as db:  # nothing recorded, so the same articles can be retried
        assert (await db.execute(vs.select(NewsVideo))).scalars().all() == []
