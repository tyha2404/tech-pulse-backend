import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.models import Article, Base, Source
from app.services import daily_briefing
from zoneinfo import ZoneInfo

from app.services.daily_briefing import (
    DAILY,
    TELEGRAM_LIMIT,
    UPDATE,
    previous_slot_start,
    _validate_briefing,
    fallback_briefing,
    format_daily_briefing_message,
    generate_daily_briefing,
)

NOW = datetime.datetime(2026, 10, 23, 22, 0)  # a Friday


def _article(i, score, tags=None, source="InfoQ", title=None):
    a = Article(
        id=i,
        title=title or f"Article {i}",
        vietnamese_title=title or f"Bài {i}",
        url=f"https://example.com/{i}",
        relevance_score=score,
        vietnamese_summary=f"Tóm tắt bài {i}.",
        key_takeaways=[f"Ý chính của bài {i}"],
        tags=tags or ["Backend"],
    )
    a.source = Source(name=source, url=f"https://{source}.test")
    return a


def _fake_llm(content: str):
    mock_cls = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
    return mock_cls


def test_validate_drops_unknown_ids_and_empty_themes():
    by_id = {1: _article(1, 9.0), 2: _article(2, 8.0)}
    data = {
        "headline": "Ngày bận rộn",
        "themes": [
            {"title": "Agents", "summary": "Hai bài.", "article_ids": [1, 999, "2", "x"]},
            {"title": "Ma", "summary": "Không có bài thật.", "article_ids": [42]},
            {"title": "", "summary": "Thiếu tiêu đề", "article_ids": [1]},
        ],
        "must_read": [{"id": 999, "why": "?"}, {"id": 1, "why": "Đáng đọc"}, {"id": 1, "why": "trùng"}],
    }
    out = _validate_briefing(data, by_id)
    assert out["themes"] == [{"title": "Agents", "summary": "Hai bài.", "article_ids": [1, 2]}]
    assert out["must_read"] == [{"id": 1, "title": "", "why": "Đáng đọc"}]
    assert _validate_briefing({"themes": [], "must_read": []}, by_id) is None


def test_format_message_escapes_html_and_counts_rest():
    articles = [_article(1, 9.2, title="<b>Lỗ hổng</b> & bản vá")] + [_article(i, 6.0) for i in range(2, 8)]
    briefing = {
        "headline": "Một ngày <script>",
        "themes": [{"title": "Bảo mật", "summary": "Bản vá khẩn.", "article_ids": [1, 2]}],
        "must_read": [{"id": 1, "why": "Ảnh hưởng mọi bản cài đặt"}],
    }
    msg, markup = format_daily_briefing_message(briefing, articles, now=NOW)

    assert "Siêu tổng hợp cuối ngày · Thứ sáu 23/10" in msg
    assert "7 bài mới" in msg
    assert "&lt;script&gt;" in msg and "<script>" not in msg
    assert "&lt;b&gt;Lỗ hổng&lt;/b&gt; &amp; bản vá" in msg
    assert "Còn 5 bài khác" in msg
    assert markup["inline_keyboard"][0][0]["text"] == "Mở TechPulse"


def test_format_message_stays_under_telegram_limit():
    articles = [_article(i, 8.0) for i in range(1, 30)]
    briefing = {
        "headline": "x" * 300,
        "themes": [
            {"title": f"Chủ đề {t}", "summary": "Rất dài. " * 200, "article_ids": [t * 3 + 1, t * 3 + 2, t * 3 + 3]}
            for t in range(4)
        ],
        "must_read": [{"id": i, "why": "Lý do " * 40} for i in (1, 2, 3)],
    }
    msg, _ = format_daily_briefing_message(briefing, articles, now=NOW)
    assert len(msg) <= TELEGRAM_LIMIT


def test_fallback_groups_by_tag_without_llm():
    articles = [
        _article(1, 9.0, tags=["AI"]),
        _article(2, 8.5, tags=["AI"]),
        _article(3, 7.0, tags=["Backend"]),
    ]
    out = fallback_briefing(articles)
    assert out["themes"][0] == {"title": "AI", "summary": "", "article_ids": [1, 2]}
    assert [m["id"] for m in out["must_read"]] == [1, 2, 3]
    msg, _ = format_daily_briefing_message(out, articles, now=NOW)
    assert "• Bài 1" in msg  # fallback themes list titles instead of prose


def test_fallback_skips_untranslated_articles():
    articles = [_article(1, 9.0, tags=["AI"], title="English title"), _article(2, 8.0, tags=["AI"])]
    articles[0].ai_model_used = "extractive-fallback"
    out = fallback_briefing(articles)
    assert out["themes"][0]["article_ids"] == [2]
    assert [m["id"] for m in out["must_read"]] == [2]
    msg, _ = format_daily_briefing_message(out, articles, now=NOW)
    assert "English title" not in msg and "Còn 1 bài khác" in msg


@pytest.mark.asyncio
async def test_generate_uses_llm_json():
    articles = [_article(1, 9.0), _article(2, 8.0)]
    llm = json.dumps(
        {
            "headline": "Hôm nay",
            "themes": [{"title": "T", "summary": "S", "article_ids": [1, 2]}],
            "must_read": [{"id": 2, "title": "Tiêu đề đã dịch", "why": "W"}],
        }
    )
    with patch("app.services.ai_analyzer.AsyncOpenAI", _fake_llm(f"```json\n{llm}\n```")):
        briefing, model = await generate_daily_briefing(articles)
    assert model != "fallback"
    assert briefing["must_read"] == [{"id": 2, "title": "Tiêu đề đã dịch", "why": "W"}]


def test_must_read_prefers_translated_title_over_stored_one():
    articles = [_article(1, 9.0, title="Postgres ships vector indexes")]
    briefing = {"headline": "", "themes": [], "must_read": [{"id": 1, "title": "Postgres có chỉ mục vector", "why": ""}]}
    msg, _ = format_daily_briefing_message(briefing, articles, now=NOW)
    assert "Postgres có chỉ mục vector" in msg and "Postgres ships" not in msg


@pytest.mark.asyncio
async def test_generate_falls_back_when_llm_returns_garbage():
    articles = [_article(1, 9.0, tags=["AI"])]
    with patch("app.services.ai_analyzer.AsyncOpenAI", _fake_llm('{"themes": [{"article_ids": [77]}]}')):
        briefing, model = await generate_daily_briefing(articles)
    assert model == "fallback"
    assert briefing["must_read"][0]["id"] == 1


@pytest.mark.asyncio
async def test_dispatch_end_to_end_on_sqlite():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    now = datetime.datetime.now(datetime.timezone.utc)
    async with Session() as db:
        src = Source(name="The Verge", url="https://verge.test")
        db.add(src)
        await db.flush()
        common = dict(source_id=src.id, is_processed=True, vietnamese_summary="S", tags=["AI"])
        db.add_all(
            [
                Article(title="Fresh", url="https://e/1", relevance_score=9.0, created_at=now, **common),
                Article(title="Old", url="https://e/2", relevance_score=9.5, created_at=now - datetime.timedelta(days=2), **common),
                Article(title="Hidden", url="https://e/3", relevance_score=9.9, created_at=now, is_hidden=True, **common),
                Article(title="Dup", url="https://e/4", relevance_score=9.9, created_at=now, is_canonical=False, **common),
            ]
        )
        await db.commit()

    with patch("app.core.database.AsyncSessionLocal", Session), \
         patch("app.services.ai_analyzer.AsyncOpenAI", _fake_llm("not json")), \
         patch.object(daily_briefing, "send_telegram_message", new_callable=AsyncMock, return_value=True) as send:
        assert await daily_briefing.dispatch_daily_briefing() is True

    msg = send.call_args[0][0]
    assert "1 bài mới" in msg and "Fresh" in msg
    assert "Old" not in msg and "Hidden" not in msg and "Dup" not in msg
    assert send.call_args.kwargs["disable_preview"] is True
    await engine.dispose()


@pytest.mark.asyncio
async def test_tomtat_command_only_for_configured_chat():
    from app.core.database import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: AsyncMock()
    try:
        with patch("app.api.endpoints.settings.TELEGRAM_CHAT_ID", "12345"), \
             patch("app.services.daily_briefing.dispatch_daily_briefing", new_callable=AsyncMock) as dispatch:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
                ok = await client.post("/api/webhooks/telegram", json={"message": {"text": "/tomtat", "chat": {"id": 12345}}})
                other = await client.post("/api/webhooks/telegram", json={"message": {"text": "/tomtat", "chat": {"id": 999}}})
        assert ok.json()["action"] == "briefing"
        assert "action" not in other.json()
        assert dispatch.await_count == 1
    finally:
        app.dependency_overrides.clear()


VN = ZoneInfo("Asia/Ho_Chi_Minh")


@pytest.mark.parametrize(
    "now_hour, expected",
    [
        (9, (22, 22)),  # 9:00 covers since 22:00 the day before
        (12, (23, 9)),
        (18, (23, 12)),
        (22, (23, 18)),
    ],
)
def test_previous_slot_start_windows(now_hour, expected):
    now = datetime.datetime(2026, 10, 23, now_hour, 0, 5, tzinfo=VN)
    start = previous_slot_start(now, [9, 12, 18, 22])
    assert (start.day, start.hour, start.minute) == (expected[0], expected[1], 0)


def test_update_hours_setting_ignores_junk_and_daily_hour():
    from app.core.config import settings

    with patch.object(settings, "BRIEFING_UPDATE_HOURS", "18, 9,x,25,22,12,9"):
        assert settings.briefing_update_hours_list == [9, 12, 18]


def test_update_message_header_and_trend_only_in_daily():
    articles = [_article(1, 8.5), _article(2, 7.0)]
    briefing = {
        "headline": "H",
        "trend": "Xu hướng lớn",
        "themes": [{"title": "T", "summary": "S", "article_ids": [1]}],
        "must_read": [{"id": 1, "why": "W"}],
    }
    now = datetime.datetime(2026, 10, 23, 12, 0, tzinfo=VN)
    since = datetime.datetime(2026, 10, 23, 9, 0, tzinfo=VN)

    upd, _ = format_daily_briefing_message({**briefing, "trend": ""}, articles, now=now, kind=UPDATE, since=since)
    assert "Cập nhật 12:00 · Thứ sáu 23/10" in upd
    assert "2 bài mới từ 09:00" in upd
    assert "CÓ GÌ MỚI" in upd and "XU HƯỚNG CỦA NGÀY" not in upd

    daily, _ = format_daily_briefing_message(briefing, articles, now=now, kind=DAILY)
    assert "XU HƯỚNG CỦA NGÀY" in daily and "Xu hướng lớn" in daily


@pytest.mark.asyncio
async def test_validate_keeps_trend_only_for_daily():
    by_id = {1: _article(1, 9.0)}
    data = {"trend": "T", "themes": [{"title": "a", "summary": "b", "article_ids": [1]}]}
    assert _validate_briefing(data, by_id, DAILY)["trend"] == "T"
    assert _validate_briefing(data, by_id, UPDATE)["trend"] == ""


@pytest.mark.asyncio
async def test_scheduled_update_only_covers_new_articles_and_skips_when_empty():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    now = datetime.datetime(2026, 10, 23, 12, 0, 3, tzinfo=VN)
    utc = lambda d: d.astimezone(datetime.timezone.utc)
    async with Session() as db:
        src = Source(name="InfoQ", url="https://infoq.test")
        db.add(src)
        await db.flush()
        common = dict(source_id=src.id, is_processed=True, vietnamese_summary="S", tags=["AI"])
        db.add_all(
            [
                Article(title="Before 9h", url="https://e/1", relevance_score=9.0,
                        created_at=utc(now.replace(hour=8, minute=30)), **common),
                Article(title="After 9h", url="https://e/2", relevance_score=7.0,
                        created_at=utc(now.replace(hour=10)), **common),
            ]
        )
        await db.commit()

    send = AsyncMock(return_value=True)
    with patch("app.core.database.AsyncSessionLocal", Session), \
         patch("app.services.ai_analyzer.AsyncOpenAI", _fake_llm("not json")), \
         patch.object(daily_briefing, "_local_now", return_value=now), \
         patch.object(daily_briefing, "send_telegram_message", send):
        assert await daily_briefing.dispatch_scheduled_update() is True
        msg = send.call_args[0][0]
        assert "After 9h" in msg and "Before 9h" not in msg
        assert "1 bài mới từ 09:00" in msg

        # 18:00 update: nothing created since 12:00, so nothing is sent
        send.reset_mock()
        with patch.object(daily_briefing, "_local_now", return_value=now.replace(hour=18)):
            assert await daily_briefing.dispatch_scheduled_update() is False
        send.assert_not_called()
    await engine.dispose()
