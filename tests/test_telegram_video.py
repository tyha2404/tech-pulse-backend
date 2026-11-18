import logging
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import telegram_service, video_service
from app.services.telegram_service import send_telegram_video


class _FakeClient:
    calls: list = []
    status = 200
    boom = False

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, data=None, files=None):
        if _FakeClient.boom:
            raise RuntimeError(f"connect failed for {url}")
        _FakeClient.calls.append({"url": url, "data": data, "file": files["video"][0]})
        return SimpleNamespace(status_code=_FakeClient.status, text="x")


@pytest.fixture(autouse=True)
def fake_telegram(monkeypatch):
    _FakeClient.calls, _FakeClient.status, _FakeClient.boom = [], 200, False
    monkeypatch.setattr(telegram_service.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "42")


@pytest.fixture
def mp4(tmp_path):
    p = tmp_path / "techpulse-20260101-000000.mp4"
    p.write_bytes(b"\x00" * 1024)
    return p


async def test_sends_vertical_streamable_video_with_caption(mp4):
    assert await send_telegram_video(mp4, "<b>hi</b>", duration=36.6) is True
    call = _FakeClient.calls[0]
    assert call["url"].endswith("/sendVideo") and call["file"] == mp4.name
    d = call["data"]
    assert d["chat_id"] == "42" and d["parse_mode"] == "HTML" and d["supports_streaming"] == "true"
    assert (d["width"], d["height"], d["duration"]) == ("1080", "1920", "37")


async def test_caption_is_cut_to_telegram_limit(mp4):
    await send_telegram_video(mp4, "x" * 5000)
    assert len(_FakeClient.calls[0]["data"]["caption"]) == 1024


async def test_not_sent_without_config_missing_file_or_oversize(mp4, tmp_path, monkeypatch):
    assert await send_telegram_video(tmp_path / "nope.mp4") is False
    monkeypatch.setattr(telegram_service, "TELEGRAM_MAX_UPLOAD_BYTES", 10)
    assert await send_telegram_video(mp4) is False
    monkeypatch.setattr(telegram_service, "TELEGRAM_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    assert await send_telegram_video(mp4) is False
    assert _FakeClient.calls == []


async def test_http_error_returns_false_and_logs_no_token(mp4, caplog):
    _FakeClient.boom = True
    with caplog.at_level(logging.DEBUG):
        assert await send_telegram_video(mp4) is False
    assert "SECRET" not in caplog.text


async def test_video_service_sends_caption_with_linked_stories(mp4, monkeypatch):
    sent = {}

    async def fake_send(path, caption_html, duration=None, **k):
        sent.update(path=path, caption=caption_html, duration=duration)
        return True

    monkeypatch.setattr(telegram_service, "send_telegram_video", fake_send)
    arts = [
        SimpleNamespace(url="https://a.example/1?x=1&y=2", title="T1", vietnamese_title="Tin <1>", source=SimpleNamespace(name="HN")),
        SimpleNamespace(url="https://a.example/2", title="T2", vietnamese_title=None, source=None),
    ]
    assert await video_service.send_video_to_telegram(mp4, "Cap #tech", arts, 30.0) is True
    c = sent["caption"]
    assert "Top 2" in c and "Top 1" in c and "Tin &lt;1&gt;" in c and "x=1&amp;y=2" in c and "<b>Cap #tech</b>" in c
