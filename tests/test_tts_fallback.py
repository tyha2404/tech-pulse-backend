import pytest

from app.core.config import settings
from app.services import video_service as vs


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "VIDEO_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(settings, "VIDEO_TTS_FALLBACK_GTTS", True)
    monkeypatch.setattr(vs, "audio_duration", lambda p: 2.5)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(vs.asyncio, "sleep", no_sleep)


def _fake_edge(calls, fail):
    class Comm:
        def __init__(self, text, voice, rate=None):
            calls.append(voice)

        async def save(self, path):
            if fail:
                raise RuntimeError("NoAudioReceived")
            open(path, "wb").write(b"edge")

    return Comm


async def test_falls_back_to_gtts_when_edge_fails(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge(calls, fail=True))
    monkeypatch.setattr(vs, "_gtts_save", lambda text, out: out.write_bytes(b"gtts"))
    out = tmp_path / "a0.mp3"
    assert await vs.synthesize("Xin chào", out) == 2.5
    assert out.read_bytes() == b"gtts"
    assert len(calls) == vs.TTS_MAX_ATTEMPTS


async def test_cached_segment_skips_tts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge(calls, fail=False))
    await vs.synthesize("Xin chào", tmp_path / "a0.mp3")
    n = len(calls)
    out = tmp_path / "b0.mp3"
    await vs.synthesize("Xin chào", out)  # a retried build: new temp dir, same text
    assert len(calls) == n and out.read_bytes() == b"edge"


async def test_both_engines_failing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge([], fail=True))

    def boom(text, out):
        raise OSError("blocked")

    monkeypatch.setattr(vs, "_gtts_save", boom)
    with pytest.raises(RuntimeError, match="TTS failed"):
        await vs.synthesize("Xin chào", tmp_path / "a0.mp3")
    assert not list((tmp_path).glob("*.part.mp3"))


async def test_fallback_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "VIDEO_TTS_FALLBACK_GTTS", False)
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge([], fail=True))
    with pytest.raises(RuntimeError, match="edge-tts failed"):
        await vs.synthesize("Xin chào", tmp_path / "a0.mp3")
