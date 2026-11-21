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


async def test_edge_retries_keep_the_same_voice(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge(calls, fail=True))
    with pytest.raises(RuntimeError, match="edge-tts failed"):
        await vs.synthesize("Xin chào", tmp_path / "a0.mp3")
    assert len(calls) == vs.TTS_MAX_ATTEMPTS and set(calls) == {settings.VIDEO_TTS_VOICE}


async def test_one_failing_segment_revoices_whole_video_with_gtts(tmp_path, monkeypatch):
    class Comm:
        def __init__(self, text, voice, rate=None):
            self.text = text

        async def save(self, path):
            if "hỏng" in self.text:
                raise RuntimeError("NoAudioReceived")
            open(path, "wb").write(b"edge")

    monkeypatch.setattr(vs.edge_tts, "Communicate", Comm)
    monkeypatch.setattr(vs, "_gtts_save", lambda text, out: out.write_bytes(b"gtts"))
    d = tmp_path / "t"
    d.mkdir()
    durations = await vs.synthesize_all(["Một.", "Đoạn hỏng.", "Ba."], d)
    assert len(durations) == 3
    assert {(d / f"a{i}.mp3").read_bytes() for i in range(3)} == {b"gtts"}  # no mixed voices


async def test_all_edge_when_it_works(tmp_path, monkeypatch):
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge([], fail=False))
    monkeypatch.setattr(vs, "_gtts_save", lambda text, out: pytest.fail("gTTS must not be used"))
    d = tmp_path / "t"
    d.mkdir()
    await vs.synthesize_all(["Một.", "Hai."], d)
    assert {(d / f"a{i}.mp3").read_bytes() for i in range(2)} == {b"edge"}


async def test_engines_do_not_share_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge([], fail=False))
    monkeypatch.setattr(vs, "_gtts_save", lambda text, out: out.write_bytes(b"gtts"))
    await vs.synthesize("Xin chào", tmp_path / "a.mp3", "edge")
    await vs.synthesize("Xin chào", tmp_path / "b.mp3", "gtts")
    assert (tmp_path / "b.mp3").read_bytes() == b"gtts"


async def test_both_engines_failing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge([], fail=True))

    def boom(text, out):
        raise OSError("blocked")

    monkeypatch.setattr(vs, "_gtts_save", boom)
    d = tmp_path / "t"
    d.mkdir()
    with pytest.raises(RuntimeError, match="gTTS failed"):
        await vs.synthesize_all(["Xin chào"], d)


async def test_fallback_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "VIDEO_TTS_FALLBACK_GTTS", False)
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge([], fail=True))
    d = tmp_path / "t"
    d.mkdir()
    with pytest.raises(RuntimeError, match="edge-tts failed"):
        await vs.synthesize_all(["Xin chào"], d)


async def test_cached_segment_skips_tts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vs.edge_tts, "Communicate", _fake_edge(calls, fail=False))
    await vs.synthesize("Xin chào", tmp_path / "a0.mp3")
    n = len(calls)
    out = tmp_path / "b0.mp3"
    await vs.synthesize("Xin chào", out)  # a retried build: new temp dir, same text
    assert len(calls) == n and out.read_bytes() == b"edge"
