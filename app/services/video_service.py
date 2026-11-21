"""Short vertical news video (TikTok/Reels/Shorts) built from the best recent articles.

Pipeline: pick articles -> LLM script (fallback: summaries) -> edge-tts voice per segment
-> Pillow card per segment (1080x1920) -> ffmpeg clips -> concat to one mp4.
Cards are drawn by us, never the source article's images (copyright)."""
import asyncio
import datetime
import hashlib
import json
import logging
import random
import re
import subprocess
import tempfile
import time
from sqlalchemy import select
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import edge_tts
import imageio_ffmpeg

from app.core.config import settings
from app.models.models import Article, NewsVideo
from app.services.ai_analyzer import (
    record_ai_failure,
    _clean_json_response,
    get_ai_client,
    message_text,
)
from app.services.circuit_breaker import ai_circuit_breaker
from app.services.video_scenes import H, W, Scene, fetch_article_image
from app.services.daily_briefing import (
    _local_now,
    collect_briefing_articles,
    short_description,
)

logger = logging.getLogger(__name__)

FPS = 30
TTS_RATE = "+8%"

# The daily video is built this long after the end-of-day briefing, so the two don't compete for the LLM
DAILY_VIDEO_DELAY_MINUTES = 10

# A video covers every unused article at or above BRIEFING_MIN_SCORE, newest window first. Fewer than
# MIN_ARTICLES is too thin to be a video; more than MAX_ARTICLES (~3 min, well under Telegram's 50 MB)
# keeps the lowest-scored extras unused, so the next video picks them up and nothing is lost.
MIN_ARTICLES = 3
MAX_ARTICLES = 20
SECONDS_PER_ARTICLE = 9


def daily_video_schedule() -> Tuple[int, int]:
    """(hour, minute) of the automatic daily video: shortly after the DAILY briefing."""
    total = settings.DAILY_BRIEFING_HOUR * 60 + settings.DAILY_BRIEFING_MINUTE + DAILY_VIDEO_DELAY_MINUTES
    return (total // 60) % 24, total % 60


# One video at a time: rendering takes minutes and the TTS endpoint throttles parallel calls
video_lock = asyncio.Lock()

SCRIPT_PROMPT = """Bạn là biên tập viên bản tin công nghệ ngắn trên TikTok cho lập trình viên Việt Nam. Viết lời đọc cho video dọc khoảng {seconds} giây: "Top {count} tin công nghệ hôm nay". Video KHÔNG có đoạn mở đầu hay kết thúc riêng: vào thẳng tin đầu tiên, kết thúc ngay sau tin cuối. Các bài dưới đây xếp theo thứ tự đếm ngược: bài đầu là Top {count}, bài cuối là Top 1 (nổi bật nhất). Hệ thống tự đọc "Top <số>." trước mỗi lời, nên đừng viết lại số thứ hạng.

CẤU TRÚC MỖI LỜI (2 câu, 20-28 từ, đọc khoảng 8 giây):
- Câu 1, tin: ai/cái gì, làm gì, kèm MỘT chi tiết cụ thể có trong bài (tên sản phẩm, phiên bản, con số, mốc thời gian). Mở bằng chi tiết gây chú ý, không mở bằng "Hôm nay", "Mới đây", "Theo".
- Câu 2, ý nghĩa: điều này đổi gì cho lập trình viên (code, công cụ, chi phí, bảo mật, công việc), nói thẳng, cụ thể.
Lời của bài đầu tiên phải đủ cuốn để giữ người xem ở lại (người xem chưa biết có Top 1 phía sau). Lời của bài cuối (Top 1) kết bằng một câu hỏi ngắn mời bình luận, không chào tạm biệt, không xin theo dõi.

VĂN NÓI: đọc to lên phải trôi. Câu ngắn, từ thông dụng, giữ nguyên tên riêng và thuật ngữ kỹ thuật tiếng Anh quen thuộc (API, open source, LLM...). Không viết tắt khó đọc, không ký tự đặc biệt ($, %, /, &: viết thành chữ "đô la", "phần trăm", "trên", "và"), không dấu ngoặc, không trích dẫn nguyên văn, mỗi câu một ý và không quá 20 từ, không gạch đầu dòng, không emoji, không chữ Hán. Tránh sáo rỗng ("cuộc cách mạng", "bước ngoặt", "đừng bỏ lỡ") và câu lặp khuôn giữa các bài.

NGUYÊN TẮC TRUNG THỰC: chỉ dùng thông tin trong danh sách bài. Không thêm số liệu, tên, ngày tháng không có trong bài; không giật gân sai sự thật. Bài chưa chắc chắn thì nói "theo báo cáo", "dự kiến".

Trả về MỘT đối tượng JSON (không bọc markdown):
{{"items": [{{"id": <id bài>, "line": "<lời đọc 2 câu theo cấu trúc trên>"}}],
 "caption": "<tối đa 90 ký tự: dòng mô tả bài đăng, nêu tin đáng chú ý nhất, gợi tò mò, đúng sự thật>",
 "hashtags": ["#...", "<4-5 hashtag, trộn tiếng Việt và tiếng Anh, bám chủ đề các bài>"]}}

Có đúng {count} bài trong "items", theo đúng thứ tự đã cho, mỗi id đúng một lần.

DANH SÁCH BÀI:
{articles_json}
"""


# ---------- article selection ----------

def _as_utc(dt: datetime.datetime) -> datetime.datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)


async def used_article_ids(db, days: int = 8) -> set:
    """Ids of articles already in a video. Only recent videos matter: a video window is <= 7 days."""
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    rows = (await db.execute(select(NewsVideo.article_ids).where(NewsVideo.created_at >= since))).scalars().all()
    return {int(i) for ids in rows for i in (ids or [])}


async def pick_video_articles(db, hours: int, reuse: bool = False) -> List[Article]:
    """All distinct-story articles of the window scoring >= BRIEFING_MIN_SCORE that no earlier video
    used (unless reuse), best first, at most MAX_ARTICLES."""
    since = _local_now() - datetime.timedelta(hours=hours)
    articles = await collect_briefing_articles(db, since)
    used = set() if reuse else await used_article_ids(db)
    picked, seen = [], set()
    for a in articles:
        if a.id in used:
            continue
        key = a.cluster_topic_key or f"id-{a.id}"
        if key in seen:
            continue
        seen.add(key)
        picked.append(a)
        if len(picked) >= MAX_ARTICLES:
            break
    return picked


# ---------- script ----------

BASE_HASHTAGS = ["#congnghe", "#tech", "#laptrinh"]


def _hashtags_from(articles: List[Article]) -> List[str]:
    tags: List[str] = []
    for a in articles:
        for t in (getattr(a, "tags", None) or [])[:2]:
            tag = "#" + re.sub(r"[^0-9a-zA-Z]+", "", str(t).title().replace(" ", ""))
            if len(tag) > 2 and tag.lower() not in [x.lower() for x in tags]:
                tags.append(tag)
    return (tags[:2] + BASE_HASHTAGS)[:5]


def _fallback_script(articles: List[Article]) -> Dict[str, Any]:
    """`articles` are in countdown order (best last)."""
    n = len(articles)
    return {
        "items": [
            {"id": a.id, "line": spoken_text(short_description(a, 200)) or (a.vietnamese_title or a.title)}
            for a in articles
        ],
        "caption": f"Top {n} tin công nghệ nóng nhất hôm nay",
        "hashtags": _hashtags_from(articles),
    }


def _validate_script(data: dict, articles: List[Article]) -> Optional[Dict[str, Any]]:
    lines = {}
    for it in data.get("items") or []:
        try:
            lines[int(it.get("id"))] = str(it.get("line") or "").strip()
        except (TypeError, ValueError, AttributeError):
            continue
    items = [{"id": a.id, "line": lines[a.id]} for a in articles if lines.get(a.id)]
    if len(items) != len(articles):
        return None
    raw_tags = data.get("hashtags") if isinstance(data.get("hashtags"), list) else []
    hashtags = []
    for t in raw_tags:
        t = "#" + re.sub(r"[^0-9A-Za-zÀ-ỹ]+", "", str(t))
        if len(t) > 1 and t not in hashtags:
            hashtags.append(t)
    return {
        "items": items,
        "caption": str(data.get("caption") or f"Top {len(articles)} tin công nghệ hôm nay").strip()[:120],
        "hashtags": (hashtags or _hashtags_from(articles))[:5],
    }


async def generate_script(articles: List[Article]) -> Tuple[Dict[str, Any], str]:
    payload = [
        {
            "id": a.id,
            "title": a.vietnamese_title or a.title,
            "summary": (a.vietnamese_summary or "")[:300],
            "takeaways": (a.key_takeaways or [])[:2],
        }
        for a in articles
    ]
    prompt = SCRIPT_PROMPT.format(
        count=len(articles),
        seconds=SECONDS_PER_ARTICLE * len(articles),
        articles_json=json.dumps(payload, ensure_ascii=False),
    )
    client = get_ai_client(settings.AI_TIMEOUT_BACKGROUND)
    for model_name in settings.fallback_models_list:
        if not ai_circuit_breaker.can_execute(model_name):
            continue
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            script = _validate_script(_clean_json_response(message_text(response)), articles)
            if script is None:
                raise ValueError("script JSON incomplete")
            ai_circuit_breaker.record_success(model_name)
            return script, model_name
        except Exception as err:
            record_ai_failure(model_name, err)
            logger.warning(f"Video script: model {model_name} failed ({err})")
    return _fallback_script(articles), "fallback"


def _topic_chips(a: Article) -> List[str]:
    tags = [str(t).strip() for t in (getattr(a, "tags", None) or []) if str(t).strip()]
    if not tags:
        tags = [str(t.get("name", "")).strip() for t in (getattr(a, "new_tech_stacks", None) or []) if isinstance(t, dict)]
    return [t[:18] for t in tags if t][:3]


def build_segments(script: Dict[str, Any], articles: List[Article]) -> List[Dict[str, Any]]:
    """`articles` in countdown order: the first is ranked Top n, the last Top 1."""
    by_id = {a.id: a for a in articles}
    n = len(articles)
    segs: List[Dict[str, Any]] = []
    for i, it in enumerate(script["items"]):
        a = by_id[it["id"]]
        rank = n - i
        segs.append(
            {
                "kind": "item",
                "rank": rank,
                "title": a.vietnamese_title or a.title,
                "text": f"Top {rank}. {spoken_text(it['line'])}",
                "meta": a.source.name if a.source else None,
                "tags": _topic_chips(a),
                "url": a.url,
                "image": None,
            }
        )
    return segs


# ---------- voice ----------

def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_duration(path: Path) -> float:
    out = subprocess.run([_ffmpeg(), "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if not m:
        raise RuntimeError(f"cannot read duration of {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def spoken_text(text: str) -> str:
    """Strips the extractive-fallback prefix and trailing ellipses so the voice reads clean prose."""
    text = re.sub(r"^\s*Trích xuất tự động:\s*", "", text)
    text = re.sub(r"(\.{2,}|…)+\s*$", ".", text.strip())
    return " ".join(text.split())


TTS_MAX_ATTEMPTS = 4
GTTS_ATTEMPTS = 3
GTTS_TEMPO = 1.08  # gTTS has no rate option; speed it up to match edge-tts' TTS_RATE (+8%)
TTS_CACHE_MAX_AGE_SECONDS = 24 * 3600


def _cache_dir() -> Path:
    d = Path(settings.VIDEO_OUTPUT_DIR) / ".tts_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def prune_cache(max_age: float = TTS_CACHE_MAX_AGE_SECONDS) -> None:
    """Drops cached audio/scripts older than max_age so an abandoned build doesn't leave files forever."""
    try:
        cutoff = time.time() - max_age
        for f in _cache_dir().iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not prune the TTS cache", exc_info=True)


def _gtts_save(text: str, out_path: Path) -> None:
    """gTTS voice sped up to roughly edge-tts' pace."""
    from gtts import gTTS

    raw = out_path.with_suffix(".raw.mp3")
    try:
        gTTS(text=text, lang="vi").save(str(raw))
        r = subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(raw), "-filter:a", f"atempo={GTTS_TEMPO}", str(out_path)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg atempo failed: {r.stderr[-200:]}")
    finally:
        raw.unlink(missing_ok=True)


async def _synthesize_edge(text: str, out_path: Path) -> None:
    """edge-tts is a free, flaky endpoint (NoAudioReceived at random). Retries with jittered backoff, always
    with the same voice: a video must not mix voices."""
    voice = settings.VIDEO_TTS_VOICE
    last: Exception = RuntimeError("no attempt")
    for attempt in range(TTS_MAX_ATTEMPTS):
        try:
            await edge_tts.Communicate(text, voice, rate=TTS_RATE).save(str(out_path))
            return
        except Exception as err:
            last = err
            logger.warning(f"TTS attempt {attempt + 1} ({voice}) failed: {type(err).__name__}")
            if attempt < TTS_MAX_ATTEMPTS - 1:
                await asyncio.sleep(min(3.0 * (attempt + 1), 20.0) + random.uniform(0, 3.0))
    raise RuntimeError(f"edge-tts failed after {TTS_MAX_ATTEMPTS} attempts: {last}")


async def _synthesize_gtts(text: str, out_path: Path) -> None:
    last: Exception = RuntimeError("no attempt")
    for attempt in range(GTTS_ATTEMPTS):
        try:
            await asyncio.to_thread(_gtts_save, text, out_path)
            return
        except Exception as err:
            last = err
            logger.warning(f"gTTS attempt {attempt + 1} failed: {type(err).__name__}")
            if attempt < GTTS_ATTEMPTS - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"gTTS failed after {GTTS_ATTEMPTS} attempts: {last}")


async def synthesize(text: str, out_path: Path, engine: str = "edge") -> float:
    """Voices one segment with ONE engine ("edge" or "gtts"). Finished audio is cached by engine, voice and
    text, so a retried build (new temp dir, same script) only reads the segments still missing."""
    text = spoken_text(text)
    voice = settings.VIDEO_TTS_VOICE if engine == "edge" else "gtts"
    cached = _cache_dir() / f"{_cache_key(engine, voice, text)}.mp3"
    if cached.exists() and cached.stat().st_size > 0:
        try:
            duration = audio_duration(cached)
            out_path.write_bytes(cached.read_bytes())
            return duration
        except Exception:
            cached.unlink(missing_ok=True)

    tmp = out_path.with_suffix(".part.mp3")
    try:
        await (_synthesize_edge if engine == "edge" else _synthesize_gtts)(text, tmp)
        duration = audio_duration(tmp)
        cached.write_bytes(tmp.read_bytes())
        tmp.replace(out_path)
        return duration
    finally:
        tmp.unlink(missing_ok=True)


async def synthesize_all(
    texts: List[str], tmp_dir: Path, on_step: Optional[Callable[[str], None]] = None
) -> List[float]:
    """Voices every segment with the same engine and voice. edge-tts first; if any segment still fails
    after its retries, the whole video is voiced again with gTTS rather than mixing two voices."""
    step = on_step or (lambda _msg: None)
    engines = ["edge", "gtts"] if settings.VIDEO_TTS_FALLBACK_GTTS else ["edge"]
    for n, engine in enumerate(engines):
        try:
            durations = []
            for i, text in enumerate(texts):  # sequential: edge-tts throttles parallel calls
                step(f"Đọc giọng ({i + 1}/{len(texts)})")
                durations.append(await synthesize(text, tmp_dir / f"a{i}.mp3", engine))
            return durations
        except Exception as err:
            if n == len(engines) - 1:
                raise
            logger.warning(f"{err}; voicing the whole video with gTTS so the voice stays the same")
            step("Đọc lại toàn bộ bằng giọng dự phòng")
    raise RuntimeError("no TTS engine")


# ---------- assemble ----------

def _encode_scene(scene: Scene, audio: Path, duration: float, out: Path) -> None:
    """Pipes the scene's frames into ffmpeg together with its voice track."""
    fps = FPS
    frames = max(1, round(duration * fps))
    log = out.with_suffix(".log")
    with open(log, "wb") as errf:
        proc = subprocess.Popen(
            [
                _ffmpeg(), "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
                "-i", str(audio),
                "-f", "lavfi", "-i", "anoisesrc=d=0.35:c=pink:a=0.5",
                "-filter_complex",
                "[1:a]apad[v];[2:a]highpass=f=1500,afade=t=in:d=0.1,afade=t=out:st=0.12:d=0.23,volume=0.35[w];"
                "[v][w]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "0:v", "-map", "[a]", "-t", f"{frames / fps:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100", "-ac", "2", str(out),
            ],
            stdin=subprocess.PIPE, stderr=errf,
        )
        try:
            for n in range(frames):
                proc.stdin.write(scene.frame(n / fps).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {log.read_text(errors='ignore')[-400:]}")


def concat_clips(clips: List[Path], out: Path) -> None:
    listing = out.parent / f"{out.stem}.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    try:
        r = subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-movflags", "+faststart", str(out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-400:]}")
    finally:
        listing.unlink(missing_ok=True)


async def render_video(
    segments: List[Dict[str, Any]], out_path: Path, on_step: Optional[Callable[[str], None]] = None
) -> float:
    """Renders segments to an mp4; returns total duration in seconds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tail = 0.5  # breathing room after each voice line

    step = on_step or (lambda _msg: None)
    if settings.VIDEO_USE_ARTICLE_IMAGES:
        step("Tải ảnh bài viết")
        items = [s for s in segments if s["kind"] == "item"]
        images = await asyncio.gather(*[fetch_article_image(s.get("url", "")) for s in items])
        for seg, img in zip(items, images):
            seg["image"] = img
        logger.info(f"Article images: {sum(i is not None for i in images)}/{len(items)} found")

    with tempfile.TemporaryDirectory(prefix="techpulse-video-") as tmp:
        tmp_dir = Path(tmp)
        for seg in segments:
            seg["text"] = spoken_text(seg["text"])
        speech = await synthesize_all([seg["text"] for seg in segments], tmp_dir, on_step)
        durations = [d + tail for d in speech]
        total = sum(durations)

        scenes, start = [], 0.0
        for i, seg in enumerate(segments):
            sc = Scene(seg, i, len(segments), speech[i], durations[i])
            sc.global_start, sc.global_total = start, total
            scenes.append(sc)
            start += durations[i]

        clips = []
        for i, sc in enumerate(scenes):
            step(f"Dựng video ({i + 1}/{len(scenes)})")
            clip = tmp_dir / f"s{i}.mp4"
            await asyncio.to_thread(_encode_scene, sc, tmp_dir / f"a{i}.mp3", durations[i], clip)
            clips.append(clip)
        await asyncio.to_thread(concat_clips, clips, out_path)
    return total


async def _cached_script(countdown: List[Article]) -> Tuple[Dict[str, Any], str]:
    """The LLM script for this exact article set, kept for TTS_CACHE_MAX_AGE_SECONDS: a retried build must
    say the same words, or its cached audio would never match."""
    prune_cache()
    path = _cache_dir() / f"script-{_cache_key(SCRIPT_PROMPT, *[str(a.id) for a in countdown])}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["script"], data["model"]
    except (OSError, ValueError, KeyError):
        pass
    script, model = await generate_script(countdown)
    try:
        path.write_text(json.dumps({"script": script, "model": model}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.warning("Could not cache the video script", exc_info=True)
    return script, model


def _not_enough_message(found: int) -> str:
    return f"Chỉ có {found} bài từ {settings.BRIEFING_MIN_SCORE:g} điểm chưa lên video (cần ít nhất {MIN_ARTICLES})."


def fit_caption(header: List[str], story_lines: List[str], budget: int = 1000) -> str:
    """Joins whole lines until the Telegram caption budget (1024 chars, markup included) is used.
    Cutting the text instead could split an <a> tag and make Telegram reject the video."""
    out = list(header)
    used = sum(len(x) + 1 for x in out)
    shown = 0
    for line in story_lines:
        if used + len(line) + 1 > budget - 40:  # keep room for the "and N more" line
            break
        out.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(story_lines):
        out.append(f"… và {len(story_lines) - shown} bài khác (xem trong giao diện)")
    return "\n".join(out)


async def send_video_to_telegram(
    video: Path, caption: str, countdown: List[Article], duration: float
) -> bool:
    """Sends the video with the post caption to copy, plus the stories (with links) behind it."""
    from app.services.telegram_service import escape_html, send_telegram_video

    n = len(countdown)
    header = [f"<b>{escape_html(caption)}</b>", ""]
    story_lines = []
    for i, a in enumerate(countdown):
        source = f" ({escape_html(a.source.name)})" if getattr(a, "source", None) else ""
        story_lines.append(f'Top {n - i}: <a href="{escape_html(a.url)}">{escape_html(a.vietnamese_title or a.title)}</a>{source}')
    sent = await send_telegram_video(video, fit_caption(header, story_lines), duration=duration)
    if not sent:
        logger.warning(f"Could not send {video.name} to Telegram; the file is still in {video.parent}")
    return sent


async def generate_news_video(
    db,
    hours: int = 24,
    send_telegram: bool = True,
    reuse: bool = False,
    on_step: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Builds, records and (optionally) sends one video. Articles used by an earlier video are
    skipped unless reuse=True; with fewer than MIN_ARTICLES fresh articles nothing is made."""
    if video_lock.locked():
        return {"success": False, "skipped": "busy", "message": "Đang tạo một video khác, thử lại sau."}
    step = on_step or (lambda _msg: None)
    async with video_lock:
        step("Chọn bài")
        articles = await pick_video_articles(db, hours, reuse=reuse)
        if len(articles) < MIN_ARTICLES:
            return {
                "success": False,
                "skipped": "not_enough_new_articles",
                "message": _not_enough_message(len(articles)),
            }
        countdown = list(reversed(articles))  # best story last: Top 3 -> Top 1
        step("Viết kịch bản")
        script, model = await _cached_script(countdown)
        segments = build_segments(script, countdown)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path(settings.VIDEO_OUTPUT_DIR) / f"techpulse-{stamp}.mp4"
        duration = await render_video(segments, out, on_step=on_step)
        caption = f"{script['caption']} {' '.join(script['hashtags'])}".strip()
        out.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

        # Record before sending: a Telegram failure must not make the next run rebuild this video
        record = NewsVideo(
            file=out.name, article_ids=[a.id for a in countdown], caption=caption, duration_sec=round(duration, 1)
        )
        db.add(record)
        await db.commit()

        if send_telegram and settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
            step("Gửi Telegram")
            record.telegram_sent = await send_video_to_telegram(out, caption, countdown, duration)
            await db.commit()
        return {
            "success": True,
            "file": out.name,
            "duration_sec": round(duration, 1),
            "script_model": model,
            "article_ids": record.article_ids,
            "caption": caption,
            "telegram_sent": record.telegram_sent,
            "segments": [{"title": s["title"], "text": s["text"]} for s in segments],
        }


# ---------- background job (started from the UI) ----------

class VideoJob:
    """State of the one video build that can run at a time. In memory: a restart resets it to idle;
    finished videos live in the news_videos table."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.status = "idle"  # idle | running | done | skipped | failed
        self.step = ""
        self.started_at: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        elapsed = round(time.time() - self.started_at, 1) if self.started_at else None
        return {
            "status": self.status,
            "step": self.step,
            "elapsed_sec": elapsed,
            "result": self.result,
            "error": self.error,
        }


video_job = VideoJob()
_background_tasks: set = set()


VIDEO_RETRY_DELAY_SECONDS = 20 * 60
VIDEO_AUTO_RETRIES = 2


async def _notify_video_failure(error: str, retries_left: int) -> None:
    from app.services.telegram_service import escape_html, send_telegram_message

    tail = (
        f"Sẽ tự thử lại sau {VIDEO_RETRY_DELAY_SECONDS // 60} phút."
        if retries_left > 0
        else "Đã hết lượt thử lại tự động."
    )
    try:
        await send_telegram_message(
            f"⚠️ <b>Video tổng hợp thất bại</b>\n<code>{escape_html(error)}</code>\n{tail}",
            disable_preview=True,
        )
    except Exception:
        logger.exception("Could not send video failure notice")


async def _retry_later(hours: int, send_telegram: bool, reuse: bool, retries_left: int) -> None:
    await asyncio.sleep(VIDEO_RETRY_DELAY_SECONDS)
    await start_video_job(hours, send_telegram, reuse, retries_left=retries_left)


async def _run_job(hours: int, send_telegram: bool, reuse: bool, retries_left: int = 0) -> None:
    from app.core import database

    try:
        async with database.AsyncSessionLocal() as db:
            result = await generate_news_video(
                db, hours=hours, send_telegram=send_telegram, reuse=reuse,
                on_step=lambda msg: setattr(video_job, "step", msg),
            )
        video_job.result = result
        video_job.status = "done" if result.get("success") else "skipped"
    except Exception as err:
        logger.exception("Video job failed")
        video_job.error = f"{type(err).__name__}: {err}"[:300]
        video_job.status = "failed"
        if send_telegram:
            await _notify_video_failure(video_job.error, retries_left)
        if retries_left > 0:
            _spawn(_retry_later(hours, send_telegram, reuse, retries_left - 1))
    finally:
        video_job.step = ""


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)  # keep a reference so the task isn't garbage-collected mid-run
    task.add_done_callback(_background_tasks.discard)


async def start_video_job(
    hours: int = 24, send_telegram: bool = True, reuse: bool = False, retries_left: int = 0
) -> Dict[str, Any]:
    """Starts a build in the background. Returns at once: {started: True}, or {started: False, skipped, message}
    when one is already running or there are not enough unused articles."""
    from app.core import database

    if video_lock.locked() or video_job.status == "running":
        return {"started": False, "skipped": "busy", "message": "Đang tạo một video khác."}
    async with database.AsyncSessionLocal() as db:
        fresh = await pick_video_articles(db, hours, reuse=reuse)
    if len(fresh) < MIN_ARTICLES:
        return {"started": False, "skipped": "not_enough_new_articles", "message": _not_enough_message(len(fresh))}

    video_job.reset()
    video_job.status, video_job.step, video_job.started_at = "running", "Chọn bài", time.time()
    _spawn(_run_job(hours, send_telegram, reuse, retries_left))
    return {"started": True}
