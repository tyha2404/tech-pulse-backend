"""Short vertical news video (TikTok/Reels/Shorts) built from the best recent articles.

Pipeline: pick articles -> LLM script (fallback: summaries) -> edge-tts voice per segment
-> Pillow card per segment (1080x1920) -> ffmpeg clips -> concat to one mp4.
Cards are drawn by us, never the source article's images (copyright)."""
import asyncio
import datetime
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
import imageio_ffmpeg

from app.core.config import settings
from app.models.models import Article
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

SCRIPT_PROMPT = """Bạn viết kịch bản video dọc 25-35 giây (TikTok) "Top {count} tin công nghệ hôm nay" cho lập trình viên Việt Nam. Các bài dưới đây xếp theo thứ tự đếm ngược: bài đầu là Top {count}, bài cuối là Top 1 (nổi bật nhất).

NGUYÊN TẮC: chỉ dùng thông tin trong danh sách bài. Không thêm số liệu, tên, ngày tháng không có trong bài. Không bịa, không giật gân sai sự thật, không emoji, không chữ Hán. Văn nói tự nhiên, câu ngắn.

Trả về MỘT đối tượng JSON (không bọc markdown):
{{"hook": "<8-12 từ, nhá hàng điều bất ngờ hoặc quan trọng nhất của bài Top 1 bằng một chi tiết cụ thể có trong bài; không tiết lộ hết; có thể là câu hỏi>",
 "items": [{{"id": <id bài>, "line": "<1-2 câu, 14-22 từ: chuyện gì xảy ra và vì sao lập trình viên nên quan tâm>"}}],
 "cta": "<8-14 từ: một câu hỏi để người xem bình luận về các tin trên, rồi mời theo dõi>",
 "caption": "<tối đa 90 ký tự: dòng mô tả bài đăng, gợi tò mò, đúng sự thật>",
 "hashtags": ["#...", "<4-5 hashtag, trộn tiếng Việt và tiếng Anh, bám chủ đề các bài>"]}}

Có đúng {count} bài trong "items", theo đúng thứ tự đã cho, mỗi id đúng một lần.

DANH SÁCH BÀI:
{articles_json}
"""


# ---------- article selection ----------

async def pick_video_articles(db, hours: int, limit: int) -> List[Article]:
    since = _local_now() - datetime.timedelta(hours=hours)
    articles = await collect_briefing_articles(db, since)
    picked, seen = [], set()
    for a in articles:
        key = a.cluster_topic_key or f"id-{a.id}"
        if key in seen:
            continue
        seen.add(key)
        picked.append(a)
        if len(picked) >= limit:
            break
    return picked


# ---------- script ----------

DEFAULT_CTA = "Bạn quan tâm tin nào nhất? Bình luận và theo dõi để nhận tin mỗi ngày."
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
        "hook": f"Top {n} tin công nghệ nóng nhất hôm nay",
        "items": [
            {"id": a.id, "line": spoken_text(short_description(a, 200)) or (a.vietnamese_title or a.title)}
            for a in articles
        ],
        "cta": DEFAULT_CTA,
        "caption": f"Top {n} tin công nghệ nóng nhất hôm nay",
        "hashtags": _hashtags_from(articles),
    }


def _validate_script(data: dict, articles: List[Article]) -> Optional[Dict[str, Any]]:
    hook = str(data.get("hook") or "").strip()
    cta = str(data.get("cta") or "").strip()
    lines = {}
    for it in data.get("items") or []:
        try:
            lines[int(it.get("id"))] = str(it.get("line") or "").strip()
        except (TypeError, ValueError, AttributeError):
            continue
    items = [{"id": a.id, "line": lines[a.id]} for a in articles if lines.get(a.id)]
    if not hook or len(items) != len(articles):
        return None
    raw_tags = data.get("hashtags") if isinstance(data.get("hashtags"), list) else []
    hashtags = []
    for t in raw_tags:
        t = "#" + re.sub(r"[^0-9A-Za-zÀ-ỹ]+", "", str(t))
        if len(t) > 1 and t not in hashtags:
            hashtags.append(t)
    return {
        "hook": hook,
        "items": items,
        "cta": cta or DEFAULT_CTA,
        "caption": str(data.get("caption") or hook).strip()[:120],
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
        count=len(articles), articles_json=json.dumps(payload, ensure_ascii=False)
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


def build_segments(script: Dict[str, Any], articles: List[Article]) -> List[Dict[str, Any]]:
    """`articles` in countdown order: the first is ranked Top n, the last Top 1."""
    by_id = {a.id: a for a in articles}
    n = len(articles)
    segs: List[Dict[str, Any]] = [
        {"kind": "intro", "title": "", "text": script["hook"], "score": None, "meta": None, "image": None}
    ]
    for i, it in enumerate(script["items"]):
        a = by_id[it["id"]]
        segs.append(
            {
                "kind": "item",
                "rank": n - i,
                "title": a.vietnamese_title or a.title,
                "text": it["line"],
                "score": round(a.relevance_score, 1) if a.relevance_score else None,
                "meta": a.source.name if a.source else None,
                "url": a.url,
                "image": None,
            }
        )
    segs.append({"kind": "outro", "title": "", "text": script["cta"], "score": None, "meta": None, "image": None})
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


async def synthesize(text: str, out_path: Path) -> float:
    text = spoken_text(text)
    last: Exception = RuntimeError("no attempt")
    for attempt in range(5):  # the free edge-tts endpoint sometimes returns no audio
        try:
            await edge_tts.Communicate(
                text, settings.VIDEO_TTS_VOICE, rate=TTS_RATE
            ).save(str(out_path))
            return audio_duration(out_path)
        except Exception as err:
            last = err
            logger.warning(f"TTS attempt {attempt + 1} failed: {type(err).__name__}")
            await asyncio.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"TTS failed after 5 attempts: {last}")


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


async def render_video(segments: List[Dict[str, Any]], out_path: Path) -> float:
    """Renders segments to an mp4; returns total duration in seconds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tail = 0.5  # breathing room after each voice line

    if settings.VIDEO_USE_ARTICLE_IMAGES:
        items = [s for s in segments if s["kind"] == "item"]
        images = await asyncio.gather(*[fetch_article_image(s.get("url", "")) for s in items])
        for seg, img in zip(items, images):
            seg["image"] = img
        logger.info(f"Article images: {sum(i is not None for i in images)}/{len(items)} found")
        intro = next((x for x in segments if x["kind"] == "intro"), None)
        if intro is not None and items:
            intro["image"] = items[-1]["image"]  # hook plays over the Top 1 story's image

    with tempfile.TemporaryDirectory(prefix="techpulse-video-") as tmp:
        tmp_dir = Path(tmp)
        speech: List[float] = []
        for i, seg in enumerate(segments):  # sequential: edge-tts throttles parallel calls
            seg["text"] = spoken_text(seg["text"])
            speech.append(await synthesize(seg["text"], tmp_dir / f"a{i}.mp3"))
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
            clip = tmp_dir / f"s{i}.mp4"
            await asyncio.to_thread(_encode_scene, sc, tmp_dir / f"a{i}.mp3", durations[i], clip)
            clips.append(clip)
        await asyncio.to_thread(concat_clips, clips, out_path)
    return total


async def send_video_to_telegram(
    video: Path, caption: str, countdown: List[Article], duration: float
) -> bool:
    """Sends the video with the post caption to copy, plus the stories (with links) behind it."""
    from app.services.telegram_service import escape_html, send_telegram_video

    n = len(countdown)
    lines = [f"<b>{escape_html(caption)}</b>", ""]
    for i, a in enumerate(countdown):
        source = f" ({escape_html(a.source.name)})" if getattr(a, "source", None) else ""
        lines.append(f'Top {n - i}: <a href="{escape_html(a.url)}">{escape_html(a.vietnamese_title or a.title)}</a>{source}')
    sent = await send_telegram_video(video, "\n".join(lines), duration=duration)
    if not sent:
        logger.warning(f"Could not send {video.name} to Telegram; the file is still in {video.parent}")
    return sent


async def generate_news_video(
    db, hours: int = 24, limit: int = 3, send_telegram: bool = True
) -> Dict[str, Any]:
    articles = await pick_video_articles(db, hours, limit)
    if not articles:
        return {"success": False, "message": "Không có bài đủ điểm trong khung giờ này."}
    countdown = list(reversed(articles))  # best story last: Top 3 -> Top 1
    script, model = await generate_script(countdown)
    segments = build_segments(script, countdown)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(settings.VIDEO_OUTPUT_DIR) / f"techpulse-{stamp}.mp4"
    duration = await render_video(segments, out)
    caption = f"{script['caption']} {' '.join(script['hashtags'])}".strip()
    out.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

    telegram_sent: Optional[bool] = None
    if send_telegram and settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        telegram_sent = await send_video_to_telegram(out, caption, countdown, duration)
    return {
        "success": True,
        "file": out.name,
        "duration_sec": round(duration, 1),
        "script_model": model,
        "article_ids": [a.id for a in countdown],
        "caption": caption,
        "telegram_sent": telegram_sent,
        "segments": [{"title": s["title"], "text": s["text"]} for s in segments],
    }
