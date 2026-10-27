"""Telegram briefings that turn many articles into a few themes and a short must-read list.

Two kinds run on a schedule (BRIEFING_TIMEZONE):
- short updates at BRIEFING_UPDATE_HOURS (default 9, 12, 18), each covering only the
  articles added since the previous scheduled briefing, skipped when there is nothing new;
- the end-of-day "super summary" at DAILY_BRIEFING_HOUR (default 22) covering 24 hours.
"""

import datetime
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.models import Article
from app.services.ai_analyzer import (
    EXTRACTIVE_FALLBACK_MODEL,
    _clean_json_response,
    get_ai_client,
    message_text,
)
from app.services.circuit_breaker import ai_circuit_breaker
from app.services.telegram_service import (
    FRONTEND_URL,
    escape_html,
    score_dot,
    send_telegram_message,
)

logger = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096
KEYCAPS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
WEEKDAYS = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"]


@dataclass(frozen=True)
class BriefingKind:
    key: str
    title: str
    emoji: str
    max_themes: int
    max_must_read: int
    # How many articles the LLM sees; the rest are only counted
    max_articles_for_ai: int
    # Describes the briefing to the LLM
    scope: str
    # Super summary also asks for the day's single biggest trend
    with_trend: bool
    # Updates with nothing new are skipped instead of sending an empty message
    send_when_empty: bool


UPDATE = BriefingKind(
    key="update",
    title="Cập nhật",
    emoji="📰",
    max_themes=3,
    max_must_read=2,
    max_articles_for_ai=20,
    scope="bản cập nhật nhanh giữa ngày, chỉ gồm các bài mới kể từ lần cập nhật trước, đọc trong 1-2 phút",
    with_trend=False,
    send_when_empty=False,
)

DAILY = BriefingKind(
    key="daily",
    title="Siêu tổng hợp cuối ngày",
    emoji="🌙",
    max_themes=5,
    max_must_read=5,
    max_articles_for_ai=40,
    scope="bản siêu tổng hợp cuối ngày cho toàn bộ 24 giờ qua, đọc trong 5 phút",
    with_trend=True,
    send_when_empty=True,
)

BRIEFING_PROMPT = """Bạn là biên tập viên công nghệ viết {scope} cho một kỹ sư Backend & AI.
Dưới đây là các bài đã được chấm điểm (JSON). Hãy:
1. Gom các bài thành tối đa {max_themes} chủ đề. Mỗi chủ đề viết 2-3 câu tổng hợp ý chung của các bài
   (không liệt kê lại từng bài), nêu rõ điều gì thay đổi và vì sao kỹ sư nên quan tâm.
   Bỏ qua bài lẻ tẻ, ít giá trị; không cần chủ đề nào cũng có nhiều bài.
2. Chọn tối đa {max_must_read} bài đáng đọc kỹ nhất, mỗi bài một câu giải thích lý do cụ thể
   và một tiêu đề tiếng Việt ("title"): dịch tiêu đề gốc nếu nó đang là tiếng Anh hoặc ngôn ngữ khác.
3. Viết một câu mở đầu tóm tắt nội dung chính.{trend_instruction}

Toàn bộ bằng tiếng Việt tự nhiên, ngắn gọn, không dùng ký tự Trung/Nhật; dữ liệu đầu vào có thể còn
tiếng Anh, hãy dịch sang tiếng Việt (giữ nguyên tên riêng, tên sản phẩm và thuật ngữ kỹ thuật thông dụng).
Chỉ dùng id có trong dữ liệu.
Chỉ trả về JSON đúng dạng:
{{"headline": "...",{trend_field} "themes": [{{"title": "...", "summary": "...", "article_ids": [1, 2]}}],
  "must_read": [{{"id": 1, "title": "...", "why": "..."}}]}}

DỮ LIỆU:
{articles_json}"""

TREND_INSTRUCTION = """
4. Viết "trend": 2-3 câu về xu hướng lớn nhất xuyên suốt cả ngày (nối nhiều chủ đề nếu có),
   và một gợi ý hành động cụ thể cho kỹ sư Backend."""


def _article_payload(a: Article) -> Dict[str, Any]:
    return {
        "id": a.id,
        "title": a.vietnamese_title or a.title,
        "score": round(a.relevance_score or 0, 1),
        "source": a.source.name if a.source else None,
        "summary": (a.vietnamese_summary or "")[:300],
        "takeaways": (a.key_takeaways or [])[:2],
        "tags": (a.tags or [])[:4],
    }


def _local_now() -> datetime.datetime:
    return datetime.datetime.now(ZoneInfo(settings.BRIEFING_TIMEZONE))


def schedule_hours() -> List[int]:
    """All scheduled briefing hours (updates + super summary), sorted."""
    return sorted(set(settings.briefing_update_hours_list + [settings.DAILY_BRIEFING_HOUR]))


def _slot_minute(hour: int) -> int:
    # Updates fire on the hour; the super summary at DAILY_BRIEFING_MINUTE
    return settings.DAILY_BRIEFING_MINUTE if hour == settings.DAILY_BRIEFING_HOUR else 0


def previous_slot_start(now: datetime.datetime, hours: List[int]) -> datetime.datetime:
    """Start of the window for a briefing sent at `now`: the latest scheduled slot before
    now's hour, wrapping to yesterday's last slot (9:00 covers since 22:00 yesterday)."""
    earlier = [h for h in hours if h < now.hour]
    day, hour = (now, earlier[-1]) if earlier else (now - datetime.timedelta(days=1), hours[-1])
    return day.replace(hour=hour, minute=_slot_minute(hour), second=0, microsecond=0)


async def collect_briefing_articles(db, since: datetime.datetime) -> List[Article]:
    """AI-analyzed, canonical, visible articles created since `since`, best first."""
    res = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(
            Article.created_at >= since,
            Article.is_processed == True,
            Article.is_canonical == True,
            Article.is_hidden == False,
        )
        .order_by(Article.relevance_score.desc(), Article.id.desc())
    )
    return list(res.scalars().all())


def _validate_briefing(
    data: Dict[str, Any], by_id: Dict[int, Article], kind: BriefingKind = DAILY
) -> Optional[Dict[str, Any]]:
    """Keep only well-formed themes / picks that reference real article ids."""

    def real_ids(ids) -> List[int]:
        out = []
        for i in ids or []:
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if i in by_id and i not in out:
                out.append(i)
        return out

    themes = []
    for t in data.get("themes") or []:
        if not isinstance(t, dict) or not t.get("title") or not t.get("summary"):
            continue
        ids = real_ids(t.get("article_ids"))
        if ids:
            themes.append({"title": str(t["title"]), "summary": str(t["summary"]), "article_ids": ids[:3]})

    must_read = []
    for m in data.get("must_read") or []:
        if not isinstance(m, dict):
            continue
        ids = real_ids([m.get("id")])
        if ids and ids[0] not in [x["id"] for x in must_read]:
            must_read.append(
                {"id": ids[0], "title": str(m.get("title") or ""), "why": str(m.get("why") or "")}
            )

    if not themes and not must_read:
        return None
    return {
        "headline": str(data.get("headline") or ""),
        "trend": str(data.get("trend") or "") if kind.with_trend else "",
        "themes": themes[: kind.max_themes],
        "must_read": must_read[: kind.max_must_read],
    }


def fallback_briefing(articles: List[Article], kind: BriefingKind = DAILY) -> Dict[str, Any]:
    """No-LLM briefing: group the top articles by their most common tags.
    Articles the crawler couldn't run through an LLM are left out (only counted):
    their titles were never translated and nothing here can translate them."""
    top = [a for a in articles if a.ai_model_used != EXTRACTIVE_FALLBACK_MODEL][:15]
    tag_counts = Counter(t for a in top for t in (a.tags or [])[:2])
    themes = []
    used: set = set()
    for tag, _ in tag_counts.most_common():
        ids = [a.id for a in top if tag in (a.tags or []) and a.id not in used][:3]
        if not ids:
            continue
        used.update(ids)
        themes.append({"title": tag, "summary": "", "article_ids": ids})
        if len(themes) == kind.max_themes:
            break
    must_read = [
        {"id": a.id, "why": (a.key_takeaways or [""])[0] or ""}
        for a in top[: kind.max_must_read]
    ]
    high = sum(1 for a in articles if (a.relevance_score or 0) >= 8)
    return {
        "headline": f"{len(articles)} bài mới, {high} bài từ 8 điểm. (Tóm tắt tự động, AI không phản hồi.)",
        "trend": "",
        "themes": themes,
        "must_read": must_read,
    }


async def generate_daily_briefing(
    articles: List[Article], kind: BriefingKind = DAILY
) -> Tuple[Dict[str, Any], str]:
    """Returns (briefing, model_used). Walks the model fallback chain, then falls back to tags."""
    by_id = {a.id: a for a in articles}
    prompt = BRIEFING_PROMPT.format(
        scope=kind.scope,
        max_themes=kind.max_themes,
        max_must_read=kind.max_must_read,
        trend_instruction=TREND_INSTRUCTION if kind.with_trend else "",
        trend_field=' "trend": "...",' if kind.with_trend else "",
        articles_json=json.dumps(
            [_article_payload(a) for a in articles[: kind.max_articles_for_ai]], ensure_ascii=False
        ),
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
            data = _clean_json_response(message_text(response))
            briefing = _validate_briefing(data, by_id, kind)
            if briefing is None:
                raise ValueError("briefing JSON had no usable themes or picks")
            ai_circuit_breaker.record_success(model_name)
            return briefing, model_name
        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            logger.warning(f"{kind.title}: model {model_name} failed ({err})")

    return fallback_briefing(articles, kind), "fallback"


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _header_title(kind: BriefingKind, now: datetime.datetime) -> str:
    date_label = f"{WEEKDAYS[now.weekday()]} {now.day}/{now.month}"
    if kind.key == "update":
        return f"{kind.emoji} <b>{kind.title} {now.hour:02d}:{now.minute:02d} · {date_label}</b>"
    return f"{kind.emoji} <b>{kind.title} · {date_label}</b>"


def format_daily_briefing_message(
    briefing: Dict[str, Any],
    articles: List[Article],
    now: Optional[datetime.datetime] = None,
    kind: BriefingKind = DAILY,
    since: Optional[datetime.datetime] = None,
) -> Tuple[str, Dict[str, Any]]:
    by_id = {a.id: a for a in articles}
    now = now or _local_now()

    def source_link(a: Article) -> str:
        name = escape_html(a.source.name if a.source else "Nguồn")
        return f'<a href="{escape_html(a.url)}">{name}</a>'

    def build(themes: List[Dict[str, Any]], summary_limit: int) -> str:
        body: List[str] = []
        if briefing.get("headline"):
            body += [f"<b>{escape_html(briefing['headline'])}</b>", ""]

        if briefing.get("trend"):
            body += ["<b>XU HƯỚNG CỦA NGÀY</b>", f"<blockquote>{escape_html(briefing['trend'])}</blockquote>"]

        if themes:
            body.append("<b>HÔM NAY CÓ GÌ</b>" if kind.key == "daily" else "<b>CÓ GÌ MỚI</b>")
            # One quote block per theme so each reads as its own card
            for i, t in enumerate(themes, 1):
                num = KEYCAPS[i - 1] if i <= len(KEYCAPS) else f"{i}."
                card = [f"{num} <b>{escape_html(t['title'])}</b>"]
                if t.get("summary"):
                    summary = t["summary"]
                    if len(summary) > summary_limit:
                        summary = summary[: summary_limit - 1].rstrip() + "…"
                    card.append(escape_html(summary))
                    links = [source_link(by_id[aid]) for aid in t["article_ids"] if aid in by_id]
                    if links:
                        card.append("↳ " + " · ".join(links))
                else:
                    # Fallback themes have no prose: list titles instead
                    for aid in t["article_ids"]:
                        a = by_id[aid]
                        card.append(f"• {escape_html(a.vietnamese_title or a.title)} ({source_link(a)})")
                body.append("<blockquote>" + "\n".join(card) + "</blockquote>")

        if briefing.get("must_read"):
            body.append("<b>ĐÁNG ĐỌC KỸ</b>")
            for m in briefing["must_read"]:
                a = by_id.get(m["id"])
                if not a:
                    continue
                score = a.relevance_score or 0
                # The LLM's translated title wins over a stored title that may still be English
                title = escape_html(m.get("title") or a.vietnamese_title or a.title)
                body.append(
                    f'{score_dot(score)} <b>{score:.1f}</b> · <a href="{escape_html(a.url)}">{title}</a>'
                )
                if m.get("why"):
                    body.append(f"<i>{escape_html(m['why'])}</i>")
            body.append("")

        mentioned = {i for t in themes for i in t["article_ids"]} | {
            m["id"] for m in briefing.get("must_read", [])
        }
        rest = len(articles) - len(mentioned)
        if rest > 0:
            body.append(f"<i>Còn {rest} bài khác trên TechPulse.</i>")

        text_only = _strip_tags("\n".join(body))
        minutes = max(1, round(len(text_only.split()) / 200))
        window = f" từ {since.hour:02d}:{since.minute:02d}" if since and kind.key == "update" else ""
        header = [
            _header_title(kind, now),
            f"<i>{len(articles)} bài mới{window} · đọc khoảng {minutes} phút</i>",
            "",
        ]
        return "\n".join(header + body).strip()

    # Stay under Telegram's message limit: shorten summaries first, then drop themes
    themes = list(briefing.get("themes") or [])
    msg = build(themes, 400)
    for limit in (300, 200):
        if len(msg) <= TELEGRAM_LIMIT:
            break
        msg = build(themes, limit)
    while len(msg) > TELEGRAM_LIMIT and themes:
        themes.pop()
        msg = build(themes, 200)

    reply_markup = {"inline_keyboard": [[{"text": "Mở TechPulse", "url": FRONTEND_URL}]]}
    return msg, reply_markup


async def dispatch_briefing(kind: BriefingKind, since: datetime.datetime) -> bool:
    """Build and send one briefing for articles created since `since` (tz-aware).
    Returns True if Telegram accepted a message, False if it failed or was skipped."""
    from app.core.database import AsyncSessionLocal

    now = _local_now()
    async with AsyncSessionLocal() as db:
        articles = await collect_briefing_articles(db, since.astimezone(datetime.timezone.utc))

        if not articles:
            if not kind.send_when_empty:
                logger.info(f"{kind.title} {now:%H:%M}: no new articles since {since:%H:%M}, skipped")
                return False
            msg = f"{_header_title(kind, now)}\nKhông có bài mới nào được phân tích trong 24 giờ qua."
            return await send_telegram_message(msg, disable_preview=True)

        briefing, model_used = await generate_daily_briefing(articles, kind)
        msg, markup = format_daily_briefing_message(briefing, articles, now=now, kind=kind, since=since)
        logger.info(
            f"{kind.title}: {len(articles)} articles, {len(briefing['themes'])} themes, "
            f"model={model_used}, {len(msg)} chars"
        )
        return await send_telegram_message(msg, reply_markup=markup, disable_preview=True)


async def dispatch_daily_briefing(hours: int = 24) -> bool:
    """End-of-day super summary of the last `hours` hours. Also used on demand (/tomtat)."""
    since = _local_now() - datetime.timedelta(hours=hours)
    return await dispatch_briefing(DAILY, since)


async def dispatch_scheduled_update() -> bool:
    """Short update covering everything since the previous scheduled briefing."""
    now = _local_now()
    return await dispatch_briefing(UPDATE, previous_slot_start(now, schedule_hours()))
