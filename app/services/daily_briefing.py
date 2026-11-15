"""Telegram briefings: highlights synthesized across every article in the window, a short
must-read list, then every other article with a one-line description.

Two kinds run on a schedule (BRIEFING_TIMEZONE):
- short updates at BRIEFING_UPDATE_HOURS (default 9, 12, 18), each covering only the
  articles added since the previous scheduled briefing, skipped when there is nothing new;
- the end-of-day "super summary" at DAILY_BRIEFING_HOUR (default 22) covering 24 hours.
"""

import datetime
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.models import Article, Source
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
WEEKDAYS = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"]
DIVIDER = "━━━━━━━━━━━━━━"


@dataclass(frozen=True)
class BriefingKind:
    key: str
    title: str
    emoji: str
    max_must_read: int
    # Bullets in the "Tổng hợp nổi bật" section, synthesized across every article
    max_highlights: int
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
    max_must_read=3,
    max_highlights=4,
    scope="bản cập nhật nhanh giữa ngày, chỉ gồm các bài mới kể từ lần cập nhật trước",
    with_trend=False,
    send_when_empty=False,
)

DAILY = BriefingKind(
    key="daily",
    title="Siêu tổng hợp cuối ngày",
    emoji="🌙",
    max_must_read=5,
    max_highlights=6,
    scope="bản siêu tổng hợp cuối ngày cho toàn bộ 24 giờ qua",
    with_trend=True,
    send_when_empty=True,
)

BRIEFING_PROMPT = """Bạn là biên tập viên công nghệ viết {scope} cho lập trình viên Việt Nam.
Dưới đây là các bài tiếng Việt đã được chấm điểm và tóm tắt (JSON). Chỉ dùng thông tin có trong dữ liệu này.

QUY TẮC CHỐNG BỊA:
- Không thêm sự kiện, số liệu, tên sản phẩm/công ty/người, phiên bản hay ngày tháng không có trong dữ liệu.
- Giữ nguyên số liệu và tên riêng như trong dữ liệu; không làm tròn, không suy đoán.
- Không biến tin đồn, dự kiến hay ý kiến thành sự thật chắc chắn.
- Không có gì đáng nói thì viết ngắn hơn hoặc bỏ mục đó; không viết cho đủ số lượng.
- Không dùng kiến thức bên ngoài để giải thích thêm.

Hãy:
1. Chọn tối đa {max_must_read} bài đáng đọc kỹ nhất. Với mỗi bài:
   - "title": giữ nguyên tiêu đề trong dữ liệu (chỉ rút gọn nếu quá dài);
   - "summary": 1-2 câu nói bài viết nói về điều gì, dựa đúng vào "summary"/"takeaways" của bài, có số liệu/tên riêng nếu dữ liệu có;
   - "why": 1 câu nêu vì sao lập trình viên nên đọc, chỉ dựa trên nội dung bài, không chung chung.
2. Viết "headline": một câu mở đầu tóm tắt điểm chính của đợt tin này.
3. Viết "highlights": tối đa {max_highlights} ý tổng hợp nội dung nổi bật của TẤT CẢ các bài trong đợt
   (không chỉ các bài đáng đọc kỹ). Gom các bài cùng chủ đề vào một ý; mỗi ý 1-2 câu, nêu cụ thể
   sự kiện, con số, tên sản phẩm/công ty có trong dữ liệu; không lặp lại "headline"; ý quan trọng nhất đứng trước.{trend_instruction}

Toàn bộ bằng tiếng Việt tự nhiên, ngắn gọn, không dùng ký tự Trung/Nhật. Giữ nguyên tên riêng, tên sản phẩm và thuật ngữ kỹ thuật thông dụng.
Chỉ dùng id có trong dữ liệu.
Chỉ trả về JSON đúng dạng:
{{"headline": "...", "highlights": ["...", "..."],{trend_field}
  "must_read": [{{"id": 1, "title": "...", "summary": "...", "why": "..."}}]}}

DỮ LIỆU:
{articles_json}"""

TREND_INSTRUCTION = """
4. Viết "trend": 2-3 câu về xu hướng lớn nhất xuyên suốt cả ngày, chỉ rút ra từ các bài trong dữ liệu
   (nêu rõ nếu chỉ có một vài bài, không khẳng định xu hướng rộng hơn dữ liệu cho phép),
   kèm một gợi ý hành động cụ thể nếu dữ liệu đủ cơ sở; nếu không thì bỏ phần gợi ý."""


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


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def short_description(a: Article, limit: int = 180) -> str:
    """First sentence of the stored Vietnamese summary, else the first key takeaway."""
    text = (a.vietnamese_summary or "").strip()
    if text:
        text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    else:
        text = next((t for t in (a.key_takeaways or []) if t), "")
    return _truncate(text, limit)


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
    """AI-analyzed, canonical, visible articles scoring >= BRIEFING_MIN_SCORE since `since`, best first."""
    res = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(
            Article.created_at >= since,
            Article.is_processed == True,
            Article.is_canonical == True,
            Article.is_hidden == False,
            Article.source_id.in_(select(Source.id).where(Source.is_active == True)),
            Article.relevance_score >= settings.BRIEFING_MIN_SCORE,
        )
        .order_by(Article.relevance_score.desc(), Article.id.desc())
    )
    return list(res.scalars().all())


def _validate_briefing(
    data: Dict[str, Any], by_id: Dict[int, Article], kind: BriefingKind = DAILY
) -> Optional[Dict[str, Any]]:
    """Keep only well-formed picks that reference real article ids."""
    must_read = []
    seen = set()
    for m in data.get("must_read") or []:
        if not isinstance(m, dict):
            continue
        try:
            aid = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        if aid not in by_id or aid in seen:
            continue
        seen.add(aid)
        must_read.append(
            {
                "id": aid,
                "title": str(m.get("title") or ""),
                "summary": str(m.get("summary") or ""),
                "why": str(m.get("why") or ""),
            }
        )

    if not must_read:
        return None
    raw_highlights = data.get("highlights")
    highlights = [
        str(h).strip()
        for h in (raw_highlights if isinstance(raw_highlights, list) else [])
        if isinstance(h, (str, int, float)) and str(h).strip()
    ]
    return {
        "headline": str(data.get("headline") or ""),
        "highlights": highlights[: kind.max_highlights],
        "trend": str(data.get("trend") or "") if kind.with_trend else "",
        "must_read": must_read[: kind.max_must_read],
    }


def fallback_briefing(articles: List[Article], kind: BriefingKind = DAILY) -> Dict[str, Any]:
    """No-LLM briefing: the top-scored articles become the picks.
    Articles the crawler couldn't run through an LLM are not picked (they are still listed):
    their titles were never translated and nothing here can translate them."""
    top = [a for a in articles if a.ai_model_used != EXTRACTIVE_FALLBACK_MODEL]
    must_read = [
        {"id": a.id, "title": "", "summary": "", "why": (a.key_takeaways or [""])[0] or ""}
        for a in top[: kind.max_must_read]
    ]
    high = sum(1 for a in articles if (a.relevance_score or 0) >= 8)
    return {
        "headline": f"{len(articles)} bài mới, {high} bài từ 8 điểm. (Tóm tắt tự động, AI không phản hồi.)",
        "highlights": fallback_highlights(top, kind.max_highlights),
        "trend": "",
        "must_read": must_read,
    }


def fallback_highlights(articles: List[Article], limit: int) -> List[str]:
    """No-LLM highlights: the most common tags, each with its best article's first summary
    sentence. Takes only LLM-analyzed articles, whose summaries are already Vietnamese."""
    groups: Dict[str, List[Article]] = {}
    for a in articles:  # best first, so each group's first article is its top one
        tag = next((t for t in (a.tags or []) if t), None)
        if tag:
            groups.setdefault(tag, []).append(a)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:limit]
    highlights = []
    for tag, group in ranked:
        description = short_description(group[0], 160)
        highlights.append(f"{tag} ({len(group)} bài): {description}" if description else f"{tag}: {len(group)} bài")
    return highlights


async def generate_daily_briefing(
    articles: List[Article], kind: BriefingKind = DAILY
) -> Tuple[Dict[str, Any], str]:
    """Returns (briefing, model_used). Walks the model fallback chain, then falls back to tags."""
    by_id = {a.id: a for a in articles}
    prompt = BRIEFING_PROMPT.format(
        scope=kind.scope,
        max_must_read=kind.max_must_read,
        max_highlights=kind.max_highlights,
        trend_instruction=TREND_INSTRUCTION if kind.with_trend else "",
        trend_field=' "trend": "...",' if kind.with_trend else "",
        articles_json=json.dumps(
            [_article_payload(a) for a in articles], ensure_ascii=False
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
                temperature=0.1,
            )
            data = _clean_json_response(message_text(response))
            briefing = _validate_briefing(data, by_id, kind)
            if briefing is None:
                raise ValueError("briefing JSON had no usable picks")
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
    """The main briefing message only; see format_briefing_messages for the full set."""
    messages, reply_markup = format_briefing_messages(briefing, articles, now=now, kind=kind, since=since)
    return messages[0], reply_markup


def format_briefing_messages(
    briefing: Dict[str, Any],
    articles: List[Article],
    now: Optional[datetime.datetime] = None,
    kind: BriefingKind = DAILY,
    since: Optional[datetime.datetime] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Main briefing (headline, trend, must-read cards), then as many messages as needed
    to list every other article, each with a short description."""
    by_id = {a.id: a for a in articles}
    now = now or _local_now()

    def meta_line(a: Article) -> str:
        score = a.relevance_score or 0
        parts = [f"{score_dot(score)} <b>{score:.1f}</b>"]
        if a.source:
            parts.append(escape_html(a.source.name))
        if a.reading_time_minutes:
            parts.append(f"{a.reading_time_minutes} phút đọc")
        return " · ".join(parts)

    def must_read_card(i: int, m: Dict[str, Any], with_summary: bool) -> str:
        a = by_id[m["id"]]
        # The LLM's translated title wins over a stored title that may still be English
        title = escape_html(m.get("title") or a.vietnamese_title or a.title)
        lines = [f'<b>{i}. <a href="{escape_html(a.url)}">{title}</a></b>', meta_line(a)]
        if with_summary:
            summary = m.get("summary") or short_description(a, 280)
            if summary:
                lines += ["", escape_html(_truncate(summary, 280))]
        if m.get("why"):
            lines.append(f"<i>→ {escape_html(_truncate(m['why'], 200))}</i>")
        return "<blockquote>" + "\n".join(lines) + "</blockquote>"

    def build(
        picks: List[Dict[str, Any]], with_summary: bool, rest_note: bool = True, with_highlights: bool = True
    ) -> str:
        body: List[str] = []
        if briefing.get("headline"):
            body += [f"<b>{escape_html(briefing['headline'])}</b>", ""]

        highlights = briefing.get("highlights") or []
        if highlights and with_highlights:
            bullets = "\n".join(f"• {escape_html(_truncate(h, 260))}" for h in highlights)
            body += ["🧭 <b>TỔNG HỢP NỔI BẬT</b>", f"<blockquote>{bullets}</blockquote>", ""]

        if briefing.get("trend"):
            body += ["📈 <b>XU HƯỚNG CỦA NGÀY</b>", f"<blockquote>{escape_html(briefing['trend'])}</blockquote>", ""]

        if picks:
            body += [DIVIDER, "⭐ <b>ĐÁNG ĐỌC KỸ</b>", ""]
            for i, m in enumerate(picks, 1):
                body += [must_read_card(i, m, with_summary), ""]

        rest = len(articles) - len(picks)
        if rest > 0 and rest_note:
            body.append(f"<i>Còn {rest} bài khác, xem danh sách bên dưới.</i>")

        text_only = _strip_tags("\n".join(body))
        minutes = max(1, round(len(text_only.split()) / 200))
        window = f" từ {since.hour:02d}:{since.minute:02d}" if since and kind.key == "update" else ""
        header = [
            _header_title(kind, now),
            f"<i>{len(articles)} bài mới{window} · đọc khoảng {minutes} phút</i>",
            "",
        ]
        return "\n".join(header + body).strip()

    # Stay under Telegram's message limit: drop pick summaries first, then move the
    # lowest picks down into the full list below, and only then drop the highlights
    picks = [m for m in briefing.get("must_read") or [] if m["id"] in by_id]
    with_summary = True
    with_highlights = True
    msg = build(picks, with_summary)
    if len(msg) > TELEGRAM_LIMIT:
        with_summary = False
        msg = build(picks, with_summary)
    while len(msg) > TELEGRAM_LIMIT and picks:
        picks.pop()
        msg = build(picks, with_summary)
    if len(msg) > TELEGRAM_LIMIT:
        with_highlights = False
        msg = build(picks, with_summary, with_highlights=with_highlights)

    reply_markup = {"inline_keyboard": [[{"text": "Mở TechPulse", "url": FRONTEND_URL}]]}

    picked = {m["id"] for m in picks}
    items = []
    for a in articles:
        if a.id in picked:
            continue
        title = escape_html(a.vietnamese_title or a.title)
        item = f'{meta_line(a)}\n<a href="{escape_html(a.url)}">{title}</a>'
        description = short_description(a)
        if description:
            item += f"\n<i>{escape_html(description)}</i>"
        items.append(item)

    if not items:
        return [msg], reply_markup

    heading = f"📋 <b>CÁC BÀI KHÁC ({len(items)})</b>"
    # One message when everything fits; split only past Telegram's limit
    single = build(picks, with_summary, rest_note=False, with_highlights=with_highlights) + "\n\n" + "\n\n".join([DIVIDER + "\n" + heading] + items)
    if len(single) <= TELEGRAM_LIMIT:
        return [single], reply_markup

    messages = [msg]
    chunk = heading
    for item in items:
        if len(chunk) + 2 + len(item) > TELEGRAM_LIMIT:
            messages.append(chunk)
            chunk = "📋 <b>CÁC BÀI KHÁC (tiếp)</b>"
        chunk += "\n\n" + item
    messages.append(chunk)
    return messages, reply_markup


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
        messages, markup = format_briefing_messages(briefing, articles, now=now, kind=kind, since=since)
        logger.info(
            f"{kind.title}: {len(articles)} articles, {len(briefing['must_read'])} picks, "
            f"model={model_used}, {len(messages)} messages"
        )
        # The "Mở TechPulse" button goes on the last message, under the full list
        sent = await send_telegram_message(
            messages[0], reply_markup=markup if len(messages) == 1 else None, disable_preview=True
        )
        for i, extra in enumerate(messages[1:], 2):
            sent = await send_telegram_message(
                extra, reply_markup=markup if i == len(messages) else None, disable_preview=True
            ) and sent
        return sent


async def dispatch_daily_briefing(hours: int = 24) -> bool:
    """End-of-day super summary of the last `hours` hours. Also used on demand (/tomtat)."""
    since = _local_now() - datetime.timedelta(hours=hours)
    return await dispatch_briefing(DAILY, since)


async def dispatch_scheduled_update() -> bool:
    """Short update covering everything since the previous scheduled briefing."""
    now = _local_now()
    return await dispatch_briefing(UPDATE, previous_slot_start(now, schedule_hours()))
