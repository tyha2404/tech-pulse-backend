import html
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

FRONTEND_URL = getattr(settings, "FRONTEND_URL", "http://127.0.0.1:5174")

# In-memory rate limiter for urgent alerts
_urgent_alert_timestamps: List[float] = []


def escape_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return html.escape(str(text))


def _check_urgent_rate_limit() -> bool:
    """Returns True if within rate limit, False if throttled"""
    global _urgent_alert_timestamps
    now = time.time()
    one_hour_ago = now - 3600.0
    # Clean up older than 1 hour
    _urgent_alert_timestamps = [t for t in _urgent_alert_timestamps if t > one_hour_ago]
    if len(_urgent_alert_timestamps) >= settings.TELEGRAM_ALERT_MAX_PER_HOUR:
        return False
    _urgent_alert_timestamps.append(now)
    return True


def format_urgent_alert_message(article: Any) -> Tuple[str, Dict[str, Any]]:
    """Format real-time urgent push for score >= 9.0 with inline feedback buttons"""
    article_id = getattr(article, "id", None)
    title = escape_html(
        getattr(article, "vietnamese_title", None)
        or getattr(article, "title", None)
        or "Tin đột phá công nghệ"
    )
    score = getattr(article, "relevance_score", 9.0) or 9.0
    source_name = escape_html(
        article.source.name if hasattr(article, "source") and article.source else "TechPulse"
    )
    summary = escape_html(
        getattr(article, "vietnamese_summary", None)
        or getattr(article, "title", None)
        or ""
    )
    url = getattr(article, "url", FRONTEND_URL)
    takeaways = getattr(article, "key_takeaways", []) or []
    tech_stacks = getattr(article, "new_tech_stacks", []) or []
    tags = getattr(article, "tags", []) or []

    lines = [
        f"🚨 <b>[TECHPULSE BREAKING / MUST READ] ⭐ {score:.1f}/10</b>",
        "",
        f"📌 <b>{title}</b>",
        f"🌐 <i>Nguồn: {source_name}</i>",
        "",
        f"📝 {summary}",
    ]

    if takeaways:
        lines.append("")
        lines.append("💡 <b>Điểm cốt lõi kỹ thuật:</b>")
        for pt in takeaways[:3]:
            lines.append(f"• {escape_html(pt)}")

    if tech_stacks:
        tech_names = [escape_html(t.get("name") if isinstance(t, dict) else str(t)) for t in tech_stacks[:3]]
        lines.append("")
        lines.append(f"🛠 <b>Tech Stack phát hiện:</b> {', '.join(tech_names)}")

    if tags:
        tag_str = " ".join([f"#{t.replace(' ', '_').replace('-', '_')}" for t in tags[:5]])
        lines.append("")
        lines.append(f"🏷 {escape_html(tag_str)}")

    msg = "\n".join(lines)

    webapp_url = (
        f"{FRONTEND_URL}/?article_id={article_id}" if article_id else FRONTEND_URL
    )

    # Inline feedback buttons & action links
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "👍 Hữu ích", "callback_data": f"fb:like:{article_id}"},
                {"text": "👎 Kém", "callback_data": f"fb:dislike:{article_id}"},
                {"text": "🔖 Lưu bài", "callback_data": f"fb:bm:{article_id}"},
            ],
            [
                {"text": "📖 Đọc bài gốc ↗", "url": url},
                {"text": "⚡ Mở TechPulse ↗", "url": webapp_url},
            ],
        ]
    }
    return msg, reply_markup


def format_elite_article_message(article: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    title = escape_html(
        article.get("vietnamese_title") or article.get("title") or "Bản tin công nghệ"
    )
    score = article.get("relevance_score", 0.0) or 0.0
    source_name = escape_html(article.get("source_name") or "TechPulse")
    reading_time = article.get("reading_time_minutes", 3)
    summary = escape_html(
        article.get("vietnamese_summary") or article.get("title") or ""
    )
    url = article.get("url") or FRONTEND_URL
    article_id = article.get("id")

    lines = [
        f"🔥 <b>[TECHPULSE ELITE] {score:.1f}/10</b>",
        f"📌 <b>{title}</b>",
        f"🌐 <i>Nguồn: {source_name} | ⏱ {reading_time} phút đọc</i>",
        "",
        f"📝 {summary}",
    ]

    takeaways = article.get("key_takeaways") or []
    if takeaways:
        lines.append("")
        lines.append("💡 <b>Điểm cốt lõi kỹ thuật:</b>")
        for pt in takeaways[:3]:
            lines.append(f"• {escape_html(pt)}")

    tags = article.get("tags") or []
    if tags:
        tag_str = " ".join(
            [f"#{t.replace(' ', '_').replace('-', '_')}" for t in tags[:5]]
        )
        lines.append("")
        lines.append(f"🏷 {escape_html(tag_str)}")

    msg = "\n".join(lines)

    webapp_url = (
        f"{FRONTEND_URL}/?article_id={article_id}" if article_id else FRONTEND_URL
    )
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "👍", "callback_data": f"fb:like:{article_id}"},
                {"text": "👎", "callback_data": f"fb:dislike:{article_id}"},
                {"text": "🔖", "callback_data": f"fb:bm:{article_id}"},
                {"text": "📖 Đọc gốc ↗", "url": url},
                {"text": "⚡ TechPulse ↗", "url": webapp_url},
            ]
        ]
    }
    return msg, reply_markup


def format_cluster_alert_message(
    cluster_data: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    title = escape_html(cluster_data.get("title") or "Sự kiện công nghệ mới")
    sources = cluster_data.get("sources") or []
    summary = escape_html(cluster_data.get("summary") or "")
    url = cluster_data.get("url") or FRONTEND_URL
    source_count = len(sources)

    sources_str = ", ".join([escape_html(s) for s in sources])

    lines = [
        f"🚨 <b>XU HƯỚNG CÔNG NGHỆ ĐANG NÓNG ({source_count} nguồn cùng đưa tin)</b>",
        "",
        f"📌 <b>{title}</b>",
        f"📰 <b>Được đưa tin bởi:</b> {sources_str}",
        "",
        f"🎯 <b>Góc nhìn TechPulse:</b> {summary}",
    ]
    msg = "\n".join(lines)

    reply_markup = {
        "inline_keyboard": [[{"text": "🌐 Đọc chi tiết trên TechPulse ↗", "url": url}]]
    }
    return msg, reply_markup


def format_espresso_digest_message(
    articles: List[Dict[str, Any]], title_label: str
) -> Tuple[str, Dict[str, Any]]:
    lines = [
        f"<b>{escape_html(title_label)}</b>",
        "<i>Top bài phân tích chuyên sâu tuyển chọn dành cho bạn:</i>",
        "",
    ]
    for idx, art in enumerate(articles[:5], 1):
        art_title = escape_html(art.get("vietnamese_title") or art.get("title") or "")
        score = art.get("relevance_score", 0.0) or 0.0
        art_url = art.get("url") or FRONTEND_URL
        lines.append(f"<b>{idx}. {art_title}</b> (⭐ {score:.1f})")
        lines.append(f'👉 <a href="{art_url}">Xem bài viết</a>')
        lines.append("")

    lines.append(f"📱 <i>Khám phá toàn bộ bảng tin tại: {FRONTEND_URL}</i>")
    msg = "\n".join(lines)

    reply_markup = {
        "inline_keyboard": [[{"text": "⚡ Mở TechPulse Webapp ↗", "url": FRONTEND_URL}]]
    }
    return msg, reply_markup


def _sanitize_reply_markup_urls(
    reply_markup: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not reply_markup:
        return reply_markup
    import copy

    markup = copy.deepcopy(reply_markup)
    inline_keyboard = markup.get("inline_keyboard", [])
    for row in inline_keyboard:
        for btn in row:
            if "url" in btn and isinstance(btn["url"], str):
                btn["url"] = (
                    btn["url"]
                    .replace("http://localhost", "http://127.0.0.1")
                    .replace("https://localhost", "https://127.0.0.1")
                )
    return markup


async def send_telegram_message(
    html_text: str,
    reply_markup: Optional[Dict[str, Any]] = None,
    timeout: float = 6.0,
) -> bool:
    bot_token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    if reply_markup:
        payload["reply_markup"] = _sanitize_reply_markup_urls(reply_markup)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return True
            logger.warning(f"Telegram send failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        logger.error(f"Telegram notification error: {e}")
    return False


async def send_urgent_alert(article: Any) -> bool:
    """Send immediate alert for score >= 9.0 with rate limit protection"""
    if not _check_urgent_rate_limit():
        logger.info("Urgent alert throttled (exceeded max per hour)")
        return False

    msg, markup = format_urgent_alert_message(article)
    return await send_telegram_message(msg, reply_markup=markup)


async def answer_telegram_callback(callback_query_id: str, text: str) -> bool:
    bot_token = settings.TELEGRAM_BOT_TOKEN
    if not bot_token:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/answerCallbackQuery"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={"callback_query_id": callback_query_id, "text": text})
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"Failed to answer callback: {e}")
        return False
