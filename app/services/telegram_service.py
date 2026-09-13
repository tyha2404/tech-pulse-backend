import html
import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

# Telegram API does not allow 'localhost' as inline button domain; use 127.0.0.1 or production domain
FRONTEND_URL = getattr(settings, "FRONTEND_URL", "http://127.0.0.1:5174")

def escape_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return html.escape(str(text))

def format_elite_article_message(article: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    title = escape_html(article.get("vietnamese_title") or article.get("title") or "Bản tin công nghệ")
    score = article.get("relevance_score", 0.0) or 0.0
    source_name = escape_html(article.get("source_name") or "TechPulse")
    reading_time = article.get("reading_time_minutes", 3)
    summary = escape_html(article.get("vietnamese_summary") or article.get("title") or "")
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
        tag_str = " ".join([f"#{t.replace(' ', '_').replace('-', '_')}" for t in tags[:5]])
        lines.append("")
        lines.append(f"🏷 {escape_html(tag_str)}")

    msg = "\n".join(lines)

    webapp_url = f"{FRONTEND_URL}/?article_id={article_id}" if article_id else FRONTEND_URL
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "📖 Đọc bài gốc ↗", "url": url},
                {"text": "⚡ Mở TechPulse ↗", "url": webapp_url},
            ]
        ]
    }
    return msg, reply_markup

def format_cluster_alert_message(cluster_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
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
        "inline_keyboard": [
            [
                {"text": "🌐 Đọc chi tiết trên TechPulse ↗", "url": url}
            ]
        ]
    }
    return msg, reply_markup

def format_espresso_digest_message(articles: List[Dict[str, Any]], title_label: str) -> Tuple[str, Dict[str, Any]]:
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
        lines.append(f"👉 <a href=\"{art_url}\">Xem bài viết</a>")
        lines.append("")

    lines.append(f"📱 <i>Khám phá toàn bộ bảng tin tại: {FRONTEND_URL}</i>")
    msg = "\n".join(lines)

    reply_markup = {
        "inline_keyboard": [
            [{"text": "⚡ Mở TechPulse Webapp ↗", "url": FRONTEND_URL}]
        ]
    }
    return msg, reply_markup

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
        payload["reply_markup"] = reply_markup

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return True
            logger.warning(f"Telegram send failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        logger.error(f"Telegram notification error: {e}")
    return False


async def dispatch_daily_espresso_digest(title_label: str = "☕ Morning Tech Espresso (8:00 AM)") -> bool:
    """
    Queries top articles from the last 24 hours, formats a digest and sends it to Telegram.
    """
    import datetime
    from sqlalchemy import select
    from app.core.database import AsyncSessionLocal
    from app.models.models import Article

    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    async with AsyncSessionLocal() as db:
        stmt = (
            select(Article)
            .where(
                Article.created_at >= since,
                Article.is_canonical == True,
                Article.is_hidden == False,
            )
            .order_by(Article.relevance_score.desc(), Article.id.desc())
            .limit(3)
        )
        res = await db.execute(stmt)
        articles = res.scalars().all()

        if not articles:
            # Fallback to recent top articles if none in 24h
            fallback_stmt = (
                select(Article)
                .where(Article.is_canonical == True, Article.is_hidden == False)
                .order_by(Article.relevance_score.desc(), Article.id.desc())
                .limit(3)
            )
            res = await db.execute(fallback_stmt)
            articles = res.scalars().all()

        if not articles:
            return False

        data_list = [
            {
                "id": a.id,
                "title": a.title,
                "vietnamese_title": a.vietnamese_title,
                "relevance_score": a.relevance_score,
                "url": a.url,
            }
            for a in articles
        ]

        msg, markup = format_espresso_digest_message(data_list, title_label)
        return await send_telegram_message(msg, reply_markup=markup)

