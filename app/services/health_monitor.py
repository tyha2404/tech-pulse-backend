"""Periodic self-check that pings Telegram when the AI gateway or the crawl pipeline breaks.

Each check returns a problem description or None. Alerts go out when a problem first
appears, as a reminder every HEALTH_REMIND_HOURS while it persists, and once on recovery.
State is in-memory, so a restart may re-announce an ongoing problem once.
"""

import datetime
import logging
import time
from typing import Dict, Optional

from sqlalchemy import func, select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.models import Article, CrawlRun
from app.services.ai_analyzer import EXTRACTIVE_FALLBACK_MODEL, get_ai_client, message_text
from app.services.telegram_service import escape_html, send_telegram_message

logger = logging.getLogger(__name__)

CHECK_LABELS = {
    "ai_gateway": "AI gateway (9router)",
    "ai_fallback": "Chấm điểm bài viết",
    "crawl_failures": "Crawl nguồn tin",
    "crawl_stale": "Bài mới",
}

# check name -> monotonic time of the last alert sent for it
_active: Dict[str, float] = {}


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


async def check_ai_gateway() -> Optional[str]:
    """Problem if no model in the fallback chain answers a tiny prompt."""
    client = get_ai_client(settings.AI_TIMEOUT_BACKGROUND)
    errors = []
    for model_name in settings.fallback_models_list:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Trả lời đúng một chữ: ok"}],
                # Reasoning models spend tokens before answering, so leave headroom
                max_tokens=256,
            )
            message_text(response)
            return None
        except Exception as e:
            errors.append(f"{model_name}: {str(e)[:120]}")
    return "Không model nào phản hồi.\n" + "\n".join(errors)


async def check_ai_fallback_ratio(db) -> Optional[str]:
    """Problem if most recently analyzed articles got the no-LLM heuristic score."""
    since = _utc_now() - datetime.timedelta(minutes=settings.HEALTH_WINDOW_MINUTES)
    res = await db.execute(
        select(
            func.count(),
            func.count().filter(Article.ai_model_used == EXTRACTIVE_FALLBACK_MODEL),
        ).where(Article.created_at >= since, Article.is_processed == True)
    )
    total, fallback = res.one()
    if total >= 5 and fallback / total >= 0.5:
        return (
            f"{fallback}/{total} bài trong {settings.HEALTH_WINDOW_MINUTES} phút qua bị chấm bằng "
            "heuristic (không qua LLM), điểm sẽ thấp bất thường."
        )
    return None


async def check_crawl_failures(db) -> Optional[str]:
    """Problem if most crawl runs in the window failed."""
    since = _utc_now() - datetime.timedelta(minutes=settings.HEALTH_WINDOW_MINUTES)
    res = await db.execute(
        select(func.count(), func.count().filter(CrawlRun.status == "failed")).where(
            CrawlRun.started_at >= since
        )
    )
    total, failed = res.one()
    if total >= 5 and failed / total >= 0.8:
        return f"{failed}/{total} lượt crawl trong {settings.HEALTH_WINDOW_MINUTES} phút qua thất bại."
    return None


async def check_crawl_stale(db) -> Optional[str]:
    """Problem if no article has been saved for HEALTH_STALE_HOURS."""
    latest = (await db.execute(select(func.max(Article.created_at)))).scalar()
    if latest is None:
        return None
    age = _utc_now() - latest
    if age >= datetime.timedelta(hours=settings.HEALTH_STALE_HOURS):
        return f"Không có bài mới nào trong {age.total_seconds() / 3600:.1f} giờ qua."
    return None


async def collect_problems() -> Dict[str, Optional[str]]:
    results: Dict[str, Optional[str]] = {"ai_gateway": await check_ai_gateway()}
    async with AsyncSessionLocal() as db:
        results["ai_fallback"] = await check_ai_fallback_ratio(db)
        results["crawl_failures"] = await check_crawl_failures(db)
        results["crawl_stale"] = await check_crawl_stale(db)
    return results


async def run_health_check() -> Dict[str, Optional[str]]:
    results = await collect_problems()
    now = time.monotonic()
    remind_after = settings.HEALTH_REMIND_HOURS * 3600

    for name, problem in results.items():
        label = CHECK_LABELS[name]
        if problem:
            last_sent = _active.get(name)
            if last_sent is not None and now - last_sent < remind_after:
                continue
            prefix = "Vẫn còn lỗi" if last_sent is not None else "Phát hiện lỗi"
            logger.warning(f"🩺 [Health] {name}: {problem}")
            msg = f"⚠️ <b>{prefix}: {escape_html(label)}</b>\n\n{escape_html(problem)}"
            if await send_telegram_message(msg, disable_preview=True):
                _active[name] = now
        elif name in _active:
            logger.info(f"🩺 [Health] {name} recovered")
            msg = f"✅ <b>Đã ổn định lại: {escape_html(label)}</b>"
            if await send_telegram_message(msg, disable_preview=True):
                del _active[name]
    return results
