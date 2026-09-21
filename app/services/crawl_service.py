import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import httpx
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import Source, Article, CrawlRun
from app.crawlers.article_crawler import (
    fetch_rss_feed,
    fetch_json_feed,
    fetch_sitemap_feed,
    fetch_hacker_news,
    extract_clean_article_content,
    make_tz_aware,
)
from app.services.ai_analyzer import analyze_article_with_9router
from app.services.clustering_service import assign_article_cluster, assign_article_cluster_async
from app.services.embedding_service import generate_article_embedding
from app.services.triage_service import fast_triage_article
from app.services.telegram_service import (
    format_elite_article_message,
    format_cluster_alert_message,
    send_urgent_alert,
    send_instant_flash_alert,
    send_telegram_message,
)


async def crawl_single_source(
    source: Source, db: AsyncSession, run_ai: bool = True
) -> int:
    """Crawls a given source (RSS, JSON Feed, Sitemap, HN) and tracks CrawlRun metrics"""
    start_time = time.time()
    started_at = datetime.now(timezone.utc)
    source.status = "pending"
    await db.commit()

    articles_found = 0
    articles_added = 0
    error_message = None
    http_status = 200

    try:
        raw_items = []
        target_url = source.feed_url or source.url
        source_type = (source.source_type or "").lower()

        if source_type == "hn" or "ycombinator.com" in source.url:
            raw_items = await fetch_hacker_news()
        elif source_type == "json":
            raw_items = await fetch_json_feed(target_url)
        elif source_type == "sitemap":
            raw_items = await fetch_sitemap_feed(target_url)
        elif source_type == "rss" or source.feed_url:
            raw_items = await fetch_rss_feed(target_url)
        else:
            # Fallback attempt: rss, then direct
            try:
                raw_items = await fetch_rss_feed(source.url)
            except Exception:
                raw_items = []

        articles_found = len(raw_items)

        for item in raw_items:
            # Check if already exists
            existing = await db.execute(
                select(Article).where(Article.url == item["url"])
            )
            if existing.scalar_one_or_none():
                continue

            # Extract full text if raw_content is short
            full_content = item.get("raw_content", "")
            if len(full_content) < 300:
                extracted = await extract_clean_article_content(item["url"])
                if extracted:
                    full_content = extracted

            now_utc = datetime.now(timezone.utc)
            pub_at = make_tz_aware(item.get("published_at")) or now_utc

            article = Article(
                source_id=source.id,
                title=item["title"],
                url=item["url"],
                author=item.get("author"),
                published_at=pub_at,
                raw_content=full_content,
                is_processed=False,
                created_at=now_utc,
            )

            # Tier-1 Fast Triage (System One / Jev pattern)
            triage_res, triage_model = await fast_triage_article(
                title=article.title, snippet=full_content[:500], url=article.url
            )

            # Speculative Fan-out: If critical breaking news detected, dispatch Flash Alert in <200ms
            flash_alert_dispatched = False
            if triage_res.is_breaking_news or triage_res.urgency_level == "CRITICAL":
                try:
                    await send_instant_flash_alert(
                        title=article.title,
                        url=article.url,
                        source_name=source.name,
                        snippet=full_content[:300],
                        depth_score=triage_res.tech_depth_score,
                    )
                    flash_alert_dispatched = True
                    print(f"[Flash Breaking Alert] Dispatched instantly for: '{article.title[:50]}'")
                except Exception as flash_err:
                    print(f"[Flash Alert Notice] {flash_err}")

            # Confidence-Gated Routing:
            # If high confidence discard (spam/marketing/non-tech), drop or skip full AI
            if triage_res.suggested_priority == "DISCARD" and triage_res.confidence >= 0.80:
                print(f"[Triage DISCARD] '{article.title[:50]}' via {triage_model} (confidence: {triage_res.confidence})")
                continue

            # Full AI analysis with 9routers (multi-model fallback)
            if run_ai:
                if triage_res.suggested_priority == "STORE_UNANALYZED" and triage_res.confidence >= 0.85:
                    # Light record without heavy NestJS Blueprint generation
                    article.is_processed = True
                    article.is_worth_reading = False
                    article.relevance_score = 4.0
                    article.vietnamese_title = article.title
                    article.vietnamese_summary = full_content[:200]
                    article.ai_model_used = f"{triage_model}-light"
                else:
                    analysis, model_used = await analyze_article_with_9router(
                        title=article.title, content=full_content, url=article.url
                    )
                    article.is_processed = True
                    article.is_worth_reading = analysis.is_worth_reading
                    article.relevance_score = analysis.relevance_score
                    article.vietnamese_title = analysis.vietnamese_title
                    article.vietnamese_summary = analysis.vietnamese_summary
                    article.key_takeaways = analysis.key_takeaways
                    article.new_tech_stacks = [
                        t.model_dump() for t in analysis.new_tech_stack
                    ]
                    article.tags = analysis.tags
                    article.target_audience = analysis.target_audience
                    article.architectural_tradeoffs = (
                        analysis.architectural_tradeoffs.model_dump()
                        if analysis.architectural_tradeoffs
                        else {}
                    )
                    article.nestjs_blueprint = (
                        analysis.nestjs_blueprint.model_dump()
                        if analysis.nestjs_blueprint
                        else {}
                    )
                    article.learning_path = (
                        analysis.learning_path.model_dump()
                        if analysis.learning_path
                        else {}
                    )
                    article.cluster_topic_key = analysis.cluster_topic_key
                    article.ai_model_used = model_used

            # Story Clustering & Deduplication within 48h window
            since_time = datetime.now(timezone.utc) - timedelta(hours=48)
            recent_res = await db.execute(
                select(Article).where(Article.created_at >= since_time)
            )
            recent_candidates = recent_res.scalars().all()

            cluster_id, is_canonical, demoted_id = await assign_article_cluster_async(
                article, recent_candidates
            )
            article.cluster_id = cluster_id
            article.is_canonical = is_canonical

            if demoted_id:
                demoted_article = await db.get(Article, demoted_id)
                if demoted_article:
                    demoted_article.is_canonical = False

            db.add(article)
            await db.commit()  # Commit each article safely to generate article.id
            articles_added += 1

            # Generate and save embedding (Module 4)
            try:
                embed_vec = await generate_article_embedding(article)
                if embed_vec:
                    article.embedding = embed_vec
                    await db.commit()
            except Exception as emb_err:
                print(f"Embedding generation notice: {emb_err}")

            # Smart & Urgent Telegram Notification Trigger (Module 3)
            try:
                cluster_sources = []
                if article.cluster_id:
                    c_res = await db.execute(
                        select(Article)
                        .options(selectinload(Article.source))
                        .where(Article.cluster_id == article.cluster_id)
                    )
                    c_articles = c_res.scalars().all()
                    cluster_sources = [
                        a.source.name for a in c_articles if a.source and a.source.name
                    ]
                await trigger_smart_article_notifications(
                    article,
                    cluster_sources=cluster_sources,
                    skip_urgent_if_flashed=flash_alert_dispatched,
                )
            except Exception as noti_err:
                print(f"Notification error: {noti_err}")

        # Update source health status
        source.status = "healthy"
        source.last_crawled_at = datetime.now(timezone.utc)
        source.last_error = None
        source.articles_count = (source.articles_count or 0) + articles_added
        
        # Record successful CrawlRun
        duration_ms = int((time.time() - start_time) * 1000)
        crawl_run = CrawlRun(
            source_id=source.id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            http_status=http_status,
            articles_found=articles_found,
            articles_new=articles_added,
            status="success",
            error_message=None,
        )
        db.add(crawl_run)
        await db.commit()

    except Exception as e:
        await db.rollback()
        duration_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        source.status = "error"
        source.last_error = error_message
        
        # Record failed CrawlRun
        crawl_run = CrawlRun(
            source_id=source.id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            http_status=500,
            articles_found=articles_found,
            articles_new=articles_added,
            status="failed",
            error_message=error_message,
        )
        db.add(crawl_run)
        try:
            await db.commit()
        except Exception:
            pass
        raise e

    return articles_added


async def trigger_smart_article_notifications(
    article: Article,
    cluster_sources: List[str] = [],
    skip_urgent_if_flashed: bool = False,
):
    """
    Quy tắc gửi thông báo Telegram:
    - Score >= 9.0: Gửi khẩn cấp Real-time Urgent Alert (Module 3) nếu chưa bắn Flash Alert
    - Score >= 7.5: Gửi thông báo tin tuyển chọn hoặc cụm tin nóng
    """
    relevance = article.relevance_score or 0.0

    # Module 3: Real-time Urgent Alert for score >= 9.0 (unless already sent via Flash Alert)
    if relevance >= settings.TELEGRAM_ALERT_THRESHOLD:
        if not skip_urgent_if_flashed:
            await send_urgent_alert(article)
        return

    # Normal smart alert for score >= 7.5
    if relevance >= 7.5:
        unique_sources = list(dict.fromkeys(cluster_sources))
        if len(unique_sources) >= 3:
            cluster_data = {
                "title": article.vietnamese_title or article.title,
                "sources": unique_sources,
                "summary": article.vietnamese_summary
                or "Đang được nhiều nguồn uy tín đưa tin.",
                "url": article.url,
            }
            msg, markup = format_cluster_alert_message(cluster_data)
            await send_telegram_message(msg, reply_markup=markup)
            return

        content_for_estimate = (
            article.raw_content or article.vietnamese_summary or article.title
        )
        reading_time = (
            max(1, round(len(content_for_estimate.split()) / 200))
            if content_for_estimate
            else 3
        )

        data = {
            "id": article.id,
            "title": article.title,
            "vietnamese_title": article.vietnamese_title,
            "vietnamese_summary": article.vietnamese_summary,
            "relevance_score": article.relevance_score,
            "source_name": article.source.name if article.source else "TechPulse",
            "reading_time_minutes": reading_time,
            "key_takeaways": article.key_takeaways or [],
            "tags": article.tags or [],
            "url": article.url,
        }
        msg, markup = format_elite_article_message(data)
        await send_telegram_message(msg, reply_markup=markup)
