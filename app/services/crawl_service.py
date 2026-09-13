from datetime import datetime, timezone, timedelta
from typing import List, Optional
import httpx
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Source, Article
from app.crawlers.article_crawler import (
    fetch_rss_feed,
    fetch_hacker_news,
    extract_clean_article_content,
    make_tz_aware,
)
from app.services.ai_analyzer import analyze_article_with_9router
from app.core.config import settings


async def crawl_single_source(
    source: Source, db: AsyncSession, run_ai: bool = True
) -> int:
    """Crawls a given source and optionally triggers 9routers AI analysis"""
    source.status = "pending"
    await db.commit()

    articles_added = 0
    try:
        raw_items = []
        if source.source_type == "hn" or "ycombinator.com" in source.url:
            raw_items = await fetch_hacker_news()
        elif source.feed_url:
            raw_items = await fetch_rss_feed(source.feed_url)
        else:
            raw_items = await fetch_rss_feed(source.url)

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

            # AI analysis with 9routers
            if run_ai:
                analysis = await analyze_article_with_9router(
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
                article.ai_model_used = settings.AI_MODEL

            # Story Clustering & Deduplication within 48h window
            from app.services.clustering_service import assign_article_cluster
            since_time = datetime.now(timezone.utc) - timedelta(hours=48)
            recent_res = await db.execute(
                select(Article).where(Article.created_at >= since_time)
            )
            recent_candidates = recent_res.scalars().all()

            cluster_id, is_canonical, demoted_id = assign_article_cluster(
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

            # Smart Telegram Notification Trigger
            try:
                cluster_sources = []
                if article.cluster_id:
                    # Find all distinct source names in this cluster
                    c_res = await db.execute(
                        select(Article).options(selectinload(Article.source)).where(Article.cluster_id == article.cluster_id)
                    )
                    c_articles = c_res.scalars().all()
                    cluster_sources = [a.source.name for a in c_articles if a.source and a.source.name]
                await trigger_smart_article_notifications(article, cluster_sources=cluster_sources)
            except Exception as noti_err:
                print(f"Notification error: {noti_err}")

        # Update source health status
        source.status = "healthy"
        source.last_crawled_at = datetime.now(timezone.utc)
        source.last_error = None
        source.articles_count = (source.articles_count or 0) + articles_added
        await db.commit()

    except Exception as e:
        await db.rollback()
        source.status = "error"
        source.last_error = str(e)
        try:
            await db.commit()
        except Exception:
            pass
        raise e

    return articles_added


async def trigger_smart_article_notifications(article: Article, cluster_sources: List[str] = []):
    """
    Quy tắc gửi thông báo Telegram:
    Chỉ cần bài viết đạt relevance_score >= 7.5 là gửi thông báo ngay lập tức.
    """
    from app.services.telegram_service import (
        format_elite_article_message,
        format_cluster_alert_message,
        send_telegram_message,
    )

    # Nếu bài viết đạt từ 7.5 điểm AI trở lên -> Gửi thông báo ngay
    if (article.relevance_score or 0.0) >= 7.5:
        # Nếu bài viết thuộc sự kiện nóng được >= 3 nguồn cùng đưa tin
        unique_sources = list(dict.fromkeys(cluster_sources))
        if len(unique_sources) >= 3:
            cluster_data = {
                "title": article.vietnamese_title or article.title,
                "sources": unique_sources,
                "summary": article.vietnamese_summary or "Đang được nhiều nguồn uy tín đưa tin.",
                "url": article.url,
            }
            msg, markup = format_cluster_alert_message(cluster_data)
            await send_telegram_message(msg, reply_markup=markup)
            return

        # Mặc định: Gửi thông báo chi tiết bài viết tinh tuyển
        content_for_estimate = article.raw_content or article.vietnamese_summary or article.title
        reading_time = max(1, round(len(content_for_estimate.split()) / 200)) if content_for_estimate else 3

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
