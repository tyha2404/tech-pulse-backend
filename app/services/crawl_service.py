import datetime
import httpx
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Source, Article
from app.crawlers.article_crawler import (
    fetch_rss_feed,
    fetch_hacker_news,
    extract_clean_article_content,
    make_naive,
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

            pub_at = make_naive(item.get("published_at")) or datetime.datetime.utcnow()

            article = Article(
                source_id=source.id,
                title=item["title"],
                url=item["url"],
                author=item.get("author"),
                published_at=pub_at,
                raw_content=full_content,
                is_processed=False,
                created_at=datetime.datetime.utcnow(),
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
            since_time = datetime.datetime.utcnow() - datetime.timedelta(hours=48)
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
            await db.commit()  # Commit each article safely
            articles_added += 1

        # Update source health status
        source.status = "healthy"
        source.last_crawled_at = datetime.datetime.utcnow()
        source.last_error = None
        source.articles_count = (source.articles_count or 0) + articles_added
        await db.commit()

        # Webhook notifications if telegram or discord configured
        if articles_added > 0 and (
            settings.TELEGRAM_BOT_TOKEN or settings.DISCORD_WEBHOOK_URL
        ):
            await notify_webhook(
                f"🚀 TechPulse: Vừa cào được {articles_added} bài mới từ nguồn '{source.name}'!"
            )

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


async def notify_webhook(message: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
                tg_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
                await client.post(
                    tg_url, json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": message}
                )
            if settings.DISCORD_WEBHOOK_URL:
                await client.post(
                    settings.DISCORD_WEBHOOK_URL, json={"content": message}
                )
    except Exception:
        pass
