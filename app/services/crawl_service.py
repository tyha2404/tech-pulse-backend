import datetime
import httpx
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Source, Article
from app.crawlers.article_crawler import fetch_rss_feed, fetch_hacker_news, extract_clean_article_content
from app.services.ai_analyzer import analyze_article_with_9router
from app.core.config import settings

async def crawl_single_source(source: Source, db: AsyncSession, run_ai: bool = True) -> int:
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
            existing = await db.execute(select(Article).where(Article.url == item["url"]))
            if existing.scalar_one_or_none():
                continue

            # Extract full text if raw_content is short
            full_content = item.get("raw_content", "")
            if len(full_content) < 300:
                extracted = await extract_clean_article_content(item["url"])
                if extracted:
                    full_content = extracted

            article = Article(
                source_id=source.id,
                title=item["title"],
                url=item["url"],
                author=item.get("author"),
                published_at=item.get("published_at") or datetime.datetime.utcnow(),
                raw_content=full_content,
                is_processed=False
            )

            # AI analysis with 9routers
            if run_ai:
                analysis = await analyze_article_with_9router(
                    title=article.title,
                    content=full_content,
                    url=article.url
                )
                article.is_processed = True
                article.is_worth_reading = analysis.is_worth_reading
                article.relevance_score = analysis.relevance_score
                article.vietnamese_title = analysis.vietnamese_title
                article.vietnamese_summary = analysis.vietnamese_summary
                article.key_takeaways = analysis.key_takeaways
                article.new_tech_stacks = [t.model_dump() for t in analysis.new_tech_stack]
                article.tags = analysis.tags
                article.target_audience = analysis.target_audience
                article.ai_model_used = settings.AI_MODEL

            db.add(article)
            articles_added += 1

        # Update source health status
        source.status = "healthy"
        source.last_crawled_at = datetime.datetime.utcnow()
        source.last_error = None
        source.articles_count = (source.articles_count or 0) + articles_added
        await db.commit()

        # Webhook notifications if telegram or discord configured
        if articles_added > 0 and (settings.TELEGRAM_BOT_TOKEN or settings.DISCORD_WEBHOOK_URL):
            await notify_webhook(f"🚀 TechPulse: Vừa cào được {articles_added} bài mới từ nguồn '{source.name}'!")

    except Exception as e:
        source.status = "error"
        source.last_error = str(e)
        await db.commit()
        raise e

    return articles_added

async def notify_webhook(message: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
                tg_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
                await client.post(tg_url, json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": message})
            if settings.DISCORD_WEBHOOK_URL:
                await client.post(settings.DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception:
        pass
