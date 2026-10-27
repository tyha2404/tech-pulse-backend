import logging
import asyncio
import inspect
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Any
import httpx
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.future import select
from sqlalchemy.orm import load_only
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Source, Article, CrawlRun
from app.crawlers.article_crawler import (
    fetch_rss_feed,
    fetch_json_feed,
    fetch_sitemap_feed,
    fetch_hacker_news,
    extract_clean_article_content,
    make_tz_aware,
)
from app.services.ai_analyzer import (
    EXTRACTIVE_FALLBACK_MODEL,
    analyze_article_with_9router,
)
from app.services.clustering_service import assign_article_cluster
from app.services.embedding_service import generate_article_embedding
from app.core.config import settings

logger = logging.getLogger(__name__)

# Prevents the scheduled crawl and a manual /crawl-all from running at the same time
crawl_lock = asyncio.Lock()

# Caps concurrent extract + LLM + embedding work across every source being crawled
_article_semaphore = asyncio.Semaphore(settings.AI_MAX_CONCURRENCY)


def estimate_reading_time(text: Optional[str]) -> int:
    """Average reading speed ~ 200 words per minute"""
    if not text:
        return 1
    return max(1, round(len(text.split()) / 200))


async def _fetch_source_items(source: Source) -> list[dict]:
    target_url = source.feed_url or source.url
    source_type = (source.source_type or "").lower()

    if source_type == "hn" or "ycombinator.com" in source.url:
        return await fetch_hacker_news()
    if source_type == "json":
        return await fetch_json_feed(target_url)
    if source_type == "sitemap":
        return await fetch_sitemap_feed(target_url)
    if source_type == "rss" or source.feed_url:
        return await fetch_rss_feed(target_url)
    # Fallback attempt: rss, then direct
    try:
        return await fetch_rss_feed(source.url)
    except Exception:
        return []


def apply_analysis(article: Article, analysis: Any, model_used: str) -> None:
    """Copies an AIAnalysisResult onto an Article."""
    article.is_processed = True
    article.is_worth_reading = analysis.is_worth_reading
    article.relevance_score = analysis.relevance_score
    article.vietnamese_title = analysis.vietnamese_title
    article.vietnamese_summary = analysis.vietnamese_summary
    article.key_takeaways = analysis.key_takeaways
    article.new_tech_stacks = [t.model_dump() for t in analysis.new_tech_stack]
    article.tags = analysis.tags
    article.target_audience = analysis.target_audience
    article.architectural_tradeoffs = (
        analysis.architectural_tradeoffs.model_dump()
        if analysis.architectural_tradeoffs
        else {}
    )
    article.nestjs_blueprint = (
        analysis.nestjs_blueprint.model_dump() if analysis.nestjs_blueprint else {}
    )
    article.learning_path = (
        analysis.learning_path.model_dump() if analysis.learning_path else {}
    )
    article.cluster_topic_key = analysis.cluster_topic_key
    article.ai_model_used = model_used


async def reanalyze_fallback_articles(batch_size: int = 40) -> int:
    """Re-runs the LLM on articles saved with the extractive fallback (gateway was down),
    which otherwise keep the flat 5.5 score and untranslated English text forever."""
    from app.core.database import AsyncSessionLocal

    async def analyze(article: Article):
        async with _article_semaphore:
            return await analyze_article_with_9router(
                title=article.title,
                content=article.raw_content or "",
                url=article.url,
            )

    fixed = 0
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Article)
            .where(Article.ai_model_used == EXTRACTIVE_FALLBACK_MODEL)
            .order_by(Article.id.desc())
            .limit(batch_size)
        )
        articles = list(res.scalars().all())
        chunk_size = max(1, settings.AI_MAX_CONCURRENCY)
        for i in range(0, len(articles), chunk_size):
            chunk = articles[i : i + chunk_size]
            results = await asyncio.gather(*(analyze(a) for a in chunk))
            chunk_fixed = 0
            for article, (analysis, model_used) in zip(chunk, results):
                if model_used == EXTRACTIVE_FALLBACK_MODEL:
                    continue
                apply_analysis(article, analysis, model_used)
                # Old vector came from the untranslated text; embedding_backfill redoes it
                article.embedding = None
                chunk_fixed += 1
            await db.commit()
            fixed += chunk_fixed
            if chunk_fixed == 0:
                # Gateway still down; retry on the next run instead of hammering it
                break
    if fixed:
        logger.info(f"Re-analyzed {fixed} articles that had used the extractive fallback")
    return fixed


async def _prepare_article(item: dict, source_id: int, run_ai: bool) -> Article:
    """Network-bound work for one feed item: body extraction, AI analysis, embedding.
    Touches no DB session, so many of these can run in parallel."""
    async with _article_semaphore:
        full_content = item.get("raw_content", "")
        if len(full_content) < 300:
            extracted = await extract_clean_article_content(item["url"])
            if extracted:
                full_content = extracted

        now_utc = datetime.now(timezone.utc)
        article = Article(
            source_id=source_id,
            title=item["title"],
            url=item["url"],
            author=item.get("author"),
            published_at=make_tz_aware(item.get("published_at")) or now_utc,
            raw_content=full_content,
            reading_time_minutes=estimate_reading_time(full_content or item["title"]),
            is_processed=False,
            created_at=now_utc,
        )

        if run_ai:
            # AI analysis with 9routers (multi-model fallback)
            analysis, model_used = await analyze_article_with_9router(
                title=article.title, content=full_content, url=article.url
            )
            apply_analysis(article, analysis, model_used)

            # Semantic search vector (Module 4). Left empty when the gateway is down so
            # backfill_missing_embeddings can fill it later instead of storing a fake vector.
            try:
                article.embedding = await generate_article_embedding(
                    article, allow_fallback=False
                )
            except Exception as emb_err:
                logger.warning(f"Embedding generation failed for {article.url}: {emb_err}")

        return article


async def _cluster_source_names(db: AsyncSession, cluster_id: Optional[str]) -> List[str]:
    if not cluster_id:
        return []
    res = await db.execute(
        select(Source.name)
        .join(Article, Article.source_id == Source.id)
        .where(Article.cluster_id == cluster_id)
    )
    return [name for name in res.scalars().all() if name]


async def crawl_single_source(
    source: Source, db: AsyncSession, run_ai: bool = True
) -> int:
    """Crawls a given source (RSS, JSON Feed, Sitemap, HN) and tracks CrawlRun metrics"""
    start_time = time.time()
    started_at = datetime.now(timezone.utc)
    # Captured up front: after a rollback `source` is expired, and reading its
    # attributes would trigger a lazy load that fails under AsyncSession
    source_id = source.id
    source_name = source.name
    source.status = "pending"
    await db.commit()

    articles_found = 0
    articles_added = 0
    error_message = None
    http_status = 200

    try:
        raw_items = await _fetch_source_items(source)
        articles_found = len(raw_items)

        # Dedupe against the DB in one query, and within the feed itself
        urls = list(dict.fromkeys(i["url"] for i in raw_items if i.get("url")))
        existing_urls = set()
        if urls:
            existing_res = await db.execute(
                select(Article.url).where(Article.url.in_(urls))
            )
            existing_urls = set(existing_res.scalars().all())
        new_items, seen = [], set(existing_urls)
        for item in raw_items:
            if item.get("url") and item["url"] not in seen:
                seen.add(item["url"])
                new_items.append(item)

        results = await asyncio.gather(
            *(_prepare_article(item, source_id, run_ai) for item in new_items),
            return_exceptions=True,
        )
        prepared = []
        for item, result in zip(new_items, results):
            if isinstance(result, BaseException):
                logger.error(f"Failed to process {item['url']}: {result}")
            else:
                prepared.append(result)

        # Story clustering candidates (48h window), loaded once per source with only
        # the columns assign_article_cluster needs
        if prepared:
            since_time = datetime.now(timezone.utc) - timedelta(hours=48)
            recent_res = await db.execute(
                select(Article)
                .options(
                    load_only(
                        Article.id,
                        Article.title,
                        Article.vietnamese_title,
                        Article.cluster_id,
                        Article.cluster_topic_key,
                        Article.relevance_score,
                        Article.is_canonical,
                    )
                )
                .where(Article.created_at >= since_time)
            )
            recent_candidates = list(recent_res.scalars().all())

        for article in prepared:
            cluster_id, is_canonical, demoted_id = assign_article_cluster(
                article, recent_candidates
            )
            article.cluster_id = cluster_id
            article.is_canonical = is_canonical

            try:
                async with db.begin_nested():
                    db.add(article)
                    # Set the relationship (not just source_id) so notifications can
                    # read article.source without an async lazy load
                    article.source = source
            except IntegrityError:
                # Same URL inserted concurrently by another source's crawl
                logger.info(f"Skipping duplicate article {article.url}")
                continue
            except DBAPIError as e:
                # A bad row (e.g. wrong embedding dimension) must not abort the whole source
                logger.error(f"Skipping article {article.url}: {e.orig}")
                continue
            if demoted_id:
                for cand in recent_candidates:
                    if cand.id == demoted_id:
                        cand.is_canonical = False
            await db.commit()
            articles_added += 1
            recent_candidates.append(article)

            # Smart & Urgent Telegram Notification Trigger (Module 3)
            try:
                cluster_sources = []
                relevance = article.relevance_score or 0.0
                if (
                    settings.TELEGRAM_INSTANT_MODE == "all"
                    and 7.5 <= relevance < settings.TELEGRAM_ALERT_THRESHOLD
                ):
                    cluster_sources = await _cluster_source_names(db, article.cluster_id)
                await trigger_smart_article_notifications(
                    article, cluster_sources=cluster_sources
                )
            except Exception as noti_err:
                logger.error(f"Notification error for {article.url}: {noti_err}")

        # Update source health status
        source.status = "healthy"
        source.last_crawled_at = datetime.now(timezone.utc)
        source.last_error = None
        source.articles_count = (source.articles_count or 0) + articles_added

        # Record successful CrawlRun
        duration_ms = int((time.time() - start_time) * 1000)
        crawl_run = CrawlRun(
            source_id=source_id,
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

    except asyncio.CancelledError:
        try:
            await db.rollback()
        except Exception:
            pass
        raise
    except Exception as e:
        await db.rollback()
        duration_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        logger.error(
            f"Lỗi khi cào nguồn '{source_name}' (id={source_id}): {error_message}"
        )
        source.status = "error"
        source.last_error = error_message

        status_code = 500
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
            status_code = e.response.status_code
        
        # Record failed CrawlRun
        crawl_run = CrawlRun(
            source_id=source_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            http_status=status_code,
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

    return articles_added


async def trigger_smart_article_notifications(
    article: Article, cluster_sources: List[str] = []
):
    """
    Quy tắc gửi thông báo Telegram:
    - Score >= 9.0: Gửi khẩn cấp Real-time Urgent Alert (Module 3)
    - Score >= 7.5: Gửi thông báo tin tuyển chọn hoặc cụm tin nóng
    """
    from app.services.telegram_service import (
        format_elite_article_message,
        format_cluster_alert_message,
        send_urgent_alert,
        send_telegram_message,
    )

    relevance = article.relevance_score or 0.0

    # Module 3: Real-time Urgent Alert for score >= 9.0
    if relevance >= settings.TELEGRAM_ALERT_THRESHOLD:
        await send_urgent_alert(article)
        return

    # In "urgent_only" mode everything below the urgent threshold waits for the
    # end-of-day briefing instead of pushing one message per article
    if settings.TELEGRAM_INSTANT_MODE != "all":
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


async def crawl_all_active_sources(
    concurrency_limit: int = 5,
    run_ai: bool = True,
    progress_callback: Optional[Any] = None,
) -> int:
    """
    Crawls all active sources concurrently using asyncio.Semaphore for safe throughput.
    Uses separate database sessions per task to avoid concurrency conflicts with SQLAlchemy.
    Skips the run entirely if another crawl-all is already in progress.
    """
    if crawl_lock.locked():
        logger.info("Crawl already in progress; skipping this run.")
        return 0
    async with crawl_lock:
        return await _crawl_all_active_sources(
            concurrency_limit, run_ai, progress_callback
        )


async def _crawl_all_active_sources(
    concurrency_limit: int, run_ai: bool, progress_callback: Optional[Any]
) -> int:
    from app.core.database import AsyncSessionLocal

    sem = asyncio.Semaphore(concurrency_limit)
    total_new_articles = 0

    try:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(Source).where(Source.is_active == True))
            sources = res.scalars().all()
            # Extract source IDs to process independently in concurrent sessions
            source_ids = [s.id for s in sources]
    except asyncio.CancelledError:
        logger.warning("⚠️ [Crawl Service] Cào tin bị huỷ trong lúc lấy danh sách nguồn.")
        raise

    async def crawl_worker(source_id: int) -> int:
        async with sem:
            async with AsyncSessionLocal() as session:
                try:
                    source = await session.get(Source, source_id)
                    if not source:
                        return 0
                    if progress_callback:
                        try:
                            if inspect.iscoroutinefunction(progress_callback):
                                await progress_callback(f"Đang cào nguồn: {source.name}...")
                            else:
                                progress_callback(f"Đang cào nguồn: {source.name}...")
                        except Exception as cb_err:
                            logger.warning(f"Progress callback error: {cb_err}")

                    return await crawl_single_source(source, session, run_ai=run_ai)
                except asyncio.CancelledError:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    raise
                except Exception as err:
                    logger.error(f"Lỗi cào nguồn (id={source_id}): {err}")
                    return 0

    tasks = [asyncio.create_task(crawl_worker(s_id)) for s_id in source_ids]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.warning("⚠️ [Crawl Service] Quá trình cào nhận tín hiệu huỷ. Đang dọn dẹp các task con an toàn...")
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for r in results:
        if isinstance(r, int):
            total_new_articles += r
        elif isinstance(r, asyncio.CancelledError):
            pass
        elif isinstance(r, Exception):
            logger.error(f"Task exception in crawl_all_active_sources: {r}")

    return total_new_articles

