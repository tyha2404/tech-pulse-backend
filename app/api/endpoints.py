import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from sqlalchemy.orm import selectinload
from typing import List, Optional

from app.core.database import get_db, AsyncSessionLocal
from app.models.models import Source, Article
from app.schemas.schemas import (
    SourceCreate,
    SourceResponse,
    ArticleResponse,
    ArticleUpdate,
    CrawlTestResult,
    ReaderModeResponse,
    ArticleChatRequest,
    ArticleChatResponse,
    RelatedArticleItem,
    WeeklyRadarDigestResponse,
)
from app.crawlers.feed_discoverer import discover_feed_url
from app.crawlers.article_crawler import (
    fetch_rss_feed,
    fetch_hacker_news,
    extract_clean_article_content,
)
from app.services.crawl_service import crawl_single_source
from app.services.ai_analyzer import (
    analyze_article_with_9router,
    chat_with_article,
    generate_weekly_radar_digest,
)
from app.core.config import settings

router = APIRouter()

# Global crawl state
crawl_status = {"is_crawling": False, "progress": "", "total_new_articles": 0}


async def run_crawl_all_task():
    global crawl_status
    if crawl_status["is_crawling"]:
        return
    crawl_status["is_crawling"] = True
    crawl_status["progress"] = "Đang khởi động cào..."
    crawl_status["total_new_articles"] = 0

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Source).where(Source.is_active == True))
            sources = result.scalars().all()
            for s in sources:
                crawl_status["progress"] = f"Đang cào nguồn: {s.name}..."
                try:
                    added = await crawl_single_source(s, db, run_ai=True)
                    crawl_status["total_new_articles"] += added
                except Exception as err:
                    print(f"Lỗi cào nguồn {s.name}: {err}")
            crawl_status["progress"] = (
                f"Hoàn thành! Đã thu thập {crawl_status['total_new_articles']} bài mới."
            )
    finally:
        crawl_status["is_crawling"] = False


# ----------------- SOURCES ENDPOINTS -----------------


@router.get("/sources", response_model=List[SourceResponse])
async def list_sources(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Source).order_by(Source.id))
    return result.scalars().all()


@router.post("/sources/test", response_model=CrawlTestResult)
async def test_source_url(
    target_url: str = Query(..., description="URL website or RSS"),
):
    try:
        detected_type, feed_or_url = await discover_feed_url(target_url)
        if detected_type == "hn":
            items = await fetch_hacker_news()
        else:
            items = await fetch_rss_feed(feed_or_url)

        sample_titles = [it["title"] for it in items[:5]]
        return CrawlTestResult(
            success=True,
            detected_type=detected_type,
            feed_url=feed_or_url if detected_type == "rss" else None,
            items_count=len(items),
            sample_titles=sample_titles,
        )
    except Exception as e:
        return CrawlTestResult(
            success=False,
            detected_type="unknown",
            feed_url=None,
            items_count=0,
            sample_titles=[],
            error=str(e),
        )


@router.post("/sources", response_model=SourceResponse)
async def create_source(payload: SourceCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(Source).where(Source.url == payload.url))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400, detail="Nguồn tin này đã tồn tại trong hệ thống!"
        )

    feed_url = payload.feed_url
    source_type = payload.source_type
    if not feed_url:
        detected_type, discovered = await discover_feed_url(payload.url)
        source_type = detected_type
        if detected_type == "rss":
            feed_url = discovered

    new_source = Source(
        name=payload.name,
        url=payload.url,
        feed_url=feed_url,
        source_type=source_type,
        category=payload.category,
        is_active=payload.is_active,
        status="healthy",
    )
    db.add(new_source)
    await db.commit()
    await db.refresh(new_source)
    return new_source


@router.delete("/sources/{source_id}")
async def delete_source(source_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Source).where(Source.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Không tìm thấy nguồn tin")

    await db.delete(source)
    await db.commit()
    return {"message": f"Đã xóa thành công nguồn tin '{source.name}'"}


@router.post("/sources/{source_id}/crawl")
async def trigger_crawl_source(
    source_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Source).where(Source.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Không tìm thấy nguồn tin")

    background_tasks.add_task(crawl_single_source, source, db, True)
    return {"message": f"Đã kích hoạt cào nguồn '{source.name}' trong nền!"}


# ----------------- ARTICLES ENDPOINTS -----------------


def calculate_reading_time(text: Optional[str]) -> int:
    if not text:
        return 1
    # Average reading speed ~ 200 words per minute
    words = len(text.split())
    minutes = max(1, round(words / 200))
    return minutes


@router.get("/articles", response_model=List[ArticleResponse])
async def list_articles(
    source_id: Optional[int] = None,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    top_only: bool = False,
    query: Optional[str] = None,
    sort_by: str = Query("newest", pattern="^(newest|oldest|score)$"),
    read_status: str = Query("all", pattern="^(all|unread|read)$"),
    bookmarked_only: bool = False,
    include_hidden: bool = False,
    group_duplicates: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Article).options(selectinload(Article.source)).join(Source, isouter=True)
    )

    # Source filter
    if source_id:
        stmt = stmt.where(Article.source_id == source_id)

    # Bookmarked only filter
    if bookmarked_only:
        stmt = stmt.where(Article.is_bookmarked == True)

    # Hidden filter: by default exclude hidden articles; if include_hidden=True, show only hidden articles
    if not include_hidden:
        stmt = stmt.where(Article.is_hidden == False)
    else:
        stmt = stmt.where(Article.is_hidden == True)

    # Story clustering filter: if group_duplicates=True, only show canonical or unclustered articles
    if group_duplicates:
        stmt = stmt.where(
            (Article.is_canonical == True) | (Article.cluster_id.is_(None))
        )

    # Read status filter
    if read_status == "unread":
        stmt = stmt.where(Article.is_read == False)
    elif read_status == "read":
        stmt = stmt.where(Article.is_read == True)

    if top_only:
        stmt = stmt.where(Article.relevance_score >= 7.5)
    if category and category != "all":
        stmt = stmt.where(Source.category.ilike(f"%{category}%"))
    if query:
        stmt = stmt.where(
            Article.title.ilike(f"%{query}%")
            | Article.vietnamese_title.ilike(f"%{query}%")
        )

    # Sorting
    if sort_by == "oldest":
        stmt = stmt.order_by(Article.published_at.asc().nulls_last(), Article.id.asc())
    elif sort_by == "score":
        stmt = stmt.order_by(
            Article.relevance_score.desc(),
            Article.published_at.desc().nulls_last(),
            Article.id.desc(),
        )
    else:  # newest
        stmt = stmt.order_by(
            Article.published_at.desc().nulls_last(), Article.id.desc()
        )

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    articles = result.scalars().all()

    # If grouping duplicates, eagerly fetch non-canonical articles for the current page's clusters
    cluster_ids = [art.cluster_id for art in articles if art.cluster_id]
    related_map = {}
    if group_duplicates and cluster_ids:
        rel_stmt = (
            select(Article)
            .options(selectinload(Article.source))
            .where(
                Article.cluster_id.in_(cluster_ids),
                Article.is_canonical == False,
            )
        )
        rel_res = await db.execute(rel_stmt)
        rel_articles = rel_res.scalars().all()
        from app.schemas.schemas import RelatedSourceArticle

        for rel in rel_articles:
            item = RelatedSourceArticle(
                id=rel.id,
                title=rel.title,
                source_name=rel.source.name if rel.source else "Nguồn khác",
                url=rel.url,
                published_at=rel.published_at,
                vietnamese_title=rel.vietnamese_title,
            )
            related_map.setdefault(rel.cluster_id, []).append(item)

    output = []
    for art in articles:
        resp = ArticleResponse.model_validate(art)
        if art.source:
            resp.source_name = art.source.name
        if art.cluster_id and art.cluster_id in related_map:
            resp.related_articles = related_map[art.cluster_id]
        # Estimate reading time from raw_content or summary
        content_for_estimate = art.raw_content or art.vietnamese_summary or art.title
        resp.reading_time_minutes = calculate_reading_time(content_for_estimate)
        output.append(resp)

    return output


@router.patch("/articles/{article_id}", response_model=ArticleResponse)
async def update_article(
    article_id: int,
    payload: ArticleUpdate = Body(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài viết")

    if payload.is_read is not None:
        article.is_read = payload.is_read
    if payload.is_hidden is not None:
        article.is_hidden = payload.is_hidden
    if payload.is_bookmarked is not None:
        article.is_bookmarked = payload.is_bookmarked

    await db.commit()
    await db.refresh(article)

    resp = ArticleResponse.model_validate(article)
    if article.source:
        resp.source_name = article.source.name
    content_for_estimate = (
        article.raw_content or article.vietnamese_summary or article.title
    )
    resp.reading_time_minutes = calculate_reading_time(content_for_estimate)
    return resp


@router.get("/articles/{article_id}/reader", response_model=ReaderModeResponse)
async def get_article_reader_content(
    article_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài viết")

    # Automatically mark as read when reading
    if not article.is_read:
        article.is_read = True
        await db.commit()
        await db.refresh(article)

    # Try extracting clean article text
    extracted_text = ""
    try:
        extracted_text = await extract_clean_article_content(
            article.url, max_length=25000, output_format="txt"
        )
    except Exception:
        pass

    # Fallback to stored raw_content or vietnamese_summary if extractor finds nothing
    final_content = (
        extracted_text
        if extracted_text and len(extracted_text) > 100
        else (article.raw_content or article.vietnamese_summary or article.title)
    )

    reading_time = calculate_reading_time(final_content)

    return ReaderModeResponse(
        id=article.id,
        title=article.title,
        vietnamese_title=article.vietnamese_title,
        url=article.url,
        author=article.author,
        source_name=article.source.name if article.source else None,
        published_at=article.published_at,
        reading_time_minutes=reading_time,
        content=final_content,
        is_bookmarked=article.is_bookmarked,
    )


@router.post("/articles/{article_id}/summarize", response_model=ArticleResponse)
async def summarize_article_on_demand(
    article_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài viết")

    # Fetch fresh content if raw_content is very short
    content_to_analyze = article.raw_content or ""
    if len(content_to_analyze) < 200:
        fresh_clean = await extract_clean_article_content(article.url, max_length=5000)
        if fresh_clean:
            content_to_analyze = fresh_clean
            article.raw_content = fresh_clean

    analysis = await analyze_article_with_9router(
        article.title, content_to_analyze, article.url
    )

    article.is_processed = True
    article.is_worth_reading = analysis.is_worth_reading
    article.relevance_score = analysis.relevance_score
    article.vietnamese_title = analysis.vietnamese_title
    article.vietnamese_summary = analysis.vietnamese_summary
    article.key_takeaways = analysis.key_takeaways
    article.new_tech_stacks = [ts.model_dump() for ts in analysis.new_tech_stack]
    article.tags = analysis.tags
    article.target_audience = analysis.target_audience
    if analysis.architectural_tradeoffs:
        article.architectural_tradeoffs = analysis.architectural_tradeoffs.model_dump()
    if analysis.nestjs_blueprint:
        article.nestjs_blueprint = analysis.nestjs_blueprint.model_dump()
    if analysis.learning_path:
        article.learning_path = analysis.learning_path.model_dump()
    article.ai_model_used = settings.AI_MODEL

    await db.commit()
    await db.refresh(article)

    resp = ArticleResponse.model_validate(article)
    if article.source:
        resp.source_name = article.source.name
    resp.reading_time_minutes = calculate_reading_time(
        article.raw_content or article.vietnamese_summary or article.title
    )
    return resp


@router.post("/articles/{article_id}/chat", response_model=ArticleChatResponse)
async def chat_with_article_endpoint(
    article_id: int,
    payload: ArticleChatRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài viết")

    content_for_chat = (
        article.raw_content
        or (
            f"{article.vietnamese_summary or ''}\n\nTakeaways: {', '.join(article.key_takeaways or [])}"
        )
        or article.title
    )

    chat_res = await chat_with_article(
        article_title=article.vietnamese_title or article.title,
        article_content=content_for_chat,
        user_message=payload.message,
        history=payload.history,
    )

    return ArticleChatResponse(
        reply=chat_res.get("reply", ""),
        suggested_followups=chat_res.get("suggested_followups", []),
    )


@router.get("/articles/{article_id}/related", response_model=List[RelatedArticleItem])
async def get_related_articles(
    article_id: int,
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài viết")

    current_tags = set([t.lower() for t in (article.tags or [])])
    current_stacks = set(
        [
            s.get("name", "").lower()
            for s in (article.new_tech_stacks or [])
            if isinstance(s, dict)
        ]
    )
    combined_topics = current_tags.union(current_stacks)

    # Fetch candidate articles (not this article)
    candidates_result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(
            Article.id != article_id,
            (Article.is_hidden == False) | (Article.is_hidden.is_(None)),
        )
        .order_by(Article.id.desc())
        .limit(200)
    )
    candidates = candidates_result.scalars().all()

    scored = []
    for cand in candidates:
        cand_tags = set([t.lower() for t in (cand.tags or [])])
        cand_stacks = set(
            [
                s.get("name", "").lower()
                for s in (cand.new_tech_stacks or [])
                if isinstance(s, dict)
            ]
        )
        cand_topics = cand_tags.union(cand_stacks)

        overlap = len(combined_topics.intersection(cand_topics))
        if overlap > 0 or not combined_topics:
            scored.append((overlap, cand.relevance_score, cand))

    # Sort primarily by topic overlap, secondarily by relevance score
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_picks = [item[2] for item in scored[:limit]]

    return [
        RelatedArticleItem(
            id=a.id,
            title=a.title,
            vietnamese_title=a.vietnamese_title,
            relevance_score=a.relevance_score,
            tags=a.tags or [],
            source_name=a.source.name if a.source else None,
        )
        for a in top_picks
    ]


# In-memory cache for Radar Digest: {"data": WeeklyRadarDigestResponse, "expires_at": float}
_radar_cache = {"data": None, "expires_at": 0.0}


@router.get("/intelligence/radar-digest", response_model=WeeklyRadarDigestResponse)
async def get_weekly_radar_digest_endpoint(
    force_refresh: bool = False,
    db: AsyncSession = Depends(get_db),
):
    import time

    global _radar_cache
    now = time.time()

    # Return cached result if still valid and not force refreshing
    if (
        not force_refresh
        and _radar_cache["data"] is not None
        and now < _radar_cache["expires_at"]
    ):
        return _radar_cache["data"]

    # Select top articles from the database
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.source))
        .where(Article.is_hidden == False)
        .order_by(
            Article.relevance_score.desc(), Article.published_at.desc().nulls_last()
        )
        .limit(15)
    )
    articles = result.scalars().all()

    articles_payload = []
    top_items = []
    for a in articles:
        articles_payload.append(
            {
                "title": a.title,
                "vietnamese_title": a.vietnamese_title,
                "vietnamese_summary": a.vietnamese_summary,
                "tags": a.tags or [],
                "source_name": a.source.name if a.source else None,
            }
        )
        top_items.append(
            RelatedArticleItem(
                id=a.id,
                title=a.title,
                vietnamese_title=a.vietnamese_title,
                relevance_score=a.relevance_score,
                tags=a.tags or [],
                source_name=a.source.name if a.source else None,
            )
        )

    digest_data = await generate_weekly_radar_digest(articles_payload)

    response = WeeklyRadarDigestResponse(
        week_label=digest_data.get("week_label", "Báo cáo Radar Công nghệ Tuần"),
        dominant_trends=digest_data.get("dominant_trends", []),
        architectural_shifts=digest_data.get("architectural_shifts", []),
        actionable_recommendations=digest_data.get("actionable_recommendations", []),
        top_articles=top_items[:6],
    )

    # Cache for 60 minutes (3600 seconds)
    _radar_cache["data"] = response
    _radar_cache["expires_at"] = now + 3600.0

    return response


@router.post("/intelligence/dispatch-digest")
async def dispatch_digest_to_webhook(
    channel: str = Query("all", pattern="^(all|telegram|discord)$"),
    db: AsyncSession = Depends(get_db),
):
    """Optionally dispatch weekly radar digest to Telegram or Discord if configured."""
    # Placeholder for webhook notification with status check
    configured_channels = []
    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        configured_channels.append("telegram")
    if settings.DISCORD_WEBHOOK_URL:
        configured_channels.append("discord")

    return {
        "success": True,
        "message": f"Đã chuẩn bị bản tin radar cho kênh: {channel}",
        "configured_channels": configured_channels,
    }


@router.post("/crawl-all")
async def trigger_crawl_all(background_tasks: BackgroundTasks):
    global crawl_status
    if crawl_status["is_crawling"]:
        return {"message": "Tiến trình cào đang chạy", "status": crawl_status}

    background_tasks.add_task(run_crawl_all_task)
    return {
        "message": "Đã bắt đầu tiến trình cào tất cả nguồn tin trong nền!",
        "status": "started",
    }


@router.get("/crawl-status")
async def get_crawl_status():
    global crawl_status
    return crawl_status


@router.get("/settings")
async def get_system_settings():
    return {
        "ninerouters_base_url": settings.NINEROUTERS_BASE_URL,
        "ai_model": settings.AI_MODEL,
        "crawl_interval_minutes": settings.CRAWL_INTERVAL_MINUTES,
        "telegram_configured": bool(
            settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID
        ),
        "discord_configured": bool(settings.DISCORD_WEBHOOK_URL),
    }
