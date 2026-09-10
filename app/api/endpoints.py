import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
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
    CrawlTestResult,
)
from app.crawlers.feed_discoverer import discover_feed_url
from app.crawlers.article_crawler import fetch_rss_feed, fetch_hacker_news
from app.services.crawl_service import crawl_single_source
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


@router.get("/articles", response_model=List[ArticleResponse])
async def list_articles(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    top_only: bool = False,
    query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Article)
        .options(selectinload(Article.source))
        .join(Source, isouter=True)
        .order_by(desc(Article.published_at), desc(Article.id))
    )

    if top_only:
        stmt = stmt.where(Article.relevance_score >= 7.5)
    if category and category != "all":
        stmt = stmt.where(Source.category.ilike(f"%{category}%"))
    if query:
        stmt = stmt.where(
            Article.title.ilike(f"%{query}%")
            | Article.vietnamese_title.ilike(f"%{query}%")
        )

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    articles = result.scalars().all()

    output = []
    for art in articles:
        resp = ArticleResponse.model_validate(art)
        if art.source:
            resp.source_name = art.source.name
        output.append(resp)

    return output


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
