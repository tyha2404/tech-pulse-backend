import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import settings
from app.core.database import engine, Base, AsyncSessionLocal
from app.models.models import Source
from app.api.endpoints import router as api_router
from app.services.crawl_service import crawl_single_source
from sqlalchemy.future import select

# Default authoritative sources (Global + Vietnam)
DEFAULT_SOURCES = [
    {"name": "Hacker News (Top Stories)", "url": "https://news.ycombinator.com", "source_type": "hn", "category": "Backend & Tech"},
    {"name": "TechCrunch", "url": "https://techcrunch.com", "feed_url": "https://techcrunch.com/feed/", "source_type": "rss", "category": "Global Tech"},
    {"name": "The Verge", "url": "https://theverge.com", "feed_url": "https://www.theverge.com/rss/index.xml", "source_type": "rss", "category": "Global Tech"},
    {"name": "ArXiv Computer Science (AI)", "url": "https://arxiv.org", "feed_url": "http://export.arxiv.org/rss/cs.AI", "source_type": "rss", "category": "AI Research"},
    {"name": "Martin Fowler Blog", "url": "https://martinfowler.com", "feed_url": "https://martinfowler.com/feed.atom", "source_type": "rss", "category": "Backend Architecture"},
    {"name": "VnExpress Số Hóa", "url": "https://vnexpress.net/so-hoa", "feed_url": "https://vnexpress.net/rss/so-hoa.rss", "source_type": "rss", "category": "Vietnam Tech"},
    {"name": "Tinh Tế", "url": "https://tinhte.vn", "feed_url": "https://tinhte.vn/rss", "source_type": "rss", "category": "Vietnam Tech"},
    {"name": "GenK", "url": "https://genk.vn", "feed_url": "https://genk.vn/rss/home.rss", "source_type": "rss", "category": "Vietnam Tech"},
]

scheduler = AsyncIOScheduler()

async def scheduled_crawl_job():
    print("⏰ [Scheduler] Running background crawl cycle...")
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Source).where(Source.is_active == True))
        sources = result.scalars().all()
        for s in sources:
            try:
                await crawl_single_source(s, db, run_ai=True)
            except Exception as e:
                print(f"Error crawling {s.name}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Seed default sources if empty
    async with AsyncSessionLocal() as db:
        count_res = await db.execute(select(Source))
        if not count_res.scalars().first():
            for src_data in DEFAULT_SOURCES:
                source = Source(
                    name=src_data["name"],
                    url=src_data["url"],
                    feed_url=src_data.get("feed_url"),
                    source_type=src_data.get("source_type", "rss"),
                    category=src_data.get("category", "General"),
                    status="healthy"
                )
                db.add(source)
            await db.commit()
            print("🌱 Initialized default tech news sources successfully!")

    # Start scheduler
    scheduler.add_job(scheduled_crawl_job, 'interval', minutes=settings.CRAWL_INTERVAL_MINUTES)
    scheduler.start()
    print(f"🚀 Scheduler started: Running every {settings.CRAWL_INTERVAL_MINUTES} minutes.")

    yield

    scheduler.shutdown()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Automated Tech News Aggregator & 9routers AI Analyzer",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "TechPulse Backend API is active", "docs": "/docs"}
