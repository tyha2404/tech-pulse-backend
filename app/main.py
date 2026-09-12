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
    {
        "name": "Hacker News (Top Stories)",
        "url": "https://news.ycombinator.com",
        "source_type": "hn",
        "category": "Backend & Tech",
    },
    {
        "name": "ByteByteGo (Alex Xu)",
        "url": "https://blog.bytebytego.com",
        "feed_url": "https://blog.bytebytego.com/feed",
        "source_type": "rss",
        "category": "Backend Architecture",
    },
    {
        "name": "Node Weekly",
        "url": "https://nodeweekly.com",
        "feed_url": "https://nodeweekly.com/rss",
        "source_type": "rss",
        "category": "Node & NestJS",
    },
    {
        "name": "NestJS Releases (Kamil Mysliwiec)",
        "url": "https://github.com/nestjs/nest",
        "feed_url": "https://github.com/nestjs/nest/releases.atom",
        "source_type": "rss",
        "category": "Node & NestJS",
    },
    {
        "name": "Prisma Blog",
        "url": "https://www.prisma.io/blog",
        "feed_url": "https://www.prisma.io/blog/rss.xml",
        "source_type": "rss",
        "category": "Backend & Database",
    },
    {
        "name": "Lilian Weng (LilLog)",
        "url": "https://lilianweng.github.io",
        "feed_url": "https://lilianweng.github.io/index.xml",
        "source_type": "rss",
        "category": "AI Research & Agents",
    },
    {
        "name": "Latent Space (AI Engineering)",
        "url": "https://www.latent.space",
        "feed_url": "https://www.latent.space/feed",
        "source_type": "rss",
        "category": "AI Engineering",
    },
    {
        "name": "Vercel Blog (AI & TypeScript)",
        "url": "https://vercel.com/blog",
        "feed_url": "https://vercel.com/atom",
        "source_type": "rss",
        "category": "AI Engineering",
    },
    {
        "name": "Qdrant Vector DB Blog",
        "url": "https://qdrant.tech/blog",
        "feed_url": "https://qdrant.tech/blog/index.xml",
        "source_type": "rss",
        "category": "AI & Vector DB",
    },
    {
        "name": "Ollama Releases & Tooling",
        "url": "https://github.com/ollama/ollama",
        "feed_url": "https://github.com/ollama/ollama/releases.atom",
        "source_type": "rss",
        "category": "Local LLM & Inference",
    },
    {
        "name": "Hugging Face Blog",
        "url": "https://huggingface.co/blog",
        "feed_url": "https://huggingface.co/blog/feed.xml",
        "source_type": "rss",
        "category": "AI Research & Models",
    },
    {
        "name": "OpenAI News",
        "url": "https://openai.com/news",
        "feed_url": "https://openai.com/news/rss.xml",
        "source_type": "rss",
        "category": "AI Engineering",
    },
    {
        "name": "Simon Willison Weblog (AI & MCP)",
        "url": "https://simonwillison.net",
        "feed_url": "https://simonwillison.net/atom/everything/",
        "source_type": "rss",
        "category": "AI Engineering",
    },
    {
        "name": "The New Stack",
        "url": "https://thenewstack.io",
        "feed_url": "https://thenewstack.io/feed/",
        "source_type": "rss",
        "category": "Backend Architecture",
    },
    {
        "name": "Martin Fowler Blog",
        "url": "https://martinfowler.com",
        "feed_url": "https://martinfowler.com/feed.atom",
        "source_type": "rss",
        "category": "Backend Architecture",
    },
    {
        "name": "Netflix Tech Blog",
        "url": "https://netflixtechblog.com",
        "feed_url": "https://netflixtechblog.com/feed",
        "source_type": "rss",
        "category": "Backend Architecture",
    },
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com",
        "feed_url": "https://techcrunch.com/feed/",
        "source_type": "rss",
        "category": "Global Tech",
    },
    {
        "name": "The Verge",
        "url": "https://theverge.com",
        "feed_url": "https://www.theverge.com/rss/index.xml",
        "source_type": "rss",
        "category": "Global Tech",
    },
    {
        "name": "ArXiv Computer Science (AI)",
        "url": "https://arxiv.org",
        "feed_url": "http://export.arxiv.org/rss/cs.AI",
        "source_type": "rss",
        "category": "AI Research",
    },
    {
        "name": "VnExpress Số Hóa",
        "url": "https://vnexpress.net/so-hoa",
        "feed_url": "https://vnexpress.net/rss/so-hoa.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "Tinh Tế",
        "url": "https://tinhte.vn",
        "feed_url": "https://tinhte.vn/rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "GenK",
        "url": "https://genk.vn",
        "feed_url": "https://genk.vn/rss/home.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
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
        # Migrate new columns if they do not exist
        from sqlalchemy import text

        for col in ["is_read", "is_hidden", "is_bookmarked"]:
            try:
                await conn.execute(
                    text(
                        f"ALTER TABLE articles ADD COLUMN IF NOT EXISTS {col} BOOLEAN DEFAULT FALSE"
                    )
                )
            except Exception as e:
                print(f"Migration notice for {col}: {e}")

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
                    status="healthy",
                )
                db.add(source)
            await db.commit()
            print("🌱 Initialized default tech news sources successfully!")

    # Start scheduler
    scheduler.add_job(
        scheduled_crawl_job, "interval", minutes=settings.CRAWL_INTERVAL_MINUTES
    )
    scheduler.start()
    print(
        f"🚀 Scheduler started: Running every {settings.CRAWL_INTERVAL_MINUTES} minutes."
    )

    yield

    scheduler.shutdown()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Automated Tech News Aggregator & 9routers AI Analyzer",
    lifespan=lifespan,
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
