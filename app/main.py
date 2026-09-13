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
    # 1. AI & Công nghệ tương lai (AI & Future Tech)
    {
        "name": "The Verge (AI News)",
        "url": "https://www.theverge.com/ai-artificial-intelligence",
        "feed_url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "MIT Technology Review (AI)",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence",
        "feed_url": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Google DeepMind Blog",
        "url": "https://deepmind.google/blog",
        "feed_url": "https://deepmind.google/blog/rss.xml",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "OpenAI News",
        "url": "https://openai.com/news",
        "feed_url": "https://openai.com/news/rss.xml",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Simon Willison Weblog (AI & MCP)",
        "url": "https://simonwillison.net",
        "feed_url": "https://simonwillison.net/atom/everything/",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Latent Space (AI Engineering)",
        "url": "https://www.latent.space",
        "feed_url": "https://www.latent.space/feed",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "LangChain Blog (Agentic & RAG)",
        "url": "https://blog.langchain.dev",
        "feed_url": "https://blog.langchain.dev/rss/",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Supabase Blog (Postgres & Vector)",
        "url": "https://supabase.com/blog",
        "feed_url": "https://supabase.com/rss.xml",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Vercel Blog (AI & Next.js)",
        "url": "https://vercel.com/blog",
        "feed_url": "https://vercel.com/atom",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Qdrant Vector DB Blog",
        "url": "https://qdrant.tech/blog",
        "feed_url": "https://qdrant.tech/blog/index.xml",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Weaviate Vector DB Blog",
        "url": "https://weaviate.io/blog",
        "feed_url": "https://weaviate.io/blog/rss.xml",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "Ollama Releases & Tooling",
        "url": "https://github.com/ollama/ollama",
        "feed_url": "https://github.com/ollama/ollama/releases.atom",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "MarkTechPost (AI News)",
        "url": "https://www.marktechpost.com",
        "feed_url": "https://www.marktechpost.com/feed/",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },
    {
        "name": "KDnuggets (AI & ML)",
        "url": "https://www.kdnuggets.com",
        "feed_url": "https://www.kdnuggets.com/feed",
        "source_type": "rss",
        "category": "AI & Future Tech",
    },

    # 2. Backend & Kiến trúc hệ thống (Backend & Architecture)
    {
        "name": "The Pragmatic Engineer (Gergely Orosz)",
        "url": "https://newsletter.pragmaticengineer.com",
        "feed_url": "https://newsletter.pragmaticengineer.com/feed",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "ByteByteGo (Alex Xu)",
        "url": "https://blog.bytebytego.com",
        "feed_url": "https://blog.bytebytego.com/feed",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Cloudflare Blog (Systems & Edge)",
        "url": "https://blog.cloudflare.com",
        "feed_url": "https://blog.cloudflare.com/rss/",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "GitHub Engineering Blog",
        "url": "https://github.blog/engineering",
        "feed_url": "https://github.blog/engineering/feed/",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Discord Engineering",
        "url": "https://discord.com/blog",
        "feed_url": "https://discord.com/blog/rss.xml",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Stripe Engineering",
        "url": "https://stripe.com/blog",
        "feed_url": "https://stripe.com/blog/feed.rss",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "AWS Architecture Blog",
        "url": "https://aws.amazon.com/blogs/architecture",
        "feed_url": "https://aws.amazon.com/blogs/architecture/feed/",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Netflix Tech Blog",
        "url": "https://netflixtechblog.com",
        "feed_url": "https://netflixtechblog.com/feed",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Hacker News (Top Stories)",
        "url": "https://news.ycombinator.com",
        "source_type": "hn",
        "category": "Backend & Architecture",
    },
    {
        "name": "Martin Fowler Blog",
        "url": "https://martinfowler.com",
        "feed_url": "https://martinfowler.com/feed.atom",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "The New Stack",
        "url": "https://thenewstack.io",
        "feed_url": "https://thenewstack.io/feed/",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Node Weekly",
        "url": "https://nodeweekly.com",
        "feed_url": "https://nodeweekly.com/rss",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "NestJS Releases (Kamil Mysliwiec)",
        "url": "https://github.com/nestjs/nest",
        "feed_url": "https://github.com/nestjs/nest/releases.atom",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Prisma Blog",
        "url": "https://www.prisma.io/blog",
        "feed_url": "https://www.prisma.io/blog/rss.xml",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "Planet PostgreSQL",
        "url": "https://planet.postgresql.org",
        "feed_url": "https://planet.postgresql.org/rss20.xml",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },
    {
        "name": "The Hacker News (Cybersecurity)",
        "url": "https://thehackernews.com",
        "feed_url": "https://feeds.feedburner.com/TheHackersNews",
        "source_type": "rss",
        "category": "Backend & Architecture",
    },

    # 3. Báo Công nghệ Quốc tế (Global Tech)
    {
        "name": "Ars Technica",
        "url": "https://arstechnica.com",
        "feed_url": "https://feeds.arstechnica.com/arstechnica/index",
        "source_type": "rss",
        "category": "Global Tech",
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

    # 4. Tin Công nghệ Việt Nam (Vietnam Tech)
    {
        "name": "VnExpress Số Hóa",
        "url": "https://vnexpress.net/so-hoa",
        "feed_url": "https://vnexpress.net/rss/so-hoa.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "Viblo (Cộng đồng Lập trình Việt Nam)",
        "url": "https://viblo.asia",
        "feed_url": "https://viblo.asia/rss",
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
    {
        "name": "Znews (Tri thức & Công nghệ)",
        "url": "https://znews.vn/cong-nghe",
        "feed_url": "https://znews.vn/rss/cong-nghe.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "Báo Thanh Niên (Công nghệ)",
        "url": "https://thanhnien.vn/cong-nghe",
        "feed_url": "https://thanhnien.vn/rss/cong-nghe.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "24h (Công nghệ thông tin)",
        "url": "https://www.24h.com.vn/cong-nghe-thong-tin-c55.html",
        "feed_url": "https://www.24h.com.vn/upload/rss/congnghethongtin.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "VietnamNet ICT & Công Nghệ",
        "url": "https://vietnamnet.vn/thong-tin-truyen-thong",
        "feed_url": "https://vietnamnet.vn/thong-tin-truyen-thong.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "Dân Trí (Sức Mạnh Số)",
        "url": "https://dantri.com.vn/suc-manh-so.htm",
        "feed_url": "https://dantri.com.vn/rss/cong-nghe.rss",
        "source_type": "rss",
        "category": "Vietnam Tech",
    },
    {
        "name": "Tuổi Trẻ Công Nghệ",
        "url": "https://tuoitre.vn/cong-nghe.htm",
        "feed_url": "https://tuoitre.vn/cong-nghe.rss",
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

        # Auto-migrate timestamp columns to TIMESTAMP WITH TIME ZONE (TIMESTAMPTZ)
        try:
            await conn.execute(text("ALTER TABLE articles ALTER COLUMN published_at TYPE TIMESTAMP WITH TIME ZONE USING published_at AT TIME ZONE 'UTC'"))
            await conn.execute(text("ALTER TABLE articles ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING created_at AT TIME ZONE 'UTC'"))
            await conn.execute(text("ALTER TABLE sources ALTER COLUMN last_crawled_at TYPE TIMESTAMP WITH TIME ZONE USING last_crawled_at AT TIME ZONE 'UTC'"))
            await conn.execute(text("ALTER TABLE sources ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING created_at AT TIME ZONE 'UTC'"))
            await conn.execute(text("ALTER TABLE sources ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE USING updated_at AT TIME ZONE 'UTC'"))
        except Exception as e:
            print(f"Timestamp migration notice: {e}")

        # Auto-migrate deep AI columns (Postgres JSON or SQLite TEXT/JSON)
        for col in ["architectural_tradeoffs", "nestjs_blueprint", "learning_path"]:
            try:
                await conn.execute(
                    text(
                        f"ALTER TABLE articles ADD COLUMN IF NOT EXISTS {col} JSON DEFAULT '{{}}'::json"
                    )
                )
            except Exception:
                try:
                    await conn.execute(
                        text(
                            f"ALTER TABLE articles ADD COLUMN IF NOT EXISTS {col} JSON DEFAULT '{{}}'"
                        )
                    )
                except Exception as e:
                    print(f"Migration notice for {col}: {e}")

        # Auto-migrate Story Clustering & Deduplication columns
        try:
            await conn.execute(text("ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_id VARCHAR(100)"))
            await conn.execute(text("ALTER TABLE articles ADD COLUMN IF NOT EXISTS is_canonical BOOLEAN DEFAULT TRUE"))
            await conn.execute(text("ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_topic_key VARCHAR(255)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_cluster_id ON articles(cluster_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_is_canonical ON articles(is_canonical)"))
        except Exception as e:
            print(f"Clustering migration notice: {e}")

    # Sync and seed default sources if any are missing
    async with AsyncSessionLocal() as db:
        existing_sources_res = await db.execute(select(Source.url))
        existing_urls = set(existing_sources_res.scalars().all())

        new_sources_added = 0
        for src_data in DEFAULT_SOURCES:
            if src_data["url"] not in existing_urls:
                source = Source(
                    name=src_data["name"],
                    url=src_data["url"],
                    feed_url=src_data.get("feed_url"),
                    source_type=src_data.get("source_type", "rss"),
                    category=src_data.get("category", "General"),
                    status="healthy",
                )
                db.add(source)
                existing_urls.add(src_data["url"])
                new_sources_added += 1

        if new_sources_added > 0:
            await db.commit()
            print(f"🌱 Đã tự động đồng bộ thêm {new_sources_added} nguồn tin công nghệ mới vào cơ sở dữ liệu!")

    # Start scheduler
    scheduler.add_job(
        scheduled_crawl_job, "interval", minutes=settings.CRAWL_INTERVAL_MINUTES
    )

    # Schedule Daily Espresso Briefings via Telegram
    from app.services.telegram_service import dispatch_daily_espresso_digest

    async def morning_espresso_cron():
        await dispatch_daily_espresso_digest("☕ Morning Tech Espresso (8:00 AM)")

    async def evening_briefing_cron():
        await dispatch_daily_espresso_digest("🌇 Evening Tech Briefing (18:00 PM)")

    scheduler.add_job(morning_espresso_cron, "cron", hour=8, minute=0, id="morning_espresso")
    scheduler.add_job(evening_briefing_cron, "cron", hour=18, minute=0, id="evening_briefing")

    scheduler.start()
    print(
        f"🚀 Scheduler started: Running crawl every {settings.CRAWL_INTERVAL_MINUTES} minutes, Espresso briefings at 08:00 & 18:00."
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
