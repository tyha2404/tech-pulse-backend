import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import settings
from app.core.database import engine, Base, AsyncSessionLocal
from app.models.models import Source
from app.api.endpoints import router as api_router
from app.services.crawl_service import crawl_single_source, crawl_all_active_sources
from sqlalchemy import text
from sqlalchemy.future import select

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

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
        "url": "https://www.langchain.com/blog",
        "feed_url": "https://www.langchain.com/blog/rss.xml",
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
    logger.info("⏰ [Scheduler] Running background crawl cycle...")
    try:
        total = await crawl_all_active_sources(concurrency_limit=5, run_ai=True)
        logger.info(f"⏰ [Scheduler] Hoàn tất chu kỳ cào: đã thu thập {total} bài mới.")
    except asyncio.CancelledError:
        logger.info("⏰ [Scheduler] Scheduled crawl job đã được huỷ an toàn (CancelledError caught).")
        return
    except Exception as e:
        logger.error(f"⏰ [Scheduler] Error during scheduled crawl: {e}")



async def _try_ddl(conn, sql: str, label: str) -> bool:
    """Run one DDL statement inside a savepoint. On Postgres a failed statement aborts
    the enclosing transaction, so without the savepoint every later migration fails too."""
    try:
        async with conn.begin_nested():
            await conn.execute(text(sql))
        return True
    except Exception as e:
        logger.warning(f"Migration notice ({label}): {e}")
        return False


async def _column_type(conn, table: str, column: str) -> Optional[str]:
    res = await conn.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    )
    return res.scalar()


async def _run_schema_migrations(conn):
    """Idempotent hand-written migrations (no Alembic). Add new model columns here."""
    for col in ["is_read", "is_hidden", "is_bookmarked"]:
        await _try_ddl(
            conn,
            f"ALTER TABLE articles ADD COLUMN IF NOT EXISTS {col} BOOLEAN DEFAULT FALSE",
            col,
        )

    # Convert legacy timestamp columns to TIMESTAMPTZ. Only when needed: ALTER TYPE
    # rewrites the whole table under an exclusive lock.
    for table, col in [
        ("articles", "published_at"),
        ("articles", "created_at"),
        ("sources", "last_crawled_at"),
        ("sources", "created_at"),
        ("sources", "updated_at"),
    ]:
        try:
            current_type = await _column_type(conn, table, col)
        except Exception as e:
            logger.warning(f"Migration notice (type of {table}.{col}): {e}")
            continue
        if current_type == "timestamp without time zone":
            await _try_ddl(
                conn,
                f"ALTER TABLE {table} ALTER COLUMN {col} TYPE TIMESTAMP WITH TIME ZONE "
                f"USING {col} AT TIME ZONE 'UTC'",
                f"{table}.{col} timestamptz",
            )

    for col in ["architectural_tradeoffs", "nestjs_blueprint", "learning_path"]:
        await _try_ddl(
            conn,
            f"ALTER TABLE articles ADD COLUMN IF NOT EXISTS {col} JSON DEFAULT '{{}}'::json",
            col,
        )

    # Story Clustering & Deduplication columns
    await _try_ddl(conn, "ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_id VARCHAR(100)", "cluster_id")
    await _try_ddl(conn, "ALTER TABLE articles ADD COLUMN IF NOT EXISTS is_canonical BOOLEAN DEFAULT TRUE", "is_canonical")
    await _try_ddl(conn, "ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_topic_key VARCHAR(255)", "cluster_topic_key")
    await _try_ddl(conn, "CREATE INDEX IF NOT EXISTS idx_articles_cluster_id ON articles(cluster_id)", "idx_articles_cluster_id")

    # pgvector and embedding column
    await _try_ddl(conn, "CREATE EXTENSION IF NOT EXISTS vector", "pgvector extension")
    await _try_ddl(conn, "ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding vector(1536)", "embedding")
    await _try_ddl(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING hnsw (embedding vector_cosine_ops)",
        "idx_articles_embedding",
    )

    # Precomputed reading time, so list endpoints don't need to load raw_content
    if await _try_ddl(
        conn,
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS reading_time_minutes INTEGER",
        "reading_time_minutes",
    ):
        await _try_ddl(
            conn,
            """UPDATE articles SET reading_time_minutes = GREATEST(1, ROUND(COALESCE(
                   array_length(regexp_split_to_array(trim(
                       COALESCE(NULLIF(raw_content, ''), NULLIF(vietnamese_summary, ''), title)
                   ), '\\s+'), 1), 0) / 200.0))
               WHERE reading_time_minutes IS NULL""",
            "reading_time_minutes backfill",
        )

    # Indexes for the feed's filters and sort orders
    await _try_ddl(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_articles_feed_newest "
        "ON articles (is_hidden, created_at DESC, id DESC)",
        "idx_articles_feed_newest",
    )
    await _try_ddl(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_articles_feed_score "
        "ON articles (is_hidden, relevance_score DESC, created_at DESC)",
        "idx_articles_feed_score",
    )
    await _try_ddl(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles (source_id)",
        "idx_articles_source_id",
    )
    await _try_ddl(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_crawl_runs_source_started "
        "ON crawl_runs (source_id, started_at DESC)",
        "idx_crawl_runs_source_started",
    )

    # Trigram indexes so ILIKE '%q%' search can use an index instead of a full scan
    if await _try_ddl(conn, "CREATE EXTENSION IF NOT EXISTS pg_trgm", "pg_trgm extension"):
        for col in ["title", "vietnamese_title", "vietnamese_summary"]:
            await _try_ddl(
                conn,
                f"CREATE INDEX IF NOT EXISTS idx_articles_{col}_trgm "
                f"ON articles USING gin ({col} gin_trgm_ops)",
                f"idx_articles_{col}_trgm",
            )


async def scheduled_embedding_backfill_job():
    from app.services.embedding_service import backfill_missing_embeddings

    try:
        await backfill_missing_embeddings()
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.error(f"⏰ [Scheduler] Embedding backfill error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_schema_migrations(conn)

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
            logger.info(
                f"🌱 Đã tự động đồng bộ thêm {new_sources_added} nguồn tin công nghệ mới vào cơ sở dữ liệu!"
            )

    # Start scheduler
    scheduler.add_job(
        scheduled_crawl_job, "interval", minutes=settings.CRAWL_INTERVAL_MINUTES
    )
    # Re-embed articles saved while the embedding gateway was unavailable
    scheduler.add_job(
        scheduled_embedding_backfill_job,
        "interval",
        minutes=settings.EMBEDDING_BACKFILL_INTERVAL_MINUTES,
        id="embedding_backfill",
    )

    # Schedule Daily Espresso Briefings via Telegram
    from app.services.telegram_service import dispatch_daily_espresso_digest

    async def morning_espresso_cron():
        try:
            await dispatch_daily_espresso_digest("☕ Morning Tech Espresso (8:00 AM)")
        except asyncio.CancelledError:
            logger.info("⏰ [Scheduler] Morning espresso cron cancelled safely.")
        except Exception as e:
            logger.error(f"⏰ [Scheduler] Morning espresso error: {e}")

    async def evening_briefing_cron():
        try:
            await dispatch_daily_espresso_digest("🌇 Evening Tech Briefing (18:00 PM)")
        except asyncio.CancelledError:
            logger.info("⏰ [Scheduler] Evening briefing cron cancelled safely.")
        except Exception as e:
            logger.error(f"⏰ [Scheduler] Evening briefing error: {e}")

    scheduler.add_job(
        morning_espresso_cron, "cron", hour=8, minute=0, id="morning_espresso"
    )
    scheduler.add_job(
        evening_briefing_cron, "cron", hour=18, minute=0, id="evening_briefing"
    )

    scheduler.start()
    logger.info(
        f"🚀 Scheduler started: Running crawl every {settings.CRAWL_INTERVAL_MINUTES} minutes, Espresso briefings at 08:00 & 18:00."
    )

    yield

    # Graceful shutdown of scheduler and database connection pool
    logger.info("🛑 [Lifespan] Shutting down scheduler...")
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("🛑 [Lifespan] Scheduler stopped successfully.")
    except Exception as e:
        logger.error(f"🛑 [Lifespan] Error shutting down scheduler: {e}")

    try:
        from app.crawlers.article_crawler import close_http_client
        from app.services.ai_analyzer import close_ai_clients

        await close_http_client()
        await close_ai_clients()
    except Exception as e:
        logger.error(f"🛑 [Lifespan] Error closing HTTP clients: {e}")

    try:
        logger.info("🛑 [Lifespan] Disposing database connection pool...")
        await engine.dispose()
        logger.info("🛑 [Lifespan] Database connection pool disposed successfully.")
    except Exception as e:
        logger.error(f"🛑 [Lifespan] Error disposing database engine: {e}")


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
