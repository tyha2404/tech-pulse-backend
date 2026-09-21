import asyncio
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from pgvector.sqlalchemy import Vector

# Allow pgvector Vector type to compile gracefully to TEXT on SQLite
@compiles(Vector, "sqlite")
def compile_vector_sqlite(type_, compiler, **kw):
    return "TEXT"

import app.core.database as db_module
from app.core.database import Base, get_db
from app.main import app

# Create in-memory SQLite engine for async testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestAsyncSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)

# Monkey-patch database module at import time
db_module.engine = test_engine
db_module.AsyncSessionLocal = TestAsyncSessionLocal

# Also patch any imported references in services
import app.services.telegram_service as tg_module
tg_module.AsyncSessionLocal = TestAsyncSessionLocal


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    db_module.engine = test_engine
    db_module.AsyncSessionLocal = TestAsyncSessionLocal
    tg_module.AsyncSessionLocal = TestAsyncSessionLocal

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Override get_db dependency in FastAPI app
    async def override_get_db():
        async with TestAsyncSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
