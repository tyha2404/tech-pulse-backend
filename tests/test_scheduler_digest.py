import pytest
from unittest.mock import AsyncMock, patch
import datetime
from app.models.models import Article, Source
from app.core.database import AsyncSessionLocal, Base, engine
from app.services.telegram_service import dispatch_daily_espresso_digest

@pytest.mark.asyncio
async def test_dispatch_daily_espresso_digest():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import uuid
    rand = str(uuid.uuid4())[:8]
    now = datetime.datetime.utcnow()
    async with AsyncSessionLocal() as db:
        src = Source(name=f"Digest Source {rand}", url=f"https://digest-{rand}.com/rss")
        db.add(src)
        await db.commit()
        await db.refresh(src)

        a1 = Article(
            source_id=src.id,
            title="Kiến trúc Microservices hiệu năng cao",
            vietnamese_title="Kiến trúc Microservices hiệu năng cao",
            url=f"https://digest-{rand}.com/1",
            relevance_score=9.5,
            is_worth_reading=True,
            is_canonical=True,
            published_at=now,
        )
        db.add(a1)
        await db.commit()

    with patch("app.services.telegram_service.send_telegram_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        sent = await dispatch_daily_espresso_digest("☕ Morning Tech Espresso (8:00 AM)")
        assert sent is True
        assert mock_send.called
        args, kwargs = mock_send.call_args
        assert "Morning Tech Espresso" in args[0]
        assert "Kiến trúc Microservices hiệu năng cao" in args[0]
