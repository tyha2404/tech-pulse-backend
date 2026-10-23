import hashlib
import logging
import random
import math
from typing import List, Optional
from app.core.config import settings
from app.models.models import Article
from app.services.ai_analyzer import get_ai_client

logger = logging.getLogger(__name__)


def _pseudo_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Pure Python deterministic fallback embedding generator when remote embedding model
    gateway is unavailable or not configured.
    """
    if not text:
        return [0.0] * dim
    
    # Hash seed
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    
    # Normalize to unit vector for cosine similarity
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


async def generate_text_embedding(
    text: str, allow_fallback: bool = True
) -> Optional[List[float]]:
    """Generates a 1536-dimensional embedding vector for a given text.

    When the gateway is unavailable, returns a deterministic pseudo-embedding if
    allow_fallback is True, otherwise None. Pseudo-embeddings carry no semantic
    meaning, so callers that persist or rank by similarity should pass False.
    """
    if not text or not text.strip():
        return [0.0] * 1536 if allow_fallback else None

    clean_text = text.strip()[:4000]

    try:
        client = get_ai_client(15.0)
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=clean_text,
        )
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Embedding gateway unavailable: {e}")

    return _pseudo_embedding(clean_text, dim=1536) if allow_fallback else None


async def generate_article_embedding(
    article: Article, allow_fallback: bool = True
) -> Optional[List[float]]:
    """Builds a rich contextual representation of the article and embeds it"""
    title = article.title or ""
    vn_title = article.vietnamese_title or ""
    summary = article.vietnamese_summary or ""
    tags = " ".join(article.tags or [])
    takeaways = " ".join(article.key_takeaways or [])

    rich_text = f"{title}\n{vn_title}\n{summary}\n{tags}\n{takeaways}"
    return await generate_text_embedding(rich_text, allow_fallback=allow_fallback)


async def backfill_missing_embeddings(batch_size: int = 50) -> int:
    """Embeds processed articles that were saved without an embedding (gateway was down)."""
    from sqlalchemy import select
    from sqlalchemy.orm import load_only
    from app.core.database import AsyncSessionLocal

    filled = 0
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Article)
            .options(
                load_only(
                    Article.id,
                    Article.title,
                    Article.vietnamese_title,
                    Article.vietnamese_summary,
                    Article.tags,
                    Article.key_takeaways,
                )
            )
            .where(Article.embedding.is_(None), Article.is_processed == True)
            .order_by(Article.id.desc())
            .limit(batch_size)
        )
        for article in res.scalars().all():
            vec = await generate_article_embedding(article, allow_fallback=False)
            if vec is None:
                # Gateway still down; retry on the next run instead of hammering it
                break
            article.embedding = vec
            filled += 1
        await db.commit()
    if filled:
        logger.info(f"Backfilled embeddings for {filled} articles")
    return filled
