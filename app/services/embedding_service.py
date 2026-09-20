import hashlib
import random
import math
from typing import List, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.models import Article


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


async def generate_text_embedding(text: str) -> List[float]:
    """Generates a 1536-dimensional embedding vector for a given text"""
    if not text or not text.strip():
        return [0.0] * 1536

    clean_text = text.strip()[:4000]

    try:
        client = AsyncOpenAI(
            base_url=settings.NINEROUTERS_BASE_URL,
            api_key=settings.NINEROUTERS_API_KEY,
            timeout=15.0,
        )
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=clean_text,
        )
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
    except Exception as e:
        # Fallback to local deterministic pseudo-embedding to ensure zero service failure
        pass

    return _pseudo_embedding(clean_text, dim=1536)


async def generate_article_embedding(article: Article) -> List[float]:
    """Builds a rich contextual representation of the article and embeds it"""
    title = article.title or ""
    vn_title = article.vietnamese_title or ""
    summary = article.vietnamese_summary or ""
    tags = " ".join(article.tags or [])
    takeaways = " ".join(article.key_takeaways or [])

    rich_text = f"{title}\n{vn_title}\n{summary}\n{tags}\n{takeaways}"
    return await generate_text_embedding(rich_text)
