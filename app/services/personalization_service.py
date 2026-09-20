from typing import List, Dict, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import desc
from app.models.models import UserPreference, Article, Source, ArticleFeedback


async def get_or_create_user_preferences(
    user_id: str, db: AsyncSession
) -> UserPreference:
    stmt = select(UserPreference).where(UserPreference.user_id == user_id)
    res = await db.execute(stmt)
    pref = res.scalar_one_or_none()
    if not pref:
        pref = UserPreference(
            user_id=user_id,
            topic_weights={"AI": 1.5, "Backend": 1.5, "Architecture": 1.2, "PostgreSQL": 1.2},
            preferred_sources=[],
        )
        db.add(pref)
        await db.commit()
        await db.refresh(pref)
    return pref


async def update_user_preferences(
    user_id: str,
    topic_weights: Optional[Dict[str, float]],
    preferred_sources: Optional[List[int]],
    db: AsyncSession,
) -> UserPreference:
    pref = await get_or_create_user_preferences(user_id, db)
    if topic_weights is not None:
        pref.topic_weights = topic_weights
    if preferred_sources is not None:
        pref.preferred_sources = preferred_sources
    await db.commit()
    await db.refresh(pref)
    return pref


async def get_personalized_feed(
    user_id: str, db: AsyncSession, limit: int = 30
) -> List[Article]:
    """
    Reranks canonical articles based on:
    - Base AI score (40%)
    - User topic preference matching (35%)
    - Preferred source affinity (25%)
    """
    pref = await get_or_create_user_preferences(user_id, db)
    topic_weights = pref.topic_weights or {}
    preferred_sources = set(pref.preferred_sources or [])

    # Fetch candidate articles
    stmt = (
        select(Article)
        .options(selectinload(Article.source))
        .where(
            Article.is_canonical == True,
            Article.is_hidden == False,
        )
        .order_by(desc(Article.id))
        .limit(100)
    )
    res = await db.execute(stmt)
    candidates = res.scalars().all()

    scored_articles = []
    for art in candidates:
        base_score = float(art.relevance_score or 5.0)  # scale 1-10

        # Topic matching boost
        topic_multiplier = 1.0
        art_tags = [t.lower() for t in (art.tags or [])]
        for topic, weight in topic_weights.items():
            if topic.lower() in art_tags or topic.lower() in (art.title or "").lower():
                topic_multiplier = max(topic_multiplier, float(weight))

        # Source affinity boost
        source_boost = 1.3 if art.source_id in preferred_sources else 1.0

        # Blended personalized score
        personalized_score = base_score * topic_multiplier * source_boost
        scored_articles.append((personalized_score, art))

    # Sort descending by personalized score
    scored_articles.sort(key=lambda x: x[0], reverse=True)
    return [art for _, art in scored_articles[:limit]]
