from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from app.models.models import ArticleFeedback, Article, Source


async def record_article_feedback(
    article_id: int,
    feedback_type: str,
    user_id: str = "default_user",
    source: str = "web",
    notes: Optional[str] = None,
    db: AsyncSession = None,
) -> ArticleFeedback:
    """Save user interaction feedback (like, dislike, bookmark, read, hide)"""
    # Check if existing feedback exists for this article & user
    stmt = select(ArticleFeedback).where(
        ArticleFeedback.article_id == article_id,
        ArticleFeedback.user_id == user_id,
    )
    res = await db.execute(stmt)
    fb = res.scalar_one_or_none()

    if fb:
        fb.feedback_type = feedback_type
        fb.source = source
        if notes:
            fb.notes = notes
    else:
        fb = ArticleFeedback(
            article_id=article_id,
            user_id=user_id,
            feedback_type=feedback_type,
            source=source,
            notes=notes,
        )
        db.add(fb)

    # If bookmark or hide, also synchronize the flag on Article model
    article = await db.get(Article, article_id)
    if article:
        if feedback_type == "bookmark":
            article.is_bookmarked = True
        elif feedback_type == "hide":
            article.is_hidden = True

    await db.commit()
    await db.refresh(fb)
    return fb


async def get_calibration_examples(db: AsyncSession, limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve notable likes/dislikes to dynamically calibrate AI prompt few-shots"""
    stmt = (
        select(ArticleFeedback)
        .options(selectinload(ArticleFeedback.article))
        .where(ArticleFeedback.feedback_type.in_(["like", "dislike"]))
        .order_by(ArticleFeedback.id.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    feedbacks = res.scalars().all()

    examples = []
    for f in feedbacks:
        if f.article:
            examples.append(
                {
                    "title": f.article.vietnamese_title or f.article.title,
                    "type": f.feedback_type,
                    "note": f.notes or f"User marked as {f.feedback_type}",
                }
            )
    return examples
