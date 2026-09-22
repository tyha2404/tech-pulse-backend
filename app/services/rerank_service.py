import httpx
from typing import List, Tuple, Optional
from app.core.config import settings
from app.models.models import Article


def _heuristic_persona_scoring(article: Article, persona: str) -> float:
    """Fallback heuristic scorer matching persona keywords with tags and target audience."""
    base_score = float(article.relevance_score or 5.0)
    persona_lower = persona.lower()
    
    tags = [t.lower() for t in (article.tags or [])]
    audiences = [a.lower() for a in (article.target_audience or [])]
    combined_meta = " ".join(tags + audiences + [article.title.lower()])

    boost = 0.0
    persona_keywords = [w for w in persona_lower.replace("/", " ").replace("-", " ").split() if len(w) > 2]
    for kw in persona_keywords:
        if kw in combined_meta:
            boost += 1.0

    return min(10.0, base_score + boost)


async def rerank_articles_for_persona(
    articles: List[Article], persona: str
) -> List[Tuple[Article, float]]:
    """
    Reranks a list of candidate articles for a specific engineering persona using
    TypeSafe AI Jev System One Score primitives.
    
    Returns a list of (Article, personalized_score) sorted in descending order of relevance.
    """
    if not articles:
        return []

    # Limit to top 15 candidates for fast System One scoring budget
    candidates = articles[:15]
    remaining = articles[15:]

    if not settings.TYPESAFE_API_KEY:
        # Heuristic fallback if TypeSafe key is not configured
        scored = [(a, _heuristic_persona_scoring(a, persona)) for a in articles]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    endpoint = f"{settings.TYPESAFE_BASE_URL.rstrip('/')}/systemone"
    headers = {
        "Authorization": f"Bearer {settings.TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }

    state_candidates = []
    questions = {}

    for i, a in enumerate(candidates):
        title = a.vietnamese_title or a.title
        snippet = (a.vietnamese_summary or a.raw_content or "")[:250]
        state_candidates.append({
            "id": a.id,
            "title": title,
            "tags": a.tags or [],
            "target_audience": a.target_audience or [],
            "snippet": snippet,
        })
        questions[f"score_{i}"] = {
            "type": "score",
            "instructions": f"How relevant, actionable, and valuable is candidate `candidates[{i}]` for an engineer with persona `target_persona`?",
            "criteria": [
                "Irrelevant or noise for this engineering persona",
                "Minor general interest but not actionable",
                "Moderately useful technical concept or update",
                "Highly actionable and directly applicable to daily work",
                "Must-read technical breakthrough, mission-critical incident, or core framework release",
            ],
        }

    payload = {
        "model": "jev-latest",
        "state": {
            "target_persona": persona,
            "candidates": state_candidates,
        },
        "questions": questions,
    }

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                answers = data.get("answers", {})

                scored_candidates = []
                for i, a in enumerate(candidates):
                    ans_obj = answers.get(f"score_{i}", {})
                    # TypeSafe score is 0.0 to 4.0
                    raw_score = (
                        float(ans_obj.get("score", 2.0))
                        if isinstance(ans_obj, dict)
                        else float(ans_obj or 2.0)
                    )
                    base_relevance = float(a.relevance_score or 5.0)
                    # 40% Base General Relevance + 60% Persona Calibrated Fit
                    personalized_score = (base_relevance * 0.4) + ((raw_score / 4.0) * 6.0)
                    scored_candidates.append((a, round(personalized_score, 2)))

                # Add remaining articles with base heuristic score
                for rem in remaining:
                    scored_candidates.append((rem, _heuristic_persona_scoring(rem, persona)))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return scored_candidates
            else:
                print(f"[TypeSafe Rerank Notice] HTTP {resp.status_code}: {resp.text}")
    except Exception as err:
        print(f"[TypeSafe Rerank Error] {err}")

    # Safe fallback on any error
    scored = [(a, _heuristic_persona_scoring(a, persona)) for a in articles]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
