import re
import unicodedata
from typing import Set, Optional, Tuple
import httpx
from app.core.config import settings

# Common Vietnamese & English tech stopwords that don't differentiate news topics
STOPWORDS = {
    "va",
    "tai",
    "o",
    "cho",
    "trong",
    "tren",
    "voi",
    "cua",
    "la",
    "nhung",
    "cac",
    "nhung",
    "den",
    "tu",
    "ra",
    "vao",
    "se",
    "da",
    "dang",
    "mot",
    "nhieu",
    "moi",
    "and",
    "at",
    "in",
    "for",
    "with",
    "of",
    "is",
    "the",
    "a",
    "an",
    "on",
    "to",
}


def normalize_vietnamese_text(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text.strip().lower())
    # Remove accents for resilient matching
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tokens(text: str, remove_stopwords: bool = True) -> Set[str]:
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return set(words)


def extract_shingles(text: str, n: int = 2) -> Set[str]:
    words = [w for w in text.split() if w not in STOPWORDS]
    if not words:
        return set()
    if len(words) < n:
        return {text}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_title_similarity(title1: str, title2: str) -> float:
    norm1 = normalize_vietnamese_text(title1)
    norm2 = normalize_vietnamese_text(title2)
    if not norm1 or not norm2:
        return 0.0
    if norm1 == norm2:
        return 1.0

    tokens1 = extract_tokens(norm1)
    tokens2 = extract_tokens(norm2)

    if not tokens1 or not tokens2:
        return 0.0

    # Overlap Coefficient: len(intersection) / min(len(tokens1), len(tokens2))
    # This detects when one headline is a focused variation of another headline
    intersection = tokens1 & tokens2
    overlap_coef = len(intersection) / min(len(tokens1), len(tokens2))

    # Jaccard Token similarity
    union_tokens = tokens1 | tokens2
    token_jaccard = len(intersection) / len(union_tokens)

    # 2-gram shingle overlap
    shingles1 = extract_shingles(norm1, n=2)
    shingles2 = extract_shingles(norm2, n=2)
    union_shingles = shingles1 | shingles2
    shingle_sim = (
        (len(shingles1 & shingles2) / len(union_shingles)) if union_shingles else 0.0
    )

    # Score: 50% Overlap coef + 30% Token Jaccard + 20% Shingle sim
    return 0.5 * overlap_coef + 0.3 * token_jaccard + 0.2 * shingle_sim


def assign_article_cluster(
    new_article,
    recent_candidates,
    similarity_threshold: float = 0.45,
):
    """
    Determines if new_article belongs to an existing story cluster from recent_candidates.
    Returns:
        (cluster_id: str, is_canonical: bool, demoted_article_id: Optional[int])
    """
    import uuid

    best_match = None
    best_similarity = 0.0

    new_title = new_article.vietnamese_title or new_article.title
    new_topic_key = getattr(new_article, "cluster_topic_key", None)
    new_score = getattr(new_article, "relevance_score", 0.0) or 0.0

    for candidate in recent_candidates:
        candidate_cluster_id = getattr(candidate, "cluster_id", None)
        if not candidate_cluster_id:
            continue

        cand_topic_key = getattr(candidate, "cluster_topic_key", None)
        # 1. Exact match on non-empty cluster_topic_key
        if new_topic_key and cand_topic_key and new_topic_key == cand_topic_key:
            best_match = candidate
            best_similarity = 1.0
            break

        # 2. Similarity on title
        cand_title = getattr(candidate, "vietnamese_title", None) or getattr(
            candidate, "title", ""
        )
        sim = compute_title_similarity(new_title, cand_title)
        if sim >= similarity_threshold and sim > best_similarity:
            best_similarity = sim
            best_match = candidate

    if best_match:
        cluster_id = best_match.cluster_id
        # Find current canonical in this cluster if best_match is not canonical or to compare scores
        cand_score = getattr(best_match, "relevance_score", 0.0) or 0.0

        # If new article is significantly higher quality (> 0.5 or strictly higher if equal), promote it
        if new_score > cand_score:
            return (
                cluster_id,
                True,
                best_match.id if getattr(best_match, "is_canonical", False) else None,
            )
        else:
            return cluster_id, False, None

    # No match found -> start brand new cluster with unique cluster_id and is_canonical=True
    return str(uuid.uuid4()), True, None


async def verify_same_story_with_typesafe(title_a: str, title_b: str) -> Optional[bool]:
    """
    Calls TypeSafe Jev System One with a Noul primitive to resolve borderline
    similarity between two news headlines (0.40 <= sim <= 0.65).
    """
    if not settings.TYPESAFE_API_KEY:
        return None

    endpoint = f"{settings.TYPESAFE_BASE_URL.rstrip('/')}/systemone"
    headers = {
        "Authorization": f"Bearer {settings.TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jev-latest",
        "state": {
            "headline_a": title_a,
            "headline_b": title_b,
        },
        "questions": {
            "is_same_story": {
                "type": "noul",
                "instructions": "Do `headline_a` and `headline_b` report on the exact same underlying software engineering release, incident, or technical news event?",
            }
        },
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                raw_ans = data.get("answers", {}).get("is_same_story", {})
                p_same = float(raw_ans.get("noul", 0.0)) if isinstance(raw_ans, dict) else float(raw_ans or 0.0)
                return p_same >= 0.65
    except Exception as err:
        print(f"[TypeSafe Clustering Resolver Notice] {err}")
    return None


async def assign_article_cluster_async(
    new_article,
    recent_candidates,
    similarity_threshold: float = 0.45,
) -> Tuple[str, bool, Optional[int]]:
    """
    Async variant with TypeSafe Jev Noul semantic disambiguation for borderline scores.
    """
    import uuid

    best_match = None
    best_similarity = 0.0

    new_title = new_article.vietnamese_title or new_article.title
    new_topic_key = getattr(new_article, "cluster_topic_key", None)
    new_score = getattr(new_article, "relevance_score", 0.0) or 0.0

    borderline_candidates = []

    for candidate in recent_candidates:
        candidate_cluster_id = getattr(candidate, "cluster_id", None)
        if not candidate_cluster_id:
            continue

        cand_topic_key = getattr(candidate, "cluster_topic_key", None)
        # 1. Exact match on non-empty cluster_topic_key
        if new_topic_key and cand_topic_key and new_topic_key == cand_topic_key:
            best_match = candidate
            best_similarity = 1.0
            break

        # 2. Similarity on title
        cand_title = getattr(candidate, "vietnamese_title", None) or getattr(
            candidate, "title", ""
        )
        sim = compute_title_similarity(new_title, cand_title)
        if sim >= similarity_threshold and sim > best_similarity:
            best_similarity = sim
            best_match = candidate
        elif 0.35 <= sim < similarity_threshold:
            borderline_candidates.append((sim, candidate, cand_title))

    # If no definite heuristic match, check top borderline candidate with TypeSafe Jev
    if not best_match and borderline_candidates and settings.TYPESAFE_API_KEY:
        borderline_candidates.sort(key=lambda x: x[0], reverse=True)
        _, top_cand, cand_title = borderline_candidates[0]
        confirmed = await verify_same_story_with_typesafe(new_title, cand_title)
        if confirmed:
            best_match = top_cand

    if best_match:
        cluster_id = best_match.cluster_id
        cand_score = getattr(best_match, "relevance_score", 0.0) or 0.0
        if new_score > cand_score:
            return (
                cluster_id,
                True,
                best_match.id if getattr(best_match, "is_canonical", False) else None,
            )
        else:
            return cluster_id, False, None

    return str(uuid.uuid4()), True, None
