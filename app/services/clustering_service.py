import re
import unicodedata
from typing import Set

# Common Vietnamese & English tech stopwords that don't differentiate news topics
STOPWORDS = {
    "va", "tai", "o", "cho", "trong", "tren", "voi", "cua", "la", "nhung", "cac",
    "nhung", "den", "tu", "ra", "vao", "se", "da", "dang", "mot", "nhieu", "moi",
    "and", "at", "in", "for", "with", "of", "is", "the", "a", "an", "on", "to"
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

    # Word-level token overlap (Jaccard)
    union_tokens = tokens1 | tokens2
    token_sim = (len(tokens1 & tokens2) / len(union_tokens)) if union_tokens else 0.0

    # 2-gram shingle overlap
    shingles1 = extract_shingles(norm1, n=2)
    shingles2 = extract_shingles(norm2, n=2)
    union_shingles = shingles1 | shingles2
    shingle_sim = (len(shingles1 & shingles2) / len(union_shingles)) if union_shingles else 0.0

    # Weighted blend: 60% token overlap + 40% sequence shingle overlap
    return 0.6 * token_sim + 0.4 * shingle_sim
