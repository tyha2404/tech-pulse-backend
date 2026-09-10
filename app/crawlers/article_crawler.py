import asyncio
import httpx
import feedparser
import trafilatura
from datetime import datetime, timezone
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}


def make_naive(dt: datetime | None) -> datetime | None:
    """Ensure datetime is offset-naive UTC for PostgreSQL TIMESTAMP WITHOUT TIME ZONE"""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # Convert to UTC and strip tzinfo
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def fetch_rss_feed(feed_url: str) -> list[dict]:
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=True, headers=headers
    ) as client:
        resp = await client.get(feed_url)
        resp.raise_for_status()
        content = resp.text

    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Failed to parse RSS feed from {feed_url}")

    items = []
    for entry in parsed.entries[:15]:  # Limit to 15 most recent
        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        summary_text = getattr(entry, "summary", "")
        if summary_text:
            s_soup = BeautifulSoup(summary_text, "html.parser")
            summary_text = s_soup.get_text()

        items.append(
            {
                "title": getattr(entry, "title", "").strip(),
                "url": getattr(entry, "link", "").strip(),
                "author": getattr(entry, "author", None),
                "published_at": make_naive(pub_date),
                "raw_content": summary_text[:4000] if summary_text else "",
            }
        )
    return items


async def fetch_hacker_news() -> list[dict]:
    async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
        resp = await client.get(
            "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=15"
        )
        resp.raise_for_status()
        data = resp.json()

        items = []
        for hit in data.get("hits", []):
            url = (
                hit.get("url")
                or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            )

            raw_dt = None
            if hit.get("created_at"):
                try:
                    raw_dt = datetime.fromisoformat(
                        hit["created_at"].replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            items.append(
                {
                    "title": hit.get("title", ""),
                    "url": url,
                    "author": hit.get("author"),
                    "published_at": make_naive(raw_dt) or datetime.utcnow(),
                    "raw_content": hit.get("story_text", "") or hit.get("title", ""),
                }
            )
        return items


async def extract_clean_article_content(article_url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=5.0, follow_redirects=True, headers=headers
        ) as client:
            resp = await client.get(article_url)
            if resp.status_code == 200:
                extracted = trafilatura.extract(
                    resp.text, include_links=False, include_images=False
                )
                if extracted:
                    return extracted[:4000]
    except Exception:
        pass
    return ""
