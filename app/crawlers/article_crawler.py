import httpx
import feedparser
import trafilatura
from datetime import datetime
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}

async def fetch_rss_feed(feed_url: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        resp = await client.get(feed_url)
        resp.raise_for_status()
        content = resp.text

    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Failed to parse RSS feed from {feed_url}")

    items = []
    for entry in parsed.entries[:25]:  # Limit to 25 most recent
        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        summary_text = getattr(entry, "summary", "")
        # Clean HTML in summary
        if summary_text:
            s_soup = BeautifulSoup(summary_text, "html.parser")
            summary_text = s_soup.get_text()

        items.append({
            "title": getattr(entry, "title", "").strip(),
            "url": getattr(entry, "link", "").strip(),
            "author": getattr(entry, "author", None),
            "published_at": pub_date,
            "raw_content": summary_text[:4000] if summary_text else ""
        })
    return items

async def fetch_hacker_news() -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        resp.raise_for_status()
        story_ids = resp.json()[:25]
        
        items = []
        for sid in story_ids:
            try:
                item_resp = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                if item_resp.status_code == 200:
                    data = item_resp.json()
                    if data and data.get("type") == "story" and data.get("url"):
                        items.append({
                            "title": data.get("title", ""),
                            "url": data.get("url", ""),
                            "author": data.get("by"),
                            "published_at": datetime.fromtimestamp(data.get("time", 0)) if data.get("time") else None,
                            "raw_content": data.get("text", "") or ""
                        })
            except Exception:
                continue
        return items

async def extract_clean_article_content(article_url: str) -> str:
    """Uses trafilatura to extract clean markdown/text content from article URL"""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(article_url)
            if resp.status_code == 200:
                extracted = trafilatura.extract(resp.text, include_links=False, include_images=False)
                if extracted:
                    return extracted[:6000] # Return up to 6000 chars for AI analysis
    except Exception:
        pass
    return ""
