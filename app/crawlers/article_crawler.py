import asyncio
import httpx
import feedparser
import trafilatura
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from urllib.parse import urlparse

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}


def make_tz_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetime is offset-aware UTC for TIMESTAMP WITH TIME ZONE"""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=timezone.utc)


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
                "published_at": make_tz_aware(pub_date),
                "raw_content": summary_text[:4000] if summary_text else "",
            }
        )
    return items


async def fetch_json_feed(feed_url: str) -> list[dict]:
    """Parse JSON Feed specification (RFC: version, items)"""
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=True, headers=headers
    ) as client:
        resp = await client.get(feed_url)
        resp.raise_for_status()
        data = resp.json()

    items = []
    raw_items = data.get("items", []) if isinstance(data, dict) else []
    for item in raw_items[:15]:
        title = item.get("title") or ""
        url = item.get("url") or item.get("id") or ""
        author = item.get("author", {}).get("name") if isinstance(item.get("author"), dict) else item.get("author")
        
        pub_date = None
        date_str = item.get("date_published") or item.get("date_modified")
        if date_str:
            try:
                pub_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                pass

        content_text = item.get("content_text") or item.get("content_html") or item.get("summary") or ""
        if content_text and "<" in content_text:
            s_soup = BeautifulSoup(content_text, "html.parser")
            content_text = s_soup.get_text()

        items.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "author": author,
                "published_at": make_tz_aware(pub_date) or datetime.now(timezone.utc),
                "raw_content": content_text[:4000] if content_text else "",
            }
        )
    return items


async def fetch_sitemap_feed(sitemap_url: str) -> list[dict]:
    """Parse sitemap.xml or sitemapindex.xml to extract recent article URLs"""
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=True, headers=headers
    ) as client:
        resp = await client.get(sitemap_url)
        resp.raise_for_status()
        xml_text = resp.text

    soup = BeautifulSoup(xml_text, "html.parser")
    items = []
    
    # Check for urlset (<url><loc>...</loc><lastmod>...</lastmod></url>)
    url_tags = soup.find_all("url")
    if not url_tags:
        # Might be sitemapindex
        sitemap_tags = soup.find_all("sitemap")
        if sitemap_tags:
            # fetch the first child sitemap
            first_loc = sitemap_tags[0].find("loc")
            if first_loc and first_loc.get_text():
                return await fetch_sitemap_feed(first_loc.get_text().strip())

    for u in url_tags[:15]:
        loc_tag = u.find("loc")
        if not loc_tag or not loc_tag.get_text():
            continue
        url = loc_tag.get_text().strip()
        lastmod_tag = u.find("lastmod")
        pub_date = None
        if lastmod_tag and lastmod_tag.get_text():
            try:
                pub_date = datetime.fromisoformat(lastmod_tag.get_text().strip().replace("Z", "+00:00"))
            except Exception:
                pass
        
        # Extract title from slug as initial title placeholder
        parsed_path = urlparse(url).path.strip("/").split("/")[-1].replace("-", " ").replace("_", " ").title()
        title = parsed_path if parsed_path else url

        items.append(
            {
                "title": title,
                "url": url,
                "author": None,
                "published_at": make_tz_aware(pub_date) or datetime.now(timezone.utc),
                "raw_content": "",
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
                    "published_at": make_tz_aware(raw_dt) or datetime.now(timezone.utc),
                    "raw_content": hit.get("story_text", "") or hit.get("title", ""),
                }
            )
        return items


async def extract_clean_article_content(
    article_url: str, max_length: int = 4000, output_format: str = "txt"
) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, headers=headers
        ) as client:
            resp = await client.get(article_url)
            if resp.status_code == 200:
                extracted = trafilatura.extract(
                    resp.text,
                    include_links=True,
                    include_images=False,
                    output_format=output_format,
                )
                if extracted:
                    return extracted[:max_length]
    except Exception:
        pass
    return ""
