import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin


async def discover_feed_url(target_url: str) -> tuple[str, str]:
    """
    Given a website URL, returns (detected_type, feed_or_target_url).
    Types: 'rss', 'hn', 'html'
    """
    target_url = target_url.strip()
    if "news.ycombinator.com" in target_url:
        return "hn", target_url

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    # Try direct fetch
    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, headers=headers
        ) as client:
            resp = await client.get(target_url)
            content_type = resp.headers.get("content-type", "").lower()
            text = resp.text

            # If it's already an RSS/XML feed
            if (
                "xml" in content_type
                or "<rss" in text[:500].lower()
                or "<feed" in text[:500].lower()
            ):
                return "rss", str(resp.url)

            # Parse HTML for <link rel="alternate" type="application/rss+xml" ...>
            soup = BeautifulSoup(text, "html.parser")
            feed_links = soup.find_all(
                "link", rel=lambda r: r and "alternate" in r.lower()
            )
            for link in feed_links:
                t = link.get("type", "").lower()
                if "rss" in t or "atom" in t or "xml" in t:
                    href = link.get("href")
                    if href:
                        feed_url = urljoin(str(resp.url), href)
                        return "rss", feed_url

            # Check common feed paths
            for suffix in ["/rss", "/feed", "/rss.xml", "/feed.xml", "/atom.xml"]:
                candidate = urljoin(str(resp.url), suffix)
                try:
                    c_resp = await client.get(candidate)
                    if c_resp.status_code == 200 and (
                        "xml" in c_resp.headers.get("content-type", "")
                        or "<rss" in c_resp.text[:500].lower()
                    ):
                        return "rss", candidate
                except Exception:
                    continue

    except Exception:
        pass

    # Fallback to direct HTML scraper
    return "scraper", target_url
