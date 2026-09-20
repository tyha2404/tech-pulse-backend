import re
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}


async def discover_feed_url(target_url: str) -> tuple[str, str]:
    """
    Given a website URL, returns (detected_type, feed_or_target_url).
    Types: 'rss', 'json', 'sitemap', 'hn', 'scraper'
    """
    target_url = target_url.strip()
    if "news.ycombinator.com" in target_url:
        return "hn", target_url

    parsed_target = urlparse(target_url)
    base_url = f"{parsed_target.scheme}://{parsed_target.netloc}"

    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, headers=headers
        ) as client:
            resp = await client.get(target_url)
            content_type = resp.headers.get("content-type", "").lower()
            text = resp.text

            # 1. Check if directly a JSON Feed
            if "application/feed+json" in content_type or "application/json" in content_type:
                try:
                    data = resp.json()
                    if isinstance(data, dict) and ("items" in data or "version" in data):
                        return "json", str(resp.url)
                except Exception:
                    pass

            # 2. Check if directly an RSS/Atom/XML feed
            if (
                "xml" in content_type
                or "<rss" in text[:500].lower()
                or "<feed" in text[:500].lower()
            ):
                return "rss", str(resp.url)

            # 3. Parse HTML for <link> tags (RSS, Atom, JSON Feed)
            soup = BeautifulSoup(text, "html.parser")
            feed_links = soup.find_all(
                "link", rel=lambda r: r and "alternate" in r.lower()
            )
            for link in feed_links:
                t = (link.get("type") or "").lower()
                href = link.get("href")
                if not href:
                    continue
                resolved_href = urljoin(str(resp.url), href)
                if "feed+json" in t:
                    return "json", resolved_href
                if "rss" in t or "atom" in t or "xml" in t:
                    return "rss", resolved_href

            # 4. Check common RSS/Atom feed paths
            for suffix in ["/rss", "/feed", "/rss.xml", "/feed.xml", "/atom.xml", "/feed.json", "/json"]:
                candidate = urljoin(str(resp.url), suffix)
                try:
                    c_resp = await client.get(candidate)
                    if c_resp.status_code == 200:
                        c_type = c_resp.headers.get("content-type", "").lower()
                        if "feed+json" in c_type or "application/json" in c_type:
                            try:
                                if "items" in c_resp.json():
                                    return "json", candidate
                            except Exception:
                                pass
                        if "xml" in c_type or "<rss" in c_resp.text[:500].lower() or "<feed" in c_resp.text[:500].lower():
                            return "rss", candidate
                except Exception:
                    continue

            # 5. Check Sitemap discovery via robots.txt or /sitemap.xml
            try:
                robots_url = urljoin(base_url, "/robots.txt")
                r_resp = await client.get(robots_url)
                if r_resp.status_code == 200:
                    sitemap_matches = re.findall(r"^Sitemap:\s*(https?://[^\s]+)", r_resp.text, re.MULTILINE | re.IGNORECASE)
                    if sitemap_matches:
                        return "sitemap", sitemap_matches[0]
            except Exception:
                pass

            for sitemap_suffix in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/sitemap.xml"]:
                candidate_sitemap = urljoin(base_url, sitemap_suffix)
                try:
                    s_resp = await client.get(candidate_sitemap)
                    if s_resp.status_code == 200 and ("xml" in s_resp.headers.get("content-type", "") or "<urlset" in s_resp.text[:500] or "<sitemapindex" in s_resp.text[:500]):
                        return "sitemap", candidate_sitemap
                except Exception:
                    continue

    except Exception:
        pass

    # Fallback to direct HTML scraper
    return "scraper", target_url
