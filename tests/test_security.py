import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.database import get_db
from app.crawlers import url_guard
from app.crawlers.url_guard import UnsafeURLError, ensure_public_url
from app.main import app

FRONTEND = "http://127.0.0.1:5174"
EVIL = "https://evil.example"


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    with patch.object(settings, "FRONTEND_URL", FRONTEND):
        yield AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:8000")
    app.dependency_overrides.clear()


def fake_dns(mapping):
    """Patch getaddrinfo so host -> IP lookups never touch the network."""

    async def getaddrinfo(host, port, **kwargs):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port))]

    loop = MagicMock(getaddrinfo=getaddrinfo)
    return patch("app.crawlers.url_guard.asyncio.get_running_loop", return_value=loop)


# ---------- CORS & cross-site requests ----------


LOCAL_FRONTENDS = ["http://127.0.0.1:5174", "http://localhost:5174", "http://127.0.0.1:5173", "http://localhost:5173"]


def test_cors_origins_are_frontend_url_plus_local_frontends():
    with patch.object(settings, "FRONTEND_URL", "http://127.0.0.1:5174/"):
        assert settings.cors_origins_list == LOCAL_FRONTENDS
    # A public FRONTEND_URL (with a path) doesn't lock out the local frontends
    with patch.object(settings, "FRONTEND_URL", "https://pulse.example/app/"):
        assert settings.cors_origins_list == ["https://pulse.example", *LOCAL_FRONTENDS]


async def test_cors_only_reflects_the_frontend_origin(client):
    # The middleware captured the origin list at import, so use what it was built from
    ours_origin = settings.cors_origins_list[0]
    async with client:
        evil = await client.options(
            "/api/sources",
            headers={"Origin": EVIL, "Access-Control-Request-Method": "DELETE"},
        )
        ours = await client.options(
            "/api/sources",
            headers={"Origin": ours_origin, "Access-Control-Request-Method": "DELETE"},
        )
    assert evil.headers.get("access-control-allow-origin") is None
    assert ours.headers.get("access-control-allow-origin") == ours_origin


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": EVIL, "Sec-Fetch-Site": "cross-site"},
        {"Origin": EVIL},  # browsers without Fetch Metadata
        {"Origin": "null"},  # sandboxed iframe / file://
        {"Sec-Fetch-Site": "cross-site"},  # <form> or <img> without an Origin
        {"Origin": "http://127.0.0.1:3000", "Sec-Fetch-Site": "same-site"},  # another local app
    ],
)
async def test_cross_site_requests_are_rejected(client, headers):
    async with client:
        r = await client.post("/api/admin/circuit-breaker/reset", headers=headers)
    assert r.status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": FRONTEND, "Sec-Fetch-Site": "same-site"},  # Vite dev app calling :8000
        {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-origin"},  # nginx /api proxy
        {},  # curl, scripts, tests, Telegram's webhook
    ],
)
async def test_frontend_and_non_browser_requests_pass(client, headers):
    async with client:
        r = await client.post("/api/admin/circuit-breaker/reset", headers=headers)
    assert r.status_code == 200


# ---------- SSRF guard ----------


async def test_create_source_rejects_internal_url(client):
    async with client:
        r = await client.post(
            "/api/sources",
            json={"name": "x", "url": "http://169.254.169.254/latest/meta-data/", "category": "AI"},
        )
    assert r.status_code == 400


async def test_test_source_rejects_internal_url(client):
    async with client:
        r = await client.post("/api/sources/test", params={"target_url": "http://127.0.0.1:20128/v1/models"})
    assert r.json()["success"] is False


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1",
     "::ffff:127.0.0.1", "fd00::1", "224.0.0.1"],
)
async def test_hostnames_resolving_to_non_public_ips_are_blocked(ip):
    with fake_dns({"rebind.example": ip}), pytest.raises(UnsafeURLError):
        await ensure_public_url("https://rebind.example/feed.xml")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x.example/", "http:///nohost", "not a url"])
async def test_non_http_urls_are_blocked(url):
    with fake_dns({}), pytest.raises(UnsafeURLError):
        await ensure_public_url(url)


async def test_public_host_passes():
    with fake_dns({"news.example": "93.184.215.14"}):
        await ensure_public_url("https://news.example/rss")


async def test_redirect_to_internal_address_is_blocked():
    def handler(request):
        if request.url.host == "news.example":
            return httpx.Response(302, headers={"Location": "http://metadata.internal/latest/"})
        return httpx.Response(200, text="secret")

    with fake_dns({"news.example": "93.184.215.14", "metadata.internal": "169.254.169.254"}):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [url_guard._check_request]},
            follow_redirects=True,
        ) as c:
            with pytest.raises(UnsafeURLError):
                await c.get("https://news.example/article")


async def test_response_body_is_capped():
    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b"x" * 1024

    with patch.object(url_guard, "MAX_RESPONSE_BYTES", 10_000), pytest.raises(url_guard.ResponseTooLargeError):
        async for _ in url_guard._CappedStream(Endless()):
            pass


async def test_self_referencing_sitemap_index_stops():
    from app.crawlers.article_crawler import fetch_sitemap_feed

    resp = MagicMock(
        text="<sitemapindex><sitemap><loc>https://s.example/sitemap.xml</loc></sitemap></sitemapindex>"
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=resp) as get:
        assert await fetch_sitemap_feed("https://s.example/sitemap.xml") == []
    assert get.await_count == 4


def test_digest_link_url_is_escaped():
    from app.services.telegram_service import format_espresso_digest_message

    msg, _ = format_espresso_digest_message(
        [{"title": "t", "url": 'https://x.example/"><a href="https://phish.example">'}], "Digest"
    )
    assert "phish.example\">" not in msg
    assert "&quot;&gt;&lt;a href=" in msg
