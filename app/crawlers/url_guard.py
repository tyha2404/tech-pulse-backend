"""Guards for outbound crawler requests.

Feed, article and API-supplied URLs are untrusted: every request the crawler
sends, including each redirect hop, must resolve to a public address, and
responses are capped in size and total download time.
"""
import asyncio
import ipaddress
import socket
import time

import httpx

# Largest body read from any feed or article page, and the wall-clock budget for
# reading it (httpx's timeout only bounds the gap between chunks)
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_SECONDS = 30.0


class UnsafeURLError(ValueError):
    """The URL points somewhere the crawler must not fetch."""


class ResponseTooLargeError(ValueError):
    """The response exceeded MAX_RESPONSE_BYTES or MAX_RESPONSE_SECONDS."""


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def ensure_public_url(url: str | httpx.URL) -> None:
    """Raise UnsafeURLError unless url is http(s) and every address its host resolves to is public.

    The check resolves DNS separately from the connection, so a rebinding DNS
    server can still race it; it stops direct and redirect-based SSRF.
    """
    try:
        parsed = httpx.URL(str(url))
    except httpx.InvalidURL as e:
        raise UnsafeURLError(f"Invalid URL: {url}") from e
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise UnsafeURLError(f"Only public http(s) URLs are allowed: {url}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            parsed.host, port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise UnsafeURLError(f"Cannot resolve host: {parsed.host}") from e

    for *_, sockaddr in infos:
        if not _is_public_ip(ipaddress.ip_address(sockaddr[0])):
            raise UnsafeURLError(f"Blocked non-public address for host: {parsed.host}")


async def _check_request(request: httpx.Request) -> None:
    await ensure_public_url(request.url)


class _CappedStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream):
        self._stream = stream

    async def __aiter__(self):
        total = 0
        deadline = time.monotonic() + MAX_RESPONSE_SECONDS
        async for chunk in self._stream:
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise ResponseTooLargeError(f"Response larger than {MAX_RESPONSE_BYTES} bytes")
            if time.monotonic() > deadline:
                raise ResponseTooLargeError(f"Response took longer than {MAX_RESPONSE_SECONDS}s")
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _CappedTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        response.stream = _CappedStream(response.stream)
        return response


def guarded_client(**kwargs) -> httpx.AsyncClient:
    """httpx.AsyncClient for untrusted URLs: SSRF check on every hop, capped responses."""
    return httpx.AsyncClient(
        transport=_CappedTransport(),
        event_hooks={"request": [_check_request]},
        **kwargs,
    )
