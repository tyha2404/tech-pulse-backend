"""Request guards for the /api routes."""
from fastapi import HTTPException, Request

from app.core.config import settings


def is_cross_site_request(request: Request) -> bool:
    """True when a browser sent this request on behalf of a page we don't serve.

    CORS only hides responses; a form post or a no-preflight fetch from any
    website still reaches the handler, so state-changing routes need this check.
    """
    origin = request.headers.get("origin")
    fetch_site = request.headers.get("sec-fetch-site")
    if origin:
        if origin.rstrip("/") in settings.cors_origins_list:
            return False
        # A same-origin page behind a proxy (nginx, Vite dev server) has an Origin we don't list
        return fetch_site != "same-origin"
    # No Origin: a non-browser client (curl, Telegram's webhook), a typed URL, or a subresource such as <img>
    return fetch_site in ("cross-site", "same-site")


async def require_api_access(request: Request) -> None:
    if is_cross_site_request(request):
        raise HTTPException(status_code=403, detail="Yêu cầu từ trang web khác bị chặn")
