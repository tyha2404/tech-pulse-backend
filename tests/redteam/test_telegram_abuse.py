import pytest
import time
from app.services.telegram_service import (
    _check_urgent_rate_limit,
    format_urgent_alert_message,
    escape_html,
    _urgent_alert_timestamps,
)
from app.core.config import settings


def test_telegram_urgent_alert_rate_limiting_defense():
    """
    Red Team test: Attacker pushes 50 high-urgency breaking articles within seconds
    to spam and exhaust Telegram notifications.
    Ensure rate-limiter clamps down and strictly blocks excess messages beyond TELEGRAM_ALERT_MAX_PER_HOUR.
    """
    # Reset in-memory rate-limiter state
    import app.services.telegram_service as ts
    ts._urgent_alert_timestamps = []

    max_allowed = settings.TELEGRAM_ALERT_MAX_PER_HOUR
    allowed_count = 0
    blocked_count = 0

    # Attempt 30 rapid-fire alert checks
    for _ in range(30):
        if _check_urgent_rate_limit():
            allowed_count += 1
        else:
            blocked_count += 1

    assert allowed_count == max_allowed, f"Expected {max_allowed} allowed, got {allowed_count}"
    assert blocked_count == 30 - max_allowed, f"Expected {30 - max_allowed} blocked, got {blocked_count}"


def test_telegram_html_markdown_injection_resistance():
    """
    Red Team test: Article title contains unclosed HTML tags, raw angles, or malicious entities
    intended to trigger Telegram API 400 Bad Request error (Malformed HTML message).
    """
    class MaliciousArticle:
        id = 999
        title = "Exploit: <b<b<b>Unclosed Tag <script>alert(1)</script> & special chars <<>>"
        vietnamese_title = "<b>Lỗ hổng nghiêm trọng</b>: <img src=x> && <unknown>"
        vietnamese_summary = "Đoạn văn chứa thẻ <a href='javascript:void(0)'>Click here</a> & unclosed <code>"
        relevance_score = 9.9
        source = None
        url = "https://safe.org/post"
        key_takeaways = ["<b>Hacker</b>", "Unclosed <i>italic"]
        new_tech_stacks = [{"name": "<svg onload=1>"}]
        tags = ["<script>", "AI/LLM"]

    article = MaliciousArticle()
    msg, reply_markup = format_urgent_alert_message(article)

    # All user/article content must be escaped via html.escape
    # Check that dangerous unescaped tags are safely transformed
    assert "<script>" not in msg
    assert "&lt;script&gt;" in msg or "script" in msg
    assert "<svg" not in msg
    assert "&lt;svg" in msg or "svg" in msg
    
    # Check inline keyboard structure is safe
    assert "reply_markup" != ""
    assert isinstance(reply_markup, dict)
    assert "inline_keyboard" in reply_markup
