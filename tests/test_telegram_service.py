import pytest
from app.services.telegram_service import (
    format_elite_article_message,
    format_cluster_alert_message,
    format_espresso_digest_message,
    escape_html,
)

def test_escape_html():
    raw = "Code <test> & 'quotes' \"here\""
    escaped = escape_html(raw)
    assert "&lt;test&gt;" in escaped
    assert "&amp;" in escaped

def test_format_elite_article_message():
    article_data = {
        "id": 101,
        "title": "NestJS Microservices Outbox Pattern",
        "vietnamese_title": "Mô hình Outbox Pattern trong NestJS",
        "vietnamese_summary": "Giải pháp đảm bảo tính nhất quán dữ liệu phân tán.",
        "relevance_score": 8.8,
        "source_name": "ByteByteGo",
        "reading_time_minutes": 5,
        "key_takeaways": ["Tránh lỗi dual-write", "Đảm bảo Eventual Consistency"],
        "tags": ["NestJS", "Kafka"],
        "url": "https://bytebytego.com/article/1",
    }
    msg, reply_markup = format_elite_article_message(article_data)
    assert "TECHPULSE ELITE" in msg
    assert "8.8/10" in msg
    assert "Mô hình Outbox Pattern" in msg
    assert "inline_keyboard" in reply_markup
    # Verify button urls
    buttons = reply_markup["inline_keyboard"][0]
    assert any("bytebytego.com" in b["url"] for b in buttons)

def test_format_cluster_alert_message():
    cluster_data = {
        "title": "Ra mắt Claude 3.7 Sonnet",
        "sources": ["Tinh Tế", "Tuổi Trẻ", "GenK"],
        "summary": "Mô hình kết hợp tư duy logic và phản hồi nhanh.",
        "url": "https://tinhte.vn/claude-3-7",
    }
    msg, reply_markup = format_cluster_alert_message(cluster_data)
    assert "XU HƯỚNG CÔNG NGHỆ ĐANG NÓNG" in msg
    assert "Tinh Tế" in msg
    assert "Tuổi Trẻ" in msg
    assert "inline_keyboard" in reply_markup

def test_format_espresso_digest_message():
    articles = [
        {"title": "Bài 1", "vietnamese_title": "Tin công nghệ 1", "relevance_score": 9.0, "url": "https://a.com"},
        {"title": "Bài 2", "vietnamese_title": "Tin công nghệ 2", "relevance_score": 8.5, "url": "https://b.com"},
    ]
    msg, reply_markup = format_espresso_digest_message(articles, "☕ Morning Tech Espresso")
    assert "Morning Tech Espresso" in msg
    assert "Tin công nghệ 1" in msg
    assert "Tin công nghệ 2" in msg
