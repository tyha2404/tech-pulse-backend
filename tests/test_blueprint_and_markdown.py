import pytest
import datetime
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.database import get_db
from app.models.models import Article, Source
from app.schemas.schemas import ArticleBlueprintResponse, MarkdownExportResponse
from app.services.ai_analyzer import (
    BASE_SYSTEM_PROMPT,
    generate_blueprint_on_demand,
)
from app.api.endpoints import generate_obsidian_markdown


def test_base_prompt_excludes_nestjs_blueprint():
    """Verify that nestjs_blueprint has been removed from the default AI crawl prompt"""
    assert "nestjs_blueprint" not in BASE_SYSTEM_PROMPT
    assert '"nestjs_blueprint"' not in BASE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_generate_blueprint_on_demand_success():
    """Verify on-demand blueprint generation succeeds with AI completion"""
    mock_data = {
        "architectural_pattern": "Hexagonal Architecture with CQRS",
        "suggested_module_structure": "src/modules/crawler/\n├── crawler.controller.ts\n├── crawler.service.ts",
        "code_snippet": "@Injectable()\nexport class CrawlerService {}",
        "database_integration": "PostgreSQL with Prisma and Redis caching",
    }
    mock_resp = AsyncMock()
    mock_choice = AsyncMock()
    mock_choice.message.content = f'```json\n{import_json_str(mock_data)}\n```'
    mock_resp.choices = [mock_choice]

    with patch("app.services.ai_analyzer.AsyncOpenAI") as mock_openai:
        client = AsyncMock()
        client.chat.completions.create.return_value = mock_resp
        mock_openai.return_value = client

        res = await generate_blueprint_on_demand(
            article_title="High Concurrency RAG",
            article_content="Article details...",
            vietnamese_summary="Tóm tắt bài viết",
        )
        assert res["architectural_pattern"] == "Hexagonal Architecture with CQRS"
        assert "@Injectable()" in res["code_snippet"]
        assert "Prisma" in res["database_integration"]


@pytest.mark.asyncio
async def test_generate_blueprint_on_demand_fallback():
    """Verify fallback blueprint when AI provider fails"""
    with patch("app.services.ai_analyzer.AsyncOpenAI") as mock_openai:
        client = AsyncMock()
        client.chat.completions.create.side_effect = Exception("API connection timeout")
        mock_openai.return_value = client

        res = await generate_blueprint_on_demand(
            article_title="Microservices Architecture",
            article_content="Some content",
        )
        assert "Fallback Blueprint" in res["architectural_pattern"]
        assert "Injectable" in res["code_snippet"]
        assert "PostgreSQL" in res["database_integration"]


def test_generate_obsidian_markdown_formatter():
    """Verify markdown output meets Obsidian frontmatter requirements"""
    source = Source(id=1, name="Hacker News")
    article = Article(
        id=42,
        title="Modern Vector Databases at Scale",
        vietnamese_title="Cơ sở dữ liệu Vector Hiện đại ở Quy mô Lớn",
        url="https://example.com/vector-db",
        relevance_score=9.2,
        is_worth_reading=True,
        published_at=datetime.datetime(2026, 10, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        created_at=datetime.datetime(2026, 10, 1, 12, 5, 0, tzinfo=datetime.timezone.utc),
        vietnamese_summary="Bài viết phân tích các giải pháp Vector DB cho hệ thống hàng tỷ vectors.",
        key_takeaways=[
            "HNSW index tối ưu cho latency",
            "IVFFlat phù hợp khi tài nguyên RAM hạn chế",
        ],
        tags=["VectorDB", "AI", "PostgreSQL"],
        new_tech_stacks=[
            {"name": "pgvector", "category": "Database", "desc": "Vector extension for PG"}
        ],
        architectural_tradeoffs={
            "pros": ["Tốc độ nhanh"],
            "cons": ["Tiêu tốn RAM"],
            "when_not_to_use": ["Tập dữ liệu quá nhỏ"],
            "scalability_bottlenecks": ["CPU spike khi indexing"],
        },
        nestjs_blueprint={
            "architectural_pattern": "Modular Service Pattern",
            "suggested_module_structure": "src/modules/vector/",
            "code_snippet": "export class VectorService {}",
            "database_integration": "pgvector with Prisma",
        },
        raw_content="Full original text of the article.",
    )
    article.source = source

    md_content, filename = generate_obsidian_markdown(article)

    # Frontmatter assertions (Obsidian frontmatter: title, score, takeaways, date, original link)
    assert md_content.startswith("---")
    assert 'title: "Cơ sở dữ liệu Vector Hiện đại ở Quy mô Lớn"' in md_content
    assert "score: 9.2" in md_content
    assert 'original_link: "https://example.com/vector-db"' in md_content
    assert 'source: "Hacker News"' in md_content
    assert "takeaways:" in md_content
    assert '  - "HNSW index tối ưu cho latency"' in md_content
    assert '  - "IVFFlat phù hợp khi tài nguyên RAM hạn chế"' in md_content
    assert 'date: "2026-10-01T12:00:00+00:00"' in md_content

    # Body assertions
    assert "# Cơ sở dữ liệu Vector Hiện đại ở Quy mô Lớn" in md_content
    assert "## 📌 Tóm tắt nội dung" in md_content
    assert "## 💡 Điểm cốt lõi (Key Takeaways)" in md_content
    assert "## ⚡ Công nghệ mới xuất hiện" in md_content
    assert "## ⚖️ Đánh đổi kiến trúc (Architectural Tradeoffs)" in md_content
    assert "## 🧱 NestJS Architecture & Blueprint" in md_content
    assert "## 📖 Nội dung chi tiết bài viết" in md_content
    assert filename.endswith(".md")


@pytest.mark.asyncio
async def test_api_generate_blueprint_endpoint():
    """Verify POST /api/articles/{id}/generate-blueprint endpoint"""
    article = Article(
        id=101,
        title="Event Driven Architecture",
        vietnamese_title="Kiến trúc Hướng sự kiện",
        url="https://example.com/eda",
        raw_content="Content about Event Driven Architecture in NestJS.",
        nestjs_blueprint=None,
    )
    article.source = None

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = article
    mock_db.execute.return_value = mock_result
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    app.dependency_overrides[get_db] = lambda: mock_db

    mock_bp_data = {
        "architectural_pattern": "Event Sourcing Pattern",
        "suggested_module_structure": "src/events/",
        "code_snippet": "@Injectable()\nexport class EventBusService {}",
        "database_integration": "Redis Streams / BullMQ",
    }

    try:
        with patch("app.api.endpoints.generate_blueprint_on_demand", return_value=mock_bp_data):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.post("/api/articles/101/generate-blueprint")
                assert res.status_code == 200
                data = res.json()
                assert data["architectural_pattern"] == "Event Sourcing Pattern"
                assert "EventBusService" in data["code_snippet"]
                assert data["article_id"] == 101
                assert article.nestjs_blueprint == mock_bp_data

                # Test cached return when not forced
                res_cached = await client.post("/api/articles/101/generate-blueprint")
                assert res_cached.status_code == 200
                assert res_cached.json()["architectural_pattern"] == "Event Sourcing Pattern"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_markdown_endpoints():
    """Verify GET and POST markdown export endpoints"""
    article = Article(
        id=202,
        title="Observability with OpenTelemetry",
        vietnamese_title="Giám sát Hệ thống với OpenTelemetry",
        url="https://example.com/otel",
        relevance_score=8.5,
        is_worth_reading=True,
        published_at=datetime.datetime(2026, 9, 30, 8, 0, 0, tzinfo=datetime.timezone.utc),
        vietnamese_summary="Giám sát phân tán với OTEL traces and metrics.",
        key_takeaways=["Sử dụng trace context propagation"],
        tags=["DevOps", "Monitoring"],
        nestjs_blueprint={"code_snippet": "const tracer = opentelemetry.trace.getTracer('app');"},
        raw_content="Tracing distributed requests across NestJS services.",
    )
    article.source = Source(id=2, name="InfoQ")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = article
    mock_db.execute.return_value = mock_result

    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. GET /api/articles/202/markdown (Default text/markdown)
            res_md = await client.get("/api/articles/202/markdown")
            assert res_md.status_code == 200
            assert "text/markdown" in res_md.headers.get("content-type", "")
            assert "---" in res_md.text
            assert 'title: "Giám sát Hệ thống với OpenTelemetry"' in res_md.text
            assert "score: 8.5" in res_md.text
            assert 'original_link: "https://example.com/otel"' in res_md.text

            # 2. GET /api/articles/202/markdown?format=json
            res_json = await client.get("/api/articles/202/markdown?format=json")
            assert res_json.status_code == 200
            data = res_json.json()
            assert "filename" in data
            assert data["filename"].endswith(".md")
            assert "markdown" in data
            assert 'score: 8.5' in data["markdown"]

            # 3. POST /api/articles/202/export-markdown
            res_post = await client.post("/api/articles/202/export-markdown")
            assert res_post.status_code == 200
            post_data = res_post.json()
            assert post_data["filename"].endswith(".md")
            assert "OpenTelemetry" in post_data["title"]
            assert "takeaways:" in post_data["markdown"]
    finally:
        app.dependency_overrides.clear()


def import_json_str(data: dict) -> str:
    import json
    return json.dumps(data)
