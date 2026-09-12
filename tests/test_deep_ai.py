import pytest
from app.schemas.schemas import (
    ArchitecturalTradeoffs,
    NestJSBlueprint,
    LearningPath,
    AIAnalysisResult,
    TechStackItem,
)


def test_deep_ai_schemas():
    tradeoffs = ArchitecturalTradeoffs(
        pros=["High throughput", "Low latency"],
        cons=["High memory usage"],
        when_not_to_use=["Small scale applications with < 100 req/sec"],
        scalability_bottlenecks=["Single connection pool exhaustion"],
    )
    blueprint = NestJSBlueprint(
        architectural_pattern="Clean Architecture with Hexagonal Ports & Adapters",
        suggested_module_structure="src/modules/vector-search/{vector.module.ts, vector.service.ts}",
        code_snippet="// NestJS injectable service snippet\n@Injectable()\nexport class SearchService {}",
        database_integration="Prisma with pgvector extension",
    )
    learning = LearningPath(
        prerequisites=["PostgreSQL indexes", "Vector embeddings basics"],
        recommended_next_topics=["HNSW indexing", "IVFFlat tuning"],
    )
    analysis = AIAnalysisResult(
        relevance_score=9.0,
        is_worth_reading=True,
        target_audience=["Backend NestJS Engineer"],
        vietnamese_title="Kiến trúc Vector Search với NestJS",
        vietnamese_summary="Hướng dẫn xây dựng vector search hiệu năng cao.",
        key_takeaways=["Dùng HNSW cho tập dữ liệu > 1M vectors"],
        new_tech_stack=[TechStackItem(name="pgvector", category="Database", desc="Vector extension")],
        tags=["NestJS", "AI", "PostgreSQL"],
        architectural_tradeoffs=tradeoffs,
        nestjs_blueprint=blueprint,
        learning_path=learning,
    )
    assert analysis.architectural_tradeoffs.cons == ["High memory usage"]
    assert "SearchService" in analysis.nestjs_blueprint.code_snippet
    assert len(analysis.learning_path.prerequisites) == 2

from unittest.mock import patch, AsyncMock
from app.services.ai_analyzer import analyze_article_with_9router

@pytest.mark.asyncio
async def test_ai_analyzer_deep_fields_parsing():
    mock_completion_json = """{
      "relevance_score": 9.5,
      "is_worth_reading": true,
      "target_audience": ["Backend NestJS Engineer", "AI Engineer"],
      "vietnamese_title": "Tối ưu hóa NestJS Microservices với Hybrid Search & pgvector",
      "vietnamese_summary": "Chiến lược thiết kế backend tích hợp AI search ở quy mô lớn.",
      "key_takeaways": [
        "Sử dụng HNSW index cho pgvector để truy vấn sub-millisecond",
        "Tách biệt RAG query và write transactions qua CQRS"
      ],
      "new_tech_stack": [
        {"name": "pgvector", "category": "Database", "desc": "Vector extension for PostgreSQL"},
        {"name": "BullMQ", "category": "Queue", "desc": "Async job processor"}
      ],
      "tags": ["NestJS", "PostgreSQL", "pgvector", "Microservices"],
      "architectural_tradeoffs": {
        "pros": ["Tối ưu độ trễ", "Tái sử dụng hạ tầng PostgreSQL hiện có"],
        "cons": ["Chi phí RAM tăng cao khi xây dựng HNSW graph"],
        "when_not_to_use": ["Khi tập vector nhỏ hơn 10,000 items (dùng exact search là đủ)"],
        "scalability_bottlenecks": ["CPU spike khi re-indexing đồng thời nhiều vectors"]
      },
      "nestjs_blueprint": {
        "architectural_pattern": "Hexagonal Architecture with CQRS Pattern",
        "suggested_module_structure": "src/modules/vector-search/{vector.module.ts, vector.service.ts, vector.repository.ts}",
        "code_snippet": "@Injectable()\\nexport class VectorSearchService {\\n  constructor(private readonly prisma: PrismaService) {}\\n}",
        "database_integration": "Prisma ORM with raw SQL $queryRaw for pgvector cosine similarity"
      },
      "learning_path": {
        "prerequisites": ["NestJS Dependency Injection", "PostgreSQL GiST/GIN/HNSW indexes"],
        "recommended_next_topics": ["Quantization (IVF-PQ)", "Distributed vector cache with Redis"]
      }
    }"""

    mock_resp = AsyncMock()
    mock_choice = AsyncMock()
    mock_choice.message.content = f"```json\n{mock_completion_json}\n```"
    mock_resp.choices = [mock_choice]

    with patch("app.services.ai_analyzer.AsyncOpenAI") as mock_openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client

        res = await analyze_article_with_9router("Tối ưu hóa NestJS Microservices", "Nội dung bài viết", "https://blog.tech/nestjs-pgvector")
        assert res.relevance_score == 9.5
        assert res.architectural_tradeoffs is not None
        assert "Tối ưu độ trễ" in res.architectural_tradeoffs.pros
        assert res.nestjs_blueprint is not None
        assert "VectorSearchService" in res.nestjs_blueprint.code_snippet
        assert "NestJS Dependency Injection" in res.learning_path.prerequisites

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.models import Article
from app.core.database import AsyncSessionLocal
import datetime

@pytest.mark.asyncio
async def test_article_chat_copilot():
    unique_key = str(int(datetime.datetime.now().timestamp()))
    async with AsyncSessionLocal() as db:
        art = Article(
            title=f"NestJS BullMQ Worker Test {unique_key}",
            url=f"https://test.com/bullmq-{unique_key}",
            raw_content="NestJS with BullMQ provides robust queue processing and rate limiting.",
            published_at=datetime.datetime(2026, 9, 12),
        )
        db.add(art)
        await db.commit()
        await db.refresh(art)

    mock_chat_result = {
        "reply": "Để áp dụng BullMQ trong NestJS, bạn inject `@InjectQueue('tasks')` vào service và cấu hình Redis connection pooling.",
        "suggested_followups": [
            "Cách xử lý retry backoff trong BullMQ?",
            "Làm sao scale nhiều consumer pods với Redis cluster?"
        ]
    }

    with patch("app.api.endpoints.chat_with_article", return_value=mock_chat_result):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post(f"/api/articles/{art.id}/chat", json={
                "message": "Làm thế nào để áp dụng BullMQ vào NestJS?",
                "history": []
            })
            assert r.status_code == 200
            data = r.json()
            assert "InjectQueue" in data["reply"]
            assert len(data["suggested_followups"]) == 2

@pytest.mark.asyncio
async def test_get_related_articles():
    unique_key = str(int(datetime.datetime.now().timestamp()))
    async with AsyncSessionLocal() as db:
        art1 = Article(
            title=f"NestJS Microservices Part 1 {unique_key}",
            url=f"https://test.com/art1-{unique_key}",
            tags=["NestJS", "Microservices", "Kafka"],
            relevance_score=8.5,
            published_at=datetime.datetime(2026, 9, 10),
        )
        art2 = Article(
            title=f"NestJS Microservices Part 2 {unique_key}",
            url=f"https://test.com/art2-{unique_key}",
            tags=["NestJS", "Microservices", "RabbitMQ"],
            relevance_score=9.0,
            published_at=datetime.datetime(2026, 9, 11),
        )
        art3 = Article(
            title=f"Cooking Recipes {unique_key}",
            url=f"https://test.com/art3-{unique_key}",
            tags=["Cooking", "Food"],
            relevance_score=3.0,
            published_at=datetime.datetime(2026, 9, 12),
        )
        db.add_all([art1, art2, art3])
        await db.commit()
        await db.refresh(art1)
        await db.refresh(art2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get(f"/api/articles/{art1.id}/related")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        assert any(item["id"] == art2.id for item in data)
        assert not any(item["id"] == art1.id for item in data)  # Does not include itself

@pytest.mark.asyncio
async def test_weekly_radar_digest():
    mock_digest = {
        "week_label": "Tuần 37, 2026",
        "dominant_trends": [
            {
                "topic": "Model Context Protocol (MCP)",
                "status": "Adopt",
                "summary": "Chuẩn hóa giao tiếp giữa AI Agents và hệ thống dữ liệu doanh nghiệp.",
                "relevance": "Cực cao cho Backend NestJS Microservices"
            },
            {
                "topic": "PostgreSQL 18 Async I/O & pgvector",
                "status": "Trial",
                "summary": "Tăng 30% throughput cho workload Vector Search.",
                "relevance": "Thay thế direct vector DB cho quy mô vừa và lớn"
            }
        ],
        "architectural_shifts": [
            "Dịch chuyển từ monolithic LLM calls sang Multi-agent CQRS pipelines",
            "Sử dụng hybrid lexical + dense vector retrieval thay vì thuần vector search"
        ],
        "actionable_recommendations": [
            "Audit lại connection pool PostgreSQL khi dùng pgvector",
            "Tích hợp BullMQ rate limiter trước khi gửi request tới AI gateways"
        ]
    }

    with patch("app.api.endpoints.generate_weekly_radar_digest", return_value=mock_digest):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.get("/api/intelligence/radar-digest")
            assert r.status_code == 200
            data = r.json()
            assert data["week_label"] == "Tuần 37, 2026"
            assert len(data["dominant_trends"]) >= 1
            assert len(data["architectural_shifts"]) >= 1
