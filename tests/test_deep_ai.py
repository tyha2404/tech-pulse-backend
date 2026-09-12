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
