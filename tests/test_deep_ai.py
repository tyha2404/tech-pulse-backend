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
