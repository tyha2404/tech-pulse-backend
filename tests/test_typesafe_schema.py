import pytest
from app.schemas.schemas import (
    AIAnalysisResult,
    EngineeringPersonaEnum,
    ArchitecturalPatternEnum,
)
from app.services.ai_analyzer import sanitize_and_validate_analysis


def test_sanitize_target_audience_personas():
    """Verify loose personas strings are mapped cleanly to standard engineering personas"""
    raw_payload = {
        "relevance_score": 8.5,
        "is_worth_reading": True,
        "target_audience": ["ai researcher", "cloud/sre engineer", "tech lead", "frontend web"],
        "vietnamese_title": "Kiến trúc hệ thống AI hiện đại",
        "vietnamese_summary": "Tóm tắt chuyên sâu về AI.",
        "key_takeaways": ["Takeaway 1"],
        "new_tech_stack": [],
        "tags": ["#AI", "NestJS, Typescript"],
        "cluster_topic_key": "Anthropic Claude 3.7 Release!!",
    }

    result = sanitize_and_validate_analysis(
        raw_data=raw_payload,
        fallback_title="Fallback Title",
        fallback_content="Fallback Content",
        url="https://test.com",
    )

    assert EngineeringPersonaEnum.AI_SYSTEMS_ENGINEER.value in result.target_audience
    assert EngineeringPersonaEnum.DEVOPS_SRE.value in result.target_audience
    assert EngineeringPersonaEnum.TECH_LEAD.value in result.target_audience
    assert EngineeringPersonaEnum.FULLSTACK_DEVELOPER.value in result.target_audience

    # Slug must be strictly kebab-case
    assert result.cluster_topic_key == "anthropic-claude-3-7-release"

    # Tags must be cleaned and capitalized
    assert "Ai" in result.tags or "AI" in [t.upper() for t in result.tags]


def test_partial_failure_field_recovery():
    """If a complex nested field (e.g. nestjs_blueprint) is malformed, essential data is still preserved"""
    raw_payload = {
        "relevance_score": 9.2,
        "is_worth_reading": True,
        "vietnamese_title": "PostgreSQL 17 Sharding & Clustering",
        "vietnamese_summary": "Phân tích kỹ thuật chuyên sâu về phân mảnh dữ liệu.",
        "key_takeaways": ["Dùng Citus", "Tối ưu Transaction isolation"],
        "new_tech_stack": [],
        "tags": ["PostgreSQL"],
        "nestjs_blueprint": "invalid_blueprint_string_instead_of_dict",  # Malformed nested field
        "architectural_tradeoffs": {"pros": ["High speed"], "cons": ["High memory"]},
    }

    result = sanitize_and_validate_analysis(
        raw_data=raw_payload,
        fallback_title="Fallback Title",
        fallback_content="Fallback Content",
        url="https://test.com",
    )

    # Core data must survive
    assert result.relevance_score == 9.2
    assert result.vietnamese_title == "PostgreSQL 17 Sharding & Clustering"
    assert result.nestjs_blueprint is None  # Gracefully recovered to None without crash
    assert result.architectural_tradeoffs is not None
    assert result.architectural_tradeoffs.pros == ["High speed"]
