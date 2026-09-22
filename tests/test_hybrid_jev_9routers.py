import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from app.core.config import settings
from app.schemas.schemas import FastTriageResult, AIAnalysisResult
from app.services.ai_analyzer import (
    _build_calibrated_system_prompt,
    analyze_article_with_9router,
    BASE_SYSTEM_PROMPT,
    FAST_SYSTEM_PROMPT,
)
from app.services.clustering_service import (
    verify_same_story_with_typesafe,
    assign_article_cluster_async,
)
from app.services.rerank_service import (
    rerank_articles_for_persona,
    _heuristic_persona_scoring,
)
from app.models.models import Article


def test_tiered_model_selection():
    """Verify fast and deep model lists are parsed properly with fallbacks."""
    assert len(settings.fast_models_list) >= 2
    assert "groq/openai/gpt-oss-120b" in settings.fast_models_list
    assert len(settings.deep_models_list) >= 2
    assert "nexo-chat" in settings.deep_models_list


def test_dynamic_prompt_steering_selection():
    """Verify System-1 triage steers prompt into Deep vs Fast mode."""
    # 1. Deep Mode (depth >= 6.0)
    deep_triage = FastTriageResult(
        is_relevant_tech=True,
        is_spam_or_marketing=False,
        confidence=0.95,
        suggested_priority="PROCESS_FULL_AI",
        is_breaking_news=False,
        urgency_level="NORMAL",
        tech_depth_score=8.5,
        reason="Deep architecture analysis",
    )
    prompt, is_deep = _build_calibrated_system_prompt(triage_info=deep_triage)
    assert is_deep is True
    assert "In-Depth 3-Part Paragraph Structure" in prompt

    # 2. Fast Mode (depth < 6.0)
    fast_triage = FastTriageResult(
        is_relevant_tech=True,
        is_spam_or_marketing=False,
        confidence=0.90,
        suggested_priority="PROCESS_FULL_AI",
        is_breaking_news=True,
        urgency_level="CRITICAL",
        tech_depth_score=4.5,
        reason="Breaking hardware release",
    )
    prompt_fast, is_deep_fast = _build_calibrated_system_prompt(triage_info=fast_triage)
    assert is_deep_fast is False
    assert "Fast Technical News & Architecture Intelligence Analyst" in prompt_fast


@pytest.mark.asyncio
async def test_analyze_article_with_9router_dynamic_steering():
    """Verify analyze_article_with_9router injects triage context and returns valid analysis."""
    triage = FastTriageResult(
        is_relevant_tech=True,
        is_spam_or_marketing=False,
        confidence=0.92,
        suggested_priority="PROCESS_FULL_AI",
        is_breaking_news=False,
        urgency_level="HIGH",
        tech_depth_score=5.0,
        reason="Software tool update",
    )

    mock_llm_json = """
    {
      "relevance_score": 6.8,
      "is_worth_reading": false,
      "target_audience": ["Backend Developer"],
      "vietnamese_title": "Ra Mắt Phiên Bản Mới",
      "vietnamese_summary": "Bản tin tổng hợp kỹ thuật nhanh về bản cập nhật công cụ phát triển.",
      "cluster_topic_key": "tool-release",
      "key_takeaways": ["Cải thiện hiệu năng", "Tương thích ngược"],
      "new_tech_stack": [],
      "tags": ["DevOps", "Tooling"],
      "architectural_tradeoffs": null,
      "nestjs_blueprint": null,
      "learning_path": null
    }
    """

    mock_choice = MagicMock()
    mock_choice.message.content = mock_llm_json
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("app.services.ai_analyzer.AsyncOpenAI") as mock_openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_client

        analysis, model_used = await analyze_article_with_9router(
            title="DevOps Tool v2 Release",
            content="Full release notes for DevOps tool.",
            url="https://example.com/tool-v2",
            triage_info=triage,
        )

        assert analysis.vietnamese_title == "Ra Mắt Phiên Bản Mới"
        assert analysis.relevance_score == 6.8
        assert any(t.lower() == "devops" for t in analysis.tags)

        # Verify call arguments included SYSTEM-1 TRIAGE SIGNALS
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert "SYSTEM-1 TRIAGE SIGNALS" in user_msg
        assert "Assessed Tech Depth Score: 5.0/10.0" in user_msg


@pytest.mark.asyncio
async def test_semantic_clustering_with_jev_noul():
    """Verify TypeSafe Jev resolves borderline clustering similarity."""
    mock_typesafe_resp = MagicMock()
    mock_typesafe_resp.status_code = 200
    mock_typesafe_resp.json.return_value = {
        "answers": {
            "is_same_story": {
                "noul": 0.88,
            }
        }
    }

    with patch("app.core.config.settings.TYPESAFE_API_KEY", "test-key"):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_typesafe_resp

            is_same = await verify_same_story_with_typesafe(
                title_a="PostgreSQL 17 Released",
                title_b="Postgres 17 ra mat nang cap hieu nang",
                snippet_a="Major performance gains in query execution.",
                snippet_b="Nang cap toc do truy van du lieu.",
            )
            assert is_same is True

            # Verify payload sent to TypeSafe
            sent_payload = mock_post.call_args[1]["json"]
            assert sent_payload["state"]["article_a"]["title"] == "PostgreSQL 17 Released"
            assert sent_payload["questions"]["is_same_story"]["type"] == "noul"


@pytest.mark.asyncio
async def test_personalized_reranking_with_jev_score():
    """Verify TypeSafe Jev scores and reranks articles for a specific persona."""
    art1 = Article(
        id=1,
        title="Deep Dive into NestJS Microservices",
        vietnamese_title="Phân tích sâu Microservices trong NestJS",
        tags=["NestJS", "Backend", "Microservices"],
        target_audience=["Backend NestJS Engineer"],
        relevance_score=7.0,
    )
    art2 = Article(
        id=2,
        title="Kubernetes SRE Best Practices",
        vietnamese_title="Kinh nghiệm vận hành Kubernetes SRE",
        tags=["Kubernetes", "DevOps", "SRE"],
        target_audience=["DevOps Engineer"],
        relevance_score=8.5,
    )

    # 1. Test Heuristic Scorer
    score_nestjs = _heuristic_persona_scoring(art1, "Backend NestJS Engineer")
    score_k8s = _heuristic_persona_scoring(art2, "Backend NestJS Engineer")
    assert score_nestjs > score_k8s

    # 2. Test Jev System One Scoring Mock
    # When passed articles=[art2, art1]:
    # art2 is index 0 -> score_0 (low fit for NestJS persona)
    # art1 is index 1 -> score_1 (high fit for NestJS persona)
    mock_typesafe_resp = MagicMock()
    mock_typesafe_resp.status_code = 200
    mock_typesafe_resp.json.return_value = {
        "answers": {
            "score_0": {"score": 0.5},  # Low fit for art2
            "score_1": {"score": 3.9},  # High fit for art1
        }
    }

    with patch("app.core.config.settings.TYPESAFE_API_KEY", "test-key"):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_typesafe_resp

            reranked = await rerank_articles_for_persona(
                articles=[art2, art1],
                persona="Backend NestJS Engineer",
            )
            # art1 should be promoted to top despite having lower initial relevance_score
            assert reranked[0][0].id == 1
            assert reranked[0][1] > reranked[1][1]


@pytest.mark.asyncio
async def test_personalized_feed_endpoint():
    """Verify GET /api/v1/articles/feed/personalized returns reranked articles."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    from app.core.database import AsyncSessionLocal
    import datetime

    unique_key = str(int(datetime.datetime.now().timestamp()))
    async with AsyncSessionLocal() as db:
        art = Article(
            title=f"Personalized NestJS Testing {unique_key}",
            url=f"https://test.com/personalized-{unique_key}",
            published_at=datetime.datetime.now(datetime.timezone.utc),
            relevance_score=7.8,
            tags=["NestJS", "Backend"],
            target_audience=["Backend NestJS Engineer"],
            vietnamese_title="Bài viết NestJS cá nhân hoá",
            is_hidden=False,
            is_canonical=True,
        )
        db.add(art)
        await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/articles/feed/personalized",
            params={"persona": "Backend NestJS Engineer", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "relevance_score" in data[0]
            assert "vietnamese_title" in data[0]

