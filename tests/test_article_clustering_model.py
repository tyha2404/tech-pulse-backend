import pytest
from app.models.models import Article
from app.schemas.schemas import ArticleResponse, RelatedSourceArticle


def test_article_model_has_clustering_columns():
    article = Article(
        title="Test Title",
        url="https://example.com/1",
        cluster_id="cluster-123",
        is_canonical=True,
        cluster_topic_key="iphone-18-launch",
    )
    assert article.cluster_id == "cluster-123"
    assert article.is_canonical is True
    assert article.cluster_topic_key == "iphone-18-launch"


def test_related_source_article_schema():
    related = RelatedSourceArticle(
        id=10,
        title="Tiêu đề báo khác",
        source_name="Tuổi Trẻ",
        url="https://tuoitre.vn/test",
    )
    assert related.source_name == "Tuổi Trẻ"
    assert related.id == 10
