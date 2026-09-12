import pytest
from app.services.clustering_service import assign_article_cluster
from app.models.models import Article

def test_assign_article_cluster_to_existing_matching_title():
    existing_article = Article(
        id=1,
        title="FPT Shop mở bán iPhone 18",
        vietnamese_title="FPT Shop mở bán iPhone 18 chính hãng",
        cluster_id="cluster-abc",
        is_canonical=True,
        relevance_score=7.0,
    )
    new_article = Article(
        id=2,
        title="Viettel Store mở bán iPhone 18",
        vietnamese_title="Viettel Store chính thức mở bán iPhone 18",
        relevance_score=8.5,
    )
    cluster_id, is_canonical, demoted_id = assign_article_cluster(new_article, [existing_article])
    assert cluster_id == "cluster-abc"
    # New article has higher relevance_score (8.5 > 7.0), so it becomes canonical
    assert is_canonical is True
    assert demoted_id == 1

def test_assign_article_cluster_matching_topic_key():
    existing_article = Article(
        id=1,
        title="DeepSeek ra mắt phiên bản V3",
        cluster_id="cluster-deepseek",
        cluster_topic_key="deepseek-v3-release",
        is_canonical=True,
        relevance_score=9.0,
    )
    new_article = Article(
        id=2,
        title="Mô hình AI mã nguồn mở mới gây sốt",
        cluster_topic_key="deepseek-v3-release",
        relevance_score=7.5,
    )
    cluster_id, is_canonical, demoted_id = assign_article_cluster(new_article, [existing_article])
    assert cluster_id == "cluster-deepseek"
    # Lower score -> not canonical
    assert is_canonical is False
    assert demoted_id is None

def test_assign_article_cluster_new_cluster():
    existing_article = Article(
        id=1,
        title="Hướng dẫn NestJS Microservices",
        cluster_id="cluster-nestjs",
        is_canonical=True,
        relevance_score=8.0,
    )
    new_article = Article(
        id=2,
        title="Ra mắt máy bay không người lái mới",
        relevance_score=6.0,
    )
    cluster_id, is_canonical, demoted_id = assign_article_cluster(new_article, [existing_article])
    assert cluster_id != "cluster-nestjs"
    assert is_canonical is True
    assert demoted_id is None
