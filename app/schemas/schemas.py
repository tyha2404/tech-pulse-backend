from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Any, Dict
from datetime import datetime


class SourceBase(BaseModel):
    name: str
    url: str
    feed_url: Optional[str] = None
    source_type: Optional[str] = "rss"
    category: Optional[str] = "General"
    is_active: Optional[bool] = True


class SourceCreate(SourceBase):
    pass


class SourceResponse(SourceBase):
    id: int
    status: str
    last_crawled_at: Optional[datetime] = None
    last_error: Optional[str] = None
    articles_count: Optional[int] = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CrawlRunResponse(BaseModel):
    id: int
    source_id: int
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration_ms: int = 0
    http_status: Optional[int] = None
    articles_found: int = 0
    articles_new: int = 0
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TechStackItem(BaseModel):
    name: str
    category: Optional[str] = "Tool"
    desc: Optional[str] = ""


class ArchitecturalTradeoffs(BaseModel):
    pros: List[str] = []
    cons: List[str] = []
    when_not_to_use: List[str] = []
    scalability_bottlenecks: List[str] = []


class NestJSBlueprint(BaseModel):
    architectural_pattern: Optional[str] = None
    suggested_module_structure: Optional[str] = None
    code_snippet: Optional[str] = None
    database_integration: Optional[str] = None


class LearningPath(BaseModel):
    prerequisites: List[str] = []
    recommended_next_topics: List[str] = []


class ArticleUpdate(BaseModel):
    is_read: Optional[bool] = None
    is_hidden: Optional[bool] = None
    is_bookmarked: Optional[bool] = None


class RelatedSourceArticle(BaseModel):
    id: int
    title: str
    source_name: Optional[str] = None
    url: str
    published_at: Optional[datetime] = None
    vietnamese_title: Optional[str] = None

    class Config:
        from_attributes = True


class ArticleResponse(BaseModel):
    id: int
    source_id: Optional[int] = None
    source_name: Optional[str] = None
    title: str
    url: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    is_processed: bool
    is_worth_reading: bool
    relevance_score: float
    is_read: bool = False
    is_hidden: bool = False
    is_bookmarked: bool = False
    reading_time_minutes: int = 1
    vietnamese_title: Optional[str] = None
    vietnamese_summary: Optional[str] = None
    key_takeaways: List[str] = []
    new_tech_stacks: List[TechStackItem] = []
    tags: List[str] = []
    target_audience: List[str] = []
    architectural_tradeoffs: Optional[ArchitecturalTradeoffs] = None
    nestjs_blueprint: Optional[NestJSBlueprint] = None
    learning_path: Optional[LearningPath] = None
    ai_model_used: Optional[str] = None
    cluster_id: Optional[str] = None
    is_canonical: bool = True
    cluster_topic_key: Optional[str] = None
    similarity_score: Optional[float] = None
    user_feedback: Optional[str] = None
    related_articles: List[RelatedSourceArticle] = []
    created_at: datetime

    class Config:
        from_attributes = True


class AIAnalysisResult(BaseModel):
    relevance_score: float
    is_worth_reading: bool
    target_audience: List[str] = []
    vietnamese_title: str
    vietnamese_summary: str
    key_takeaways: List[str] = []
    new_tech_stack: List[TechStackItem] = []
    tags: List[str] = []
    architectural_tradeoffs: Optional[ArchitecturalTradeoffs] = None
    nestjs_blueprint: Optional[NestJSBlueprint] = None
    learning_path: Optional[LearningPath] = None
    cluster_topic_key: Optional[str] = None


class ArticleChatRequest(BaseModel):
    message: str
    history: List[dict] = []


class ArticleChatResponse(BaseModel):
    reply: str
    suggested_followups: List[str] = []


class RelatedArticleItem(BaseModel):
    id: int
    title: str
    vietnamese_title: Optional[str] = None
    relevance_score: float
    tags: List[str] = []
    source_name: Optional[str] = None


class WeeklyRadarDigestResponse(BaseModel):
    week_label: str
    dominant_trends: List[dict] = []
    architectural_shifts: List[str] = []
    actionable_recommendations: List[str] = []
    top_articles: List[RelatedArticleItem] = []


class CrawlTestResult(BaseModel):
    success: bool
    detected_type: str
    feed_url: Optional[str] = None
    items_count: int
    sample_titles: List[str]
    error: Optional[str] = None


class ReaderModeResponse(BaseModel):
    id: int
    title: str
    vietnamese_title: Optional[str] = None
    url: str
    author: Optional[str] = None
    source_name: Optional[str] = None
    published_at: Optional[datetime] = None
    reading_time_minutes: int
    content: str
    is_bookmarked: bool = False


# Feedback Schemas
class ArticleFeedbackCreate(BaseModel):
    feedback_type: str  # like, dislike, bookmark, read, hide
    source: Optional[str] = "web"
    notes: Optional[str] = None


class ArticleFeedbackResponse(BaseModel):
    id: int
    article_id: int
    user_id: str
    feedback_type: str
    source: str
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# User Preference Schemas
class UserPreferenceUpdate(BaseModel):
    topic_weights: Optional[Dict[str, float]] = None
    preferred_sources: Optional[List[int]] = None


class UserPreferenceResponse(BaseModel):
    user_id: str
    topic_weights: Dict[str, float] = {}
    preferred_sources: List[int] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Semantic Search Schemas
class SemanticSearchRequest(BaseModel):
    query: str
    topic: Optional[str] = None
    source_id: Optional[int] = None
    min_score: Optional[float] = 0.0
    limit: Optional[int] = 20


# Admin Metrics Schemas
class SourceHealthMetric(BaseModel):
    id: int
    name: str
    url: str
    source_type: str
    status: str
    articles_count: int
    last_crawled_at: Optional[datetime] = None
    last_error: Optional[str] = None
    success_rate: float = 100.0


class AdminMetricsResponse(BaseModel):
    total_articles: int
    analyzed_articles: int
    high_score_articles: int
    total_sources: int
    active_sources: int
    healthy_sources: int
    error_sources: int
    recent_runs: List[CrawlRunResponse] = []
    source_health: List[SourceHealthMetric] = []
    ai_model_distribution: Dict[str, int] = {}
    circuit_breakers_status: Dict[str, str] = {}
