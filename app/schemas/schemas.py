from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Any
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
    articles_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TechStackItem(BaseModel):
    name: str
    category: Optional[str] = "Tool"
    desc: Optional[str] = ""

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
    vietnamese_title: Optional[str] = None
    vietnamese_summary: Optional[str] = None
    key_takeaways: List[str] = []
    new_tech_stacks: List[TechStackItem] = []
    tags: List[str] = []
    target_audience: List[str] = []
    ai_model_used: Optional[str] = None
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

class CrawlTestResult(BaseModel):
    success: bool
    detected_type: str
    feed_url: Optional[str] = None
    items_count: int
    sample_titles: List[str]
    error: Optional[str] = None
