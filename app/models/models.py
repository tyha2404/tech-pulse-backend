import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    url = Column(String(1024), unique=True, nullable=False)
    feed_url = Column(String(1024), nullable=True)  # If discovered RSS feed
    source_type = Column(String(50), default="rss")  # rss, api, scraper
    category = Column(
        String(100), default="General"
    )  # AI, Backend, General, VN, Global
    is_active = Column(Boolean, default=True)

    # Health Monitoring
    status = Column(String(50), default="healthy")  # healthy, warning, error, pending
    last_crawled_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    articles_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    articles = relationship(
        "Article", back_populates="source", cascade="all, delete-orphan"
    )


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1024), unique=True, nullable=False, index=True)
    author = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=True)
    raw_content = Column(Text, nullable=True)

    # AI Analysis Output via 9routers
    is_processed = Column(Boolean, default=False)
    is_worth_reading = Column(Boolean, default=False)  # Filter out noise
    relevance_score = Column(Float, default=0.0)  # 1 - 10

    vietnamese_title = Column(String(500), nullable=True)
    vietnamese_summary = Column(Text, nullable=True)
    key_takeaways = Column(JSON, default=list)  # List of strings
    new_tech_stacks = Column(
        JSON, default=list
    )  # List of objects: [{name, category, desc}]
    tags = Column(JSON, default=list)  # ["Backend", "AI", "PostgreSQL", ...]
    target_audience = Column(JSON, default=list)
    ai_model_used = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    source = relationship("Source", back_populates="articles")
