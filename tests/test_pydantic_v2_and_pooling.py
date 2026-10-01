import pytest
from app.schemas.schemas import (
    SourceResponse,
    CrawlRunResponse,
    RelatedSourceArticle,
    ArticleResponse,
    ArticleFeedbackResponse,
    UserPreferenceResponse,
)
from app.core.config import settings
from app.core.database import engine
from sqlalchemy.pool import AsyncAdaptedQueuePool


def test_pydantic_v2_model_config():
    """Verify that models use Pydantic V2 model_config without deprecation warnings."""
    for model_cls in (
        SourceResponse,
        CrawlRunResponse,
        RelatedSourceArticle,
        ArticleResponse,
        ArticleFeedbackResponse,
        UserPreferenceResponse,
    ):
        assert hasattr(model_cls, "model_config"), f"{model_cls.__name__} missing model_config"
        assert model_cls.model_config.get("from_attributes") is True, f"{model_cls.__name__} from_attributes is not True"


def test_settings_config_and_pool_params():
    """Verify settings uses SettingsConfigDict and defines pool parameters."""
    assert hasattr(settings, "model_config")
    assert settings.DB_POOL_SIZE >= 5
    assert settings.DB_MAX_OVERFLOW >= 0
    assert settings.DB_POOL_TIMEOUT > 0
    assert settings.DB_POOL_RECYCLE > 0
    assert isinstance(settings.DB_POOL_PRE_PING, bool)


def test_database_connection_pooling():
    """Verify engine uses AsyncAdaptedQueuePool with configured settings."""
    assert isinstance(engine.pool, AsyncAdaptedQueuePool)
    assert engine.pool.size() == settings.DB_POOL_SIZE
