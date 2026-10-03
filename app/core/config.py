import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to backend directory and .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

# Explicitly load .env into os.environ
load_dotenv(dotenv_path=ENV_PATH, override=True)


class Settings(BaseSettings):
    PROJECT_NAME: str = "TechPulse"

    # PostgreSQL Database Config
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5433"))
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASS: str = os.getenv("DB_PASS", "mypassword")
    DB_NAME: str = os.getenv("DB_NAME", "nexo_dev")
    DB_SSLMODE: str = os.getenv("DB_SSLMODE", "disable")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    DB_POOL_PRE_PING: bool = os.getenv("DB_POOL_PRE_PING", "true").lower() in ("true", "1", "yes")

    # 9routers AI gateway & Multi-model Fallback
    NINEROUTERS_BASE_URL: str = os.getenv(
        "NINEROUTERS_BASE_URL", "http://127.0.0.1:20128/v1"
    )
    NINEROUTERS_API_KEY: str = os.getenv("NINEROUTERS_API_KEY", "9router-local")
    AI_MODEL: str = os.getenv("AI_MODEL", "nexo-chat")
    AI_FALLBACK_MODELS: str = os.getenv(
        "AI_FALLBACK_MODELS", "gemini/gemini-3.8-flash,groq/openai/gpt-oss-120b"
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # AI Request Timeouts (Interactive/Chat vs Background/Crawl/Blueprint)
    AI_TIMEOUT_INTERACTIVE: float = float(os.getenv("AI_TIMEOUT_INTERACTIVE", "30.0"))
    AI_TIMEOUT_BACKGROUND: float = float(os.getenv("AI_TIMEOUT_BACKGROUND", "60.0"))

    # Scheduler
    CRAWL_INTERVAL_MINUTES: int = int(os.getenv("CRAWL_INTERVAL_MINUTES", "60"))

    # Webhooks & Urgent Alerts
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ALERT_THRESHOLD: float = float(
        os.getenv("TELEGRAM_ALERT_THRESHOLD", "9.0")
    )
    TELEGRAM_ALERT_MAX_PER_HOUR: int = int(
        os.getenv("TELEGRAM_ALERT_MAX_PER_HOUR", "5")
    )
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://127.0.0.1:5174")

    @property
    def fallback_models_list(self) -> list[str]:
        raw_list = [m.strip() for m in self.AI_FALLBACK_MODELS.split(",") if m.strip()]
        # Ensure AI_MODEL is always the primary model first in the resolution chain
        models = [self.AI_MODEL]
        for m in raw_list:
            if m not in models:
                models.append(m)
        return models

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        extra="ignore",
    )


settings = Settings()
