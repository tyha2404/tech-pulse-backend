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

    # 9routers AI gateway & Multi-model Fallback
    NINEROUTERS_BASE_URL: str = os.getenv(
        "NINEROUTERS_BASE_URL", "http://127.0.0.1:20128/v1"
    )
    NINEROUTERS_API_KEY: str = os.getenv("NINEROUTERS_API_KEY", "9router-local")
    AI_MODEL: str = os.getenv("AI_MODEL", "gemini-2.5-flash")
    AI_FALLBACK_MODELS: str = os.getenv(
        "AI_FALLBACK_MODELS",
        "groq/openai/gpt-oss-120b,openrouter/openrouter/free,gemini/gemini-3.8-flash,gemini-2.5-flash,claude-3-5-haiku,gpt-4o-mini",
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # TypeSafe AI (Jev) System One Config
    TYPESAFE_BASE_URL: str = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
    TYPESAFE_API_KEY: str = os.getenv("TYPESAFE_API_KEY", "")

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

    FAST_TASK_MODELS: str = os.getenv(
        "FAST_TASK_MODELS", "groq/openai/gpt-oss-120b,openrouter/openrouter/free"
    )
    DEEP_TASK_MODELS: str = os.getenv(
        "DEEP_TASK_MODELS", "nexo-chat,groq/openai/gpt-oss-120b,gemini/gemini-3.8-flash"
    )

    @property
    def fallback_models_list(self) -> list[str]:
        models = [self.AI_MODEL]
        if self.AI_FALLBACK_MODELS:
            for m in self.AI_FALLBACK_MODELS.split(","):
                m_str = m.strip()
                if m_str and m_str not in models:
                    models.append(m_str)
        return models

    @property
    def fast_models_list(self) -> list[str]:
        models = []
        if self.FAST_TASK_MODELS:
            for m in self.FAST_TASK_MODELS.split(","):
                m_str = m.strip()
                if m_str and m_str not in models:
                    models.append(m_str)
        # Ensure fallback coverage
        for m in self.fallback_models_list:
            if m not in models:
                models.append(m)
        return models

    @property
    def deep_models_list(self) -> list[str]:
        models = []
        if self.DEEP_TASK_MODELS:
            for m in self.DEEP_TASK_MODELS.split(","):
                m_str = m.strip()
                if m_str and m_str not in models:
                    models.append(m_str)
        # Ensure fallback coverage
        for m in self.fallback_models_list:
            if m not in models:
                models.append(m)
        return models

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        extra="ignore"
    )


settings = Settings()
