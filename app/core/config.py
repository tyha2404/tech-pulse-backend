import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "TechPulse"

    # PostgreSQL Database Config
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5433"))
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASS: str = os.getenv("DB_PASS", "mypassword")
    DB_NAME: str = os.getenv("DB_NAME", "nexo_dev")
    DB_SSLMODE: str = os.getenv("DB_SSLMODE", "disable")

    # 9routers AI gateway
    NINEROUTERS_BASE_URL: str = os.getenv("NINEROUTERS_BASE_URL", "http://127.0.0.1:20128/v1")
    NINEROUTERS_API_KEY: str = os.getenv("NINEROUTERS_API_KEY", "9router-local")
    AI_MODEL: str = os.getenv("AI_MODEL", "gemini-2.5-flash")

    # Scheduler
    CRAWL_INTERVAL_MINUTES: int = 60

    # Webhooks (Optional)
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    @property
    def DATABASE_URL(self) -> str:
        # Construct asyncpg connection string for PostgreSQL
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
