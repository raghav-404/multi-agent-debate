import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    database_url: str | None
    retry_threshold: float


def get_settings() -> Settings:
    threshold = float(os.getenv("RETRY_THRESHOLD", "0.6"))
    if not 0 <= threshold <= 1:
        raise ValueError("RETRY_THRESHOLD must be between 0 and 1")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    if not model:
        raise ValueError("GROQ_MODEL must not be empty")
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=model,
        database_url=os.getenv("DATABASE_URL") or None,
        retry_threshold=threshold,
    )
