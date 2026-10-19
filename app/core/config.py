"""Application configuration settings using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration settings for Crawlbot.

    Loads configuration from environment variables prefixed with CRAWLBOT_
    or from a local .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CRAWLBOT_",
        extra="ignore",
    )

    # App
    app_name: str = "Crawlbot"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    # Crawling
    max_crawl_depth: int = 3
    max_pages: int = 500
    request_timeout: float = 30.0
    max_concurrency: int = 5
    delay_between_requests: float = 1.0
    max_response_size: int = 10 * 1024 * 1024  # 10MB

    # Extraction
    default_extraction_strategy: str = "trafilatura"  # or 'bs4'

    # Output
    output_dir: str = "output"
    log_level: str = "INFO"

    # OpenAI / AI Features
    openai_api_key: str = ""
    openai_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    ai_concurrency: int = 3


    # URL filtering
    default_exclude_patterns: list[str] = [
        r"/wp-admin/",
        r"/wp-login",
        r"/admin/",
        r"/login",
        r"/logout",
        r"/cart",
        r"/checkout",
        r"/my-account",
        r"\.pdf$",
        r"\.zip$",
        r"\.exe$",
        r"\.dmg$",
        r"\.jpg$",
        r"\.jpeg$",
        r"\.png$",
        r"\.gif$",
        r"\.svg$",
        r"\.mp3$",
        r"\.mp4$",
        r"\.avi$",
        r"\.mov$",
        r"\?replytocom=",
        r"#respond",
        r"/feed/",
        r"/rss/",
        r"/xmlrpc\.php",
        r"/wp-json/",
    ]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton instance.

    Returns:
        Settings: Application configuration instance.
    """
    return Settings()
