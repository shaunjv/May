"""Application configuration using Pydantic settings."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="Personal AI Agent", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Agent settings (placeholders for future components)
    # These are intentionally left minimal for the foundation
    max_concurrent_tasks: int = Field(default=10, env="MAX_CONCURRENT_TASKS")
    task_timeout_seconds: int = Field(default=300, env="TASK_TIMEOUT_SECONDS")

    # Required only when the CLI constructs the hosted NVIDIA provider.
    nvidia_api_key: SecretStr | None = Field(default=None, env="NVIDIA_API_KEY")
    nvidia_model: str | None = Field(default=None, env="NVIDIA_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
