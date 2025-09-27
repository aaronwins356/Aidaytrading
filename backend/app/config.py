"""Application configuration module."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl, EmailStr, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field(default="Aiday Trading Backend")
    debug: bool = Field(default=False)
    database_url: str = Field(default="sqlite+aiosqlite:///./db.sqlite")
    jwt_secret: str = Field(default="0123456789abcdef0123456789abcdef", min_length=32)
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_expires: int = Field(default=900)
    jwt_refresh_expires: int = Field(default=60 * 60 * 24 * 7)
    brevo_api_key: str = Field(default="", env="BREVO_API_KEY")
    owner_email: EmailStr = Field(default="owner@example.com")
    firebase_credentials_path: str = Field(default="serviceAccountKey.json")
    timezone: str = Field(default="America/Chicago")
    cors_origins: List[AnyHttpUrl] = Field(default_factory=list)

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
