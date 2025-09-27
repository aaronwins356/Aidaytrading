"""Application configuration management."""
from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from the environment."""

    db_url: str = Field(validation_alias="DB_URL")
    jwt_secret: SecretStr = Field(validation_alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    access_token_expires_min: int = Field(default=15, ge=1, validation_alias="ACCESS_TOKEN_EXPIRES_MIN")
    refresh_token_expires_days: int = Field(default=7, ge=1, validation_alias="REFRESH_TOKEN_EXPIRES_DAYS")
    env: Literal["local", "dev", "prod"] = Field(default="local", validation_alias="ENV")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost", "http://localhost:3000"],
        validation_alias="CORS_ALLOWED_ORIGINS",
    )
    brevo_api_key: SecretStr = Field(validation_alias="BREVO_API_KEY")
    brevo_smtp_server: str = Field(validation_alias="BREVO_SMTP_SERVER")
    brevo_port: int = Field(validation_alias="BREVO_PORT")
    brevo_sender_email: str = Field(validation_alias="BREVO_SENDER_EMAIL")
    brevo_sender_name: str = Field(validation_alias="BREVO_SENDER_NAME")
    admin_notification_email: str = Field(validation_alias="ADMIN_NOTIFICATION_EMAIL")
    baseline_equity: Decimal | None = Field(default=None, validation_alias="BASELINE_EQUITY")
    firebase_credentials: str | None = Field(default=None, validation_alias="FIREBASE_CREDENTIALS")
    equity_snapshot_interval_minutes: int = Field(
        default=10,
        ge=1,
        validation_alias="EQUITY_SNAPSHOT_INTERVAL_MINUTES",
    )
    scheduler_enabled: bool = Field(default=True, validation_alias="SCHEDULER_ENABLED")
    heartbeat_cron: str = Field(
        default="0 2,8,14,20 * * *",
        validation_alias="HEARTBEAT_CRON",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> list[str]:
        if isinstance(value, str):
            origins = [item.strip() for item in value.split(",") if item.strip()]
            return origins or ["http://localhost", "http://localhost:3000"]
        return value  # type: ignore[return-value]

    @property
    def is_local(self) -> bool:
        return self.env == "local"

    def require_production_secrets(self) -> None:
        if self.is_local:
            return
        if self.jwt_secret.get_secret_value() in {"", "CHANGE_ME"}:
            raise ValueError("JWT_SECRET must be set to a secure value in non-local environments.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    settings = Settings()
    settings.require_production_secrets()
    return settings


settings = get_settings()
