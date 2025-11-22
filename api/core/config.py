from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MinioSettings(BaseSettings):
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False


class ApiSettings(BaseSettings):
    ENABLE_DOCS: bool = False

    MODEL_BUCKET: str
    MODEL_FILE: str
    AUDIO_BUCKET: str
    MODEL_LOCAL_PATH: str


# ----------------------------------------------- #


class Settings(MinioSettings, ApiSettings, BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )

    @property
    def MINIO_BUCKET(self) -> str:
        """Alias for AUDIO_BUCKET for connection verification."""
        return self.AUDIO_BUCKET


settings = Settings()

# ----------------------------------------------- #
