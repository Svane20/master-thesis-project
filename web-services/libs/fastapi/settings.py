from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    USE_GPU: bool = True
    MAX_BATCH_SIZE: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')


@lru_cache()
def get_settings() -> Settings:
    return Settings()
