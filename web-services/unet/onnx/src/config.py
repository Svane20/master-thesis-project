from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    MODEL_PATH: str = ""
    USE_GPU: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')


@lru_cache()
def get_configuration() -> Settings:
    return Settings()
