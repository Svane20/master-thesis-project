from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    MODEL_PATH: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')


@lru_cache()
def get_configuration() -> Settings:
    return Settings()
