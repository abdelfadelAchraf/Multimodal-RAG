
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Values are loaded from environment variables or .env.
    """

    APP_NAME: str = "Multimodal rag"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    API_V1_PREFIX: str = "/api/v1"

    # AI configuration
    GOOGLE_API_KEY: str = ""

    MAX_UPLOAD_SIZE_MB: int = 20

    # Storage configuration
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma_db"
    UPLOAD_DIRECTORY: str = "data/uploads"
    EXTRACTED_DIRECTORY: str = "data/extracted"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100 # character shared between two consecutive chunks , that is to preserve the meaning/context between chunks

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
