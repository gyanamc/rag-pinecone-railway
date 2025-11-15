from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_index_name: str = "rag-index"
    pinecone_environment: Optional[str] = None
    
    # OpenAI Configuration
    openai_api_key: str
    
    # Application Configuration
    api_port: int = 8000
    environment: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
