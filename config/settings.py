import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "AI Research & Knowledge Assistant"
    API_V1_STR: str = "/api/v1"
    
    # Storage Paths
    RAW_DOCUMENTS_DIR: Path = BASE_DIR / "data" / "raw_documents"
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "vector_db"
    DATASET_DIR: Path = BASE_DIR / "data" / "dataset"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Model Artifacts
    TF_MODEL_PATH: Path = BASE_DIR / "models" / "tf_classifier.h5"
    TOKENIZER_PATH: Path = BASE_DIR / "models" / "tokenizer.pickle"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'assistant.db'}"
    
    # AI / LLM Configuration
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "gpt-4o"
    
    # Chunking Hyperparameters
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self):
        """Ensure all required runtime directories exist."""
        for path in [self.RAW_DOCUMENTS_DIR, self.VECTOR_DB_DIR, self.DATASET_DIR, self.MODELS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
