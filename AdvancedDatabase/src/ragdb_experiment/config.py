from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Qdrant 분리형 구조에서 사용할 PostgreSQL DSN
    postgres_meta_dsn: str = os.getenv(
        "POSTGRES_META_DSN", "postgresql://rag:rag@localhost:5433/rag_meta")
    
    # pgvector 통합형 구조에서 사용할 PostgreSQL DSN
    postgres_pgvector_dsn: str = os.getenv(
        "POSTGRES_PGVECTOR_DSN", "postgresql://rag:rag@localhost:5434/rag_vector")
    
    # Qdrant 설정
    qdrant_url: str = os.getenv(
        "QDRANT_URL", "http://localhost:6333")
    
    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION", "documents")
    
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    embedding_dim: int = int(os.getenv(
        "EMBEDDING_DIM", "384"))

    experiment_documents: int = int(os.getenv(
        "EXPERIMENT_DOCUMENTS", "1000"))

    experiment_repeats: int = int(os.getenv(
        "EXPERIMENT_REPEATS", "30"))

    experiment_warmup: int = int(os.getenv(
        "EXPERIMENT_WARMUP", "1"))


def get_settings() -> Settings:
    return Settings()
