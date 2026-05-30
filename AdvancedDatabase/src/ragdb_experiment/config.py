from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    postgres_meta_dsn: str = os.getenv(
        "POSTGRES_META_DSN", "postgresql://rag:rag@localhost:5433/rag_meta"
    )
    postgres_pgvector_dsn: str = os.getenv(
        "POSTGRES_PGVECTOR_DSN", "postgresql://rag:rag@localhost:5434/rag_vector"
    )
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documents")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))


def get_settings() -> Settings:
    return Settings()
