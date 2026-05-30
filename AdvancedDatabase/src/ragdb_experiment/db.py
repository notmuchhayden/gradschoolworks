from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import psycopg
from qdrant_client import QdrantClient, models

from .config import Settings
from .io import read_jsonl


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


def connect_meta(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.postgres_meta_dsn)


def connect_pgvector(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.postgres_pgvector_dsn)


def qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def wait_for_services(settings: Settings, timeout_seconds: int = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with connect_meta(settings) as conn:
                conn.execute("SELECT 1")
            with connect_pgvector(settings) as conn:
                conn.execute("SELECT 1")
            qdrant_client(settings).get_collections()
            return
        except Exception as exc:  # noqa: BLE001 - printed as readiness detail.
            last_error = exc
            time.sleep(2)

    raise RuntimeError(f"services did not become ready: {last_error}")


def reset_postgres_tables(settings: Settings) -> None:
    for connector in (connect_meta, connect_pgvector):
        with connector(settings) as conn:
            conn.execute("TRUNCATE TABLE document_updates, documents RESTART IDENTITY")


def recreate_qdrant_collection(settings: Settings) -> None:
    client = qdrant_client(settings)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim,
            distance=models.Distance.COSINE,
        ),
    )
    for field, schema in [
        ("category", models.PayloadSchemaType.KEYWORD),
        ("doc_type", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER),
        ("embedding_version", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field,
            field_schema=schema,
        )


def load_postgres_meta(settings: Settings, docs: Iterable[dict]) -> None:
    rows = [
        (
            doc["doc_id"],
            doc["title"],
            doc["content"],
            doc["category"],
            doc["year"],
            doc["doc_type"],
            doc["updated_at"],
            doc["embedding_version"],
        )
        for doc in docs
    ]
    with connect_meta(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO documents
                    (doc_id, title, content, category, year, doc_type, updated_at, embedding_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    year = EXCLUDED.year,
                    doc_type = EXCLUDED.doc_type,
                    updated_at = EXCLUDED.updated_at,
                    embedding_version = EXCLUDED.embedding_version
                """,
                rows,
            )


def load_pgvector(settings: Settings, docs: Iterable[dict]) -> None:
    rows = [
        (
            doc["doc_id"],
            doc["title"],
            doc["content"],
            doc["category"],
            doc["year"],
            doc["doc_type"],
            doc["updated_at"],
            doc["embedding_version"],
            vector_literal(doc["embedding"]),
        )
        for doc in docs
    ]
    with connect_pgvector(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO documents
                    (doc_id, title, content, category, year, doc_type, updated_at, embedding_version, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    year = EXCLUDED.year,
                    doc_type = EXCLUDED.doc_type,
                    updated_at = EXCLUDED.updated_at,
                    embedding_version = EXCLUDED.embedding_version,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )


def load_qdrant(settings: Settings, docs: Iterable[dict], batch_size: int = 256) -> None:
    client = qdrant_client(settings)
    batch: list[models.PointStruct] = []
    for idx, doc in enumerate(docs):
        batch.append(
            models.PointStruct(
                id=idx,
                vector=doc["embedding"],
                payload={
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "year": doc["year"],
                    "doc_type": doc["doc_type"],
                    "embedding_version": doc["embedding_version"],
                },
            )
        )
        if len(batch) >= batch_size:
            client.upsert(collection_name=settings.qdrant_collection, points=batch)
            batch = []
    if batch:
        client.upsert(collection_name=settings.qdrant_collection, points=batch)


def load_all(settings: Settings, embedded_documents_path: Path, reset: bool) -> None:
    docs = list(read_jsonl(embedded_documents_path))
    if reset:
        reset_postgres_tables(settings)
        recreate_qdrant_collection(settings)
    load_postgres_meta(settings, docs)
    load_pgvector(settings, docs)
    load_qdrant(settings, docs)
