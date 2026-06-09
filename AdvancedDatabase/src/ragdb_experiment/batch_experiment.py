from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qdrant_client import models

from .config import Settings
from .db import connect_meta, connect_pgvector, qdrant_client, vector_literal
from .embeddings import Embedder
from .io import read_jsonl, write_jsonl
from .sync_experiment import make_modified_docs


def run_batch_experiment(
    settings: Settings,
    engine: str,
    embedded_documents_path: Path,
    output_path: Path,
    documents: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    dim: int,
) -> None:
    if documents <= 0:
        raise ValueError("documents must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch size must be greater than zero")
    if repeats <= 0:
        raise ValueError("repeats must be greater than zero")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    all_docs = list(read_jsonl(embedded_documents_path))
    if documents > len(all_docs):
        raise ValueError(
            f"requested {documents} documents, but only {len(all_docs)} are available"
        )

    original = all_docs[:documents]
    modified = make_modified_docs(original, ratio=1.0)
    timestamp = datetime.now(timezone.utc).isoformat()
    for doc in modified:
        doc["updated_at"] = timestamp

    embedder = Embedder(model_name="mock", dim=dim, mock=True)
    vectors = embedder.encode([doc["content"] for doc in modified])
    for doc, vector in zip(modified, vectors):
        doc["embedding"] = vector.astype(float).tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path.with_suffix(".modified.jsonl"), modified)

    id_by_doc_id = {doc["doc_id"]: idx for idx, doc in enumerate(all_docs)}
    rows: list[dict[str, str | int | float]] = []
    total_runs = warmup + repeats

    for run_index in range(total_runs):
        if engine == "qdrant":
            _restore_qdrant_state(settings, original, id_by_doc_id)
            metrics = _run_qdrant_batch(
                settings, modified, id_by_doc_id, batch_size
            )
        elif engine == "pgvector":
            _restore_pgvector_state(settings, original)
            metrics = _run_pgvector_batch(settings, modified, batch_size)
        else:
            raise ValueError(f"unsupported engine: {engine}")

        if run_index < warmup:
            continue

        rows.append(
            {
                "engine": engine,
                "batch_size": batch_size,
                "repeat": run_index - warmup + 1,
                "documents": documents,
                "update_operations": (documents + batch_size - 1) // batch_size,
                **metrics,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_qdrant_batch(
    settings: Settings,
    modified: list[dict],
    id_by_doc_id: dict[str, int],
    batch_size: int,
) -> dict[str, float]:
    client = qdrant_client(settings)
    visibility_latencies_ms: list[float] = []
    pending_docs: list[dict] = []
    pending_started: list[float] = []
    total_started = time.perf_counter()

    with connect_meta(settings) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for doc in modified:
                event_started = time.perf_counter()
                cur.execute(
                    """
                    UPDATE documents
                    SET content = %s, updated_at = %s, embedding_version = %s
                    WHERE doc_id = %s
                    """,
                    (
                        doc["content"],
                        doc["updated_at"],
                        doc["embedding_version"],
                        doc["doc_id"],
                    ),
                )
                pending_docs.append(doc)
                pending_started.append(event_started)

                if len(pending_docs) >= batch_size:
                    _flush_qdrant_batch(
                        settings,
                        client,
                        pending_docs,
                        pending_started,
                        id_by_doc_id,
                        visibility_latencies_ms,
                    )

            if pending_docs:
                _flush_qdrant_batch(
                    settings,
                    client,
                    pending_docs,
                    pending_started,
                    id_by_doc_id,
                    visibility_latencies_ms,
                )

    total_processing_time_ms = (time.perf_counter() - total_started) * 1000.0
    consistency_error_ratio = _qdrant_consistency_error_ratio(
        settings, modified, id_by_doc_id
    )
    return {
        "total_processing_time_ms": total_processing_time_ms,
        "document_visibility_latency_p95_ms": float(
            np.percentile(visibility_latencies_ms, 95)
        ),
        "consistency_error_ratio_after_update": consistency_error_ratio,
    }


def _flush_qdrant_batch(
    settings: Settings,
    client,
    pending_docs: list[dict],
    pending_started: list[float],
    id_by_doc_id: dict[str, int],
    visibility_latencies_ms: list[float],
) -> None:
    points = [
        _qdrant_point(doc, id_by_doc_id[doc["doc_id"]]) for doc in pending_docs
    ]
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )
    completed = time.perf_counter()
    visibility_latencies_ms.extend(
        (completed - started) * 1000.0 for started in pending_started
    )
    pending_docs.clear()
    pending_started.clear()


def _run_pgvector_batch(
    settings: Settings,
    modified: list[dict],
    batch_size: int,
) -> dict[str, float]:
    visibility_latencies_ms: list[float] = []
    pending_docs: list[dict] = []
    pending_started: list[float] = []
    total_started = time.perf_counter()

    with connect_pgvector(settings) as conn:
        for doc in modified:
            pending_docs.append(doc)
            pending_started.append(time.perf_counter())
            if len(pending_docs) >= batch_size:
                _flush_pgvector_batch(
                    conn, pending_docs, pending_started, visibility_latencies_ms
                )

        if pending_docs:
            _flush_pgvector_batch(
                conn, pending_docs, pending_started, visibility_latencies_ms
            )

    total_processing_time_ms = (time.perf_counter() - total_started) * 1000.0
    consistency_error_ratio = _pgvector_consistency_error_ratio(settings, modified)
    return {
        "total_processing_time_ms": total_processing_time_ms,
        "document_visibility_latency_p95_ms": float(
            np.percentile(visibility_latencies_ms, 95)
        ),
        "consistency_error_ratio_after_update": consistency_error_ratio,
    }


def _flush_pgvector_batch(
    conn,
    pending_docs: list[dict],
    pending_started: list[float],
    visibility_latencies_ms: list[float],
) -> None:
    rows = [
        (
            doc["content"],
            doc["updated_at"],
            doc["embedding_version"],
            vector_literal(doc["embedding"]),
            doc["doc_id"],
        )
        for doc in pending_docs
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE documents
            SET content = %s,
                updated_at = %s,
                embedding_version = %s,
                embedding = %s::vector
            WHERE doc_id = %s
            """,
            rows,
        )
    conn.commit()
    completed = time.perf_counter()
    visibility_latencies_ms.extend(
        (completed - started) * 1000.0 for started in pending_started
    )
    pending_docs.clear()
    pending_started.clear()


def _restore_qdrant_state(
    settings: Settings,
    original: list[dict],
    id_by_doc_id: dict[str, int],
) -> None:
    rows = [
        (
            doc["content"],
            doc["updated_at"],
            doc["embedding_version"],
            doc["doc_id"],
        )
        for doc in original
    ]
    with connect_meta(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE documents
                SET content = %s, updated_at = %s, embedding_version = %s
                WHERE doc_id = %s
                """,
                rows,
            )

    points = [_qdrant_point(doc, id_by_doc_id[doc["doc_id"]]) for doc in original]
    qdrant_client(settings).upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )


def _restore_pgvector_state(settings: Settings, original: list[dict]) -> None:
    rows = [
        (
            doc["content"],
            doc["updated_at"],
            doc["embedding_version"],
            vector_literal(doc["embedding"]),
            doc["doc_id"],
        )
        for doc in original
    ]
    with connect_pgvector(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE documents
                SET content = %s,
                    updated_at = %s,
                    embedding_version = %s,
                    embedding = %s::vector
                WHERE doc_id = %s
                """,
                rows,
            )


def _qdrant_point(doc: dict, point_id: int) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
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


def _qdrant_consistency_error_ratio(
    settings: Settings,
    modified: list[dict],
    id_by_doc_id: dict[str, int],
) -> float:
    client = qdrant_client(settings)
    expected = {doc["doc_id"]: int(doc["embedding_version"]) for doc in modified}
    actual: dict[str, int] = {}
    point_ids = [id_by_doc_id[doc["doc_id"]] for doc in modified]

    for start in range(0, len(point_ids), 256):
        points = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=point_ids[start : start + 256],
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            doc_id = str(point.payload["doc_id"])
            actual[doc_id] = int(point.payload["embedding_version"])

    errors = sum(actual.get(doc_id) != version for doc_id, version in expected.items())
    return errors / len(expected)


def _pgvector_consistency_error_ratio(
    settings: Settings, modified: list[dict]
) -> float:
    expected = {doc["doc_id"]: int(doc["embedding_version"]) for doc in modified}
    with connect_pgvector(settings) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, embedding_version
            FROM documents
            WHERE doc_id = ANY(%s)
            """,
            (list(expected),),
        ).fetchall()
    actual = {str(doc_id): int(version) for doc_id, version in rows}
    errors = sum(actual.get(doc_id) != version for doc_id, version in expected.items())
    return errors / len(expected)
