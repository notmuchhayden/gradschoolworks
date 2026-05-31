from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import models

from .config import Settings
from .db import (
    connect_meta,
    connect_pgvector,
    qdrant_client,
    vector_literal,
)
from .embeddings import Embedder
from .io import read_jsonl, write_jsonl
from .search import search_qdrant


def make_modified_docs(docs: list[dict], ratio: float) -> list[dict]:
    count = max(1, int(len(docs) * ratio))
    modified: list[dict] = []
    for doc in docs[:count]:
        new_doc = dict(doc)
        new_doc["content"] = (
            "Zero trust authentication failure response procedure. This document "
            "replaces the previous access issue article and requires checking "
            "authentication policy, device trust, and access control logs. "
            f"The original document identifier is {doc['doc_id']}."
        )
        new_doc["embedding_version"] = int(doc["embedding_version"]) + 1
        new_doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        modified.append(new_doc)
    return modified


def run_sync_experiment(
    settings: Settings,
    embedded_documents_path: Path,
    output_path: Path,
    delay_seconds: float,
    modify_ratio: float,
    model_name: str,
    dim: int,
    mock: bool,
    delay_probe_interval_seconds: float = 5.0,
    delay_max_probes: int = 12,
    delay_probe_docs: int = 20,
) -> None:
    docs = list(read_jsonl(embedded_documents_path))
    modified = make_modified_docs(docs, modify_ratio)
    original_by_doc_id = {doc["doc_id"]: doc for doc in docs}

    embedder = Embedder(model_name=model_name, dim=dim, mock=mock)
    vectors = embedder.encode([doc["content"] for doc in modified])
    for doc, vector in zip(modified, vectors):
        doc["embedding"] = vector.astype(float).tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path.with_suffix(".modified.jsonl"), modified)

    qdrant_update_started = time.perf_counter()
    _update_meta_only(settings, modified)
    stale_ratio_before = _measure_qdrant_stale_vector_ratio(settings, modified)
    old_meaning_hits_before = _measure_qdrant_old_meaning_hits(
        settings, original_by_doc_id, modified
    )
    new_meaning_hits_before = _measure_qdrant_new_meaning_hits(settings, modified)

    delay_window = _measure_qdrant_delay_window(
        settings=settings,
        original_by_doc_id=original_by_doc_id,
        modified=modified,
        delay_seconds=delay_seconds,
        probe_interval_seconds=delay_probe_interval_seconds,
        max_probes=delay_max_probes,
        probe_docs=delay_probe_docs,
    )

    qdrant_vector_update_started = time.perf_counter()
    _update_qdrant_vectors(settings, docs, modified)
    qdrant_vector_update_latency_ms = (time.perf_counter() - qdrant_vector_update_started) * 1000.0
    qdrant_update_latency_ms = (time.perf_counter() - qdrant_update_started) * 1000.0
    stale_ratio_after = _measure_qdrant_stale_vector_ratio(settings, modified)
    old_meaning_hits_after = _measure_qdrant_old_meaning_hits(
        settings, original_by_doc_id, modified
    )
    new_meaning_hits_after = _measure_qdrant_new_meaning_hits(settings, modified)

    pgvector_started = time.perf_counter()
    _update_pgvector(settings, modified)
    pgvector_update_latency_ms = (time.perf_counter() - pgvector_started) * 1000.0

    checked_docs = len(modified[: min(100, len(modified))])
    rows = [
        {
            "engine": "qdrant_split",
            "delay_seconds": delay_seconds,
            "modified_docs": len(modified),
            "checked_docs": checked_docs,
            "stale_vector_ratio_before_sync": stale_ratio_before,
            "stale_vector_ratio_after_sync": stale_ratio_after,
            "old_meaning_retrieval_count_before_sync": old_meaning_hits_before,
            "old_meaning_retrieval_count_after_sync": old_meaning_hits_after,
            "new_meaning_retrieval_count_before_sync": new_meaning_hits_before,
            "new_meaning_retrieval_count_after_sync": new_meaning_hits_after,
            "delay_probe_interval_seconds": delay_probe_interval_seconds,
            "delay_probe_docs": delay_probe_docs,
            "delay_probe_count": delay_window["probe_count"],
            "delay_observed_seconds": delay_window["observed_seconds"],
            "delay_old_meaning_retrieval_total": delay_window["old_meaning_total"],
            "delay_new_meaning_retrieval_total": delay_window["new_meaning_total"],
            "qdrant_vector_update_latency_ms": qdrant_vector_update_latency_ms,
            "update_latency_ms": qdrant_update_latency_ms,
        },
        {
            "engine": "pgvector_integrated",
            "delay_seconds": 0,
            "modified_docs": len(modified),
            "checked_docs": checked_docs,
            "stale_vector_ratio_before_sync": 0.0,
            "stale_vector_ratio_after_sync": 0.0,
            "old_meaning_retrieval_count_before_sync": 0,
            "old_meaning_retrieval_count_after_sync": 0,
            "new_meaning_retrieval_count_before_sync": checked_docs,
            "new_meaning_retrieval_count_after_sync": checked_docs,
            "delay_probe_interval_seconds": 0.0,
            "delay_probe_docs": 0,
            "delay_probe_count": 0,
            "delay_observed_seconds": 0.0,
            "delay_old_meaning_retrieval_total": 0,
            "delay_new_meaning_retrieval_total": 0,
            "qdrant_vector_update_latency_ms": 0.0,
            "update_latency_ms": pgvector_update_latency_ms,
        },
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _measure_qdrant_delay_window(
    settings: Settings,
    original_by_doc_id: dict[str, dict],
    modified: list[dict],
    delay_seconds: float,
    probe_interval_seconds: float,
    max_probes: int,
    probe_docs: int,
) -> dict[str, float | int]:
    if delay_seconds <= 0:
        return {
            "probe_count": 0,
            "observed_seconds": 0.0,
            "old_meaning_total": 0,
            "new_meaning_total": 0,
        }

    deadline = time.perf_counter() + delay_seconds
    interval = max(0.1, probe_interval_seconds)
    probe_limit = max(1, max_probes)
    probe_count = 0
    old_meaning_total = 0
    new_meaning_total = 0
    started = time.perf_counter()

    while time.perf_counter() < deadline and probe_count < probe_limit:
        old_meaning_total += _measure_qdrant_old_meaning_hits(
            settings, original_by_doc_id, modified, check_limit=probe_docs
        )
        new_meaning_total += _measure_qdrant_new_meaning_hits(
            settings, modified, check_limit=probe_docs
        )
        probe_count += 1

        remaining = deadline - time.perf_counter()
        if remaining <= 0 or probe_count >= probe_limit:
            break
        time.sleep(min(interval, remaining))

    observed_seconds = time.perf_counter() - started
    return {
        "probe_count": probe_count,
        "observed_seconds": observed_seconds,
        "old_meaning_total": old_meaning_total,
        "new_meaning_total": new_meaning_total,
    }


def _update_meta_only(settings: Settings, modified: list[dict]) -> None:
    with connect_meta(settings) as conn:
        with conn.cursor() as cur:
            for doc in modified:
                cur.execute(
                    """
                    UPDATE documents
                    SET content = %s, updated_at = %s, embedding_version = %s
                    WHERE doc_id = %s
                    """,
                    (doc["content"], doc["updated_at"], doc["embedding_version"], doc["doc_id"]),
                )


def _update_qdrant_vectors(settings: Settings, original_docs: list[dict], modified: list[dict]) -> None:
    id_by_doc_id = {doc["doc_id"]: idx for idx, doc in enumerate(original_docs)}
    points = [
        models.PointStruct(
            id=id_by_doc_id[doc["doc_id"]],
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
        for doc in modified
    ]
    qdrant_client(settings).upsert(collection_name=settings.qdrant_collection, points=points)


def _update_pgvector(settings: Settings, modified: list[dict]) -> None:
    with connect_pgvector(settings) as conn:
        with conn.cursor() as cur:
            for doc in modified:
                cur.execute(
                    """
                    UPDATE documents
                    SET content = %s,
                        updated_at = %s,
                        embedding_version = %s,
                        embedding = %s::vector
                    WHERE doc_id = %s
                    """,
                    (
                        doc["content"],
                        doc["updated_at"],
                        doc["embedding_version"],
                        vector_literal(doc["embedding"]),
                        doc["doc_id"],
                    ),
                )


def _measure_qdrant_stale_vector_ratio(settings: Settings, modified: list[dict]) -> float:
    if not modified:
        return 0.0

    stale_count = 0
    client = qdrant_client(settings)
    for doc in modified:
        response = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc["doc_id"])
                    )
                ]
            ),
            with_payload=True,
            limit=1,
        )
        points = response[0]
        if not points:
            stale_count += 1
            continue
        stored_version = int(points[0].payload["embedding_version"])
        if stored_version < int(doc["embedding_version"]):
            stale_count += 1
    return stale_count / len(modified)


def _measure_qdrant_old_meaning_hits(
    settings: Settings,
    original_by_doc_id: dict[str, dict],
    modified: list[dict],
    check_limit: int = 100,
) -> int:
    hit_count = 0
    for doc in modified[: min(check_limit, len(modified))]:
        original_doc = original_by_doc_id[doc["doc_id"]]
        hits = search_qdrant(settings, original_doc["embedding"], k=10)
        if doc["doc_id"] in hits:
            hit_count += 1
    return hit_count


def _measure_qdrant_new_meaning_hits(
    settings: Settings, modified: list[dict], check_limit: int = 100
) -> int:
    hit_count = 0
    for doc in modified[: min(check_limit, len(modified))]:
        hits = search_qdrant(settings, doc["embedding"], k=10)
        if doc["doc_id"] in hits:
            hit_count += 1
    return hit_count
