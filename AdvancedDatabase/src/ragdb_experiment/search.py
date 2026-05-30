from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from qdrant_client import models

from .config import Settings
from .db import connect_pgvector, qdrant_client, vector_literal


@dataclass(frozen=True)
class FilterSpec:
    category: str | None = None
    doc_type: str | None = None
    year_gte: int | None = None


def parse_filter(raw: dict[str, Any] | None) -> FilterSpec:
    raw = raw or {}
    return FilterSpec(
        category=raw.get("category"),
        doc_type=raw.get("doc_type"),
        year_gte=raw.get("year_gte"),
    )


def exact_search(
    docs: list[dict],
    query_vector: list[float],
    k: int,
    filters: FilterSpec | None = None,
) -> list[str]:
    filters = filters or FilterSpec()
    q = np.asarray(query_vector, dtype=np.float32)
    candidates = []
    for doc in docs:
        if filters.category and doc["category"] != filters.category:
            continue
        if filters.doc_type and doc["doc_type"] != filters.doc_type:
            continue
        if filters.year_gte and int(doc["year"]) < filters.year_gte:
            continue
        score = float(np.dot(q, np.asarray(doc["embedding"], dtype=np.float32)))
        candidates.append((score, doc["doc_id"]))
    candidates.sort(reverse=True)
    return [doc_id for _, doc_id in candidates[:k]]


def qdrant_filter(filters: FilterSpec | None) -> models.Filter | None:
    if not filters:
        return None
    conditions = []
    if filters.category:
        conditions.append(
            models.FieldCondition(key="category", match=models.MatchValue(value=filters.category))
        )
    if filters.doc_type:
        conditions.append(
            models.FieldCondition(key="doc_type", match=models.MatchValue(value=filters.doc_type))
        )
    if filters.year_gte:
        conditions.append(
            models.FieldCondition(key="year", range=models.Range(gte=filters.year_gte))
        )
    return models.Filter(must=conditions) if conditions else None


def search_qdrant(
    settings: Settings,
    query_vector: list[float],
    k: int,
    filters: FilterSpec | None = None,
) -> list[str]:
    response = qdrant_client(settings).query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=qdrant_filter(filters),
        limit=k,
        with_payload=True,
    )
    return [point.payload["doc_id"] for point in response.points]


def search_pgvector(
    settings: Settings,
    query_vector: list[float],
    k: int,
    filters: FilterSpec | None = None,
) -> list[str]:
    filters = filters or FilterSpec()
    clauses = []
    params: list[Any] = [vector_literal(query_vector)]
    if filters.category:
        clauses.append("category = %s")
        params.append(filters.category)
    if filters.doc_type:
        clauses.append("doc_type = %s")
        params.append(filters.doc_type)
    if filters.year_gte:
        clauses.append("year >= %s")
        params.append(filters.year_gte)
    params.append(k)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT doc_id
        FROM documents
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    # The distance parameter belongs in ORDER BY, so move it after WHERE params.
    ordered_params = params[1:-1] + [params[0], params[-1]]
    with connect_pgvector(settings) as conn:
        with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
            cur.execute(sql, ordered_params)
            return [row[0] for row in cur.fetchall()]
