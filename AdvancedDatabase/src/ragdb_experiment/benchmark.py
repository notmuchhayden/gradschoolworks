from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .config import Settings
from .io import read_jsonl
from .search import exact_search, parse_filter, search_pgvector, search_qdrant


def recall_at_k(expected: list[str], actual: list[str], k: int) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    return len(set(expected[:k]) & set(actual[:k])) / min(k, len(expected))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), pct))


def run_benchmark(
    settings: Settings,
    embedded_documents_path: Path,
    embedded_queries_path: Path,
    output_path: Path,
    k_values: list[int],
    repeats: int,
    filtered: bool,
    warmup: int,
) -> None:
    docs = list(read_jsonl(embedded_documents_path))
    queries = list(read_jsonl(embedded_queries_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for engine in ["qdrant", "pgvector"]:
        for k in k_values:
            latencies_ms: list[float] = []
            recalls: list[float] = []
            result_shortages = 0
            total_searches = 0
            started = time.perf_counter()

            for query in queries:
                filters = parse_filter(query.get("filters")) if filtered else None
                expected = exact_search(docs, query["embedding"], k, filters)

                for _ in range(warmup):
                    _run_one(settings, engine, query["embedding"], k, filters)

                for _ in range(repeats):
                    before = time.perf_counter()
                    actual = _run_one(settings, engine, query["embedding"], k, filters)
                    elapsed_ms = (time.perf_counter() - before) * 1000.0
                    latencies_ms.append(elapsed_ms)
                    recalls.append(recall_at_k(expected, actual, k))
                    if len(actual) < k:
                        result_shortages += 1
                    total_searches += 1

            wall_seconds = time.perf_counter() - started
            rows.append(
                {
                    "engine": engine,
                    "filtered": filtered,
                    "queries": len(queries),
                    "k": k,
                    "repeats": repeats,
                    "avg_latency_ms": sum(latencies_ms) / len(latencies_ms),
                    "p95_latency_ms": percentile(latencies_ms, 95),
                    "throughput_qps": total_searches / wall_seconds,
                    "avg_recall_at_k": sum(recalls) / len(recalls),
                    "result_shortage_rate": result_shortages / total_searches,
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_one(settings: Settings, engine: str, query_vector: list[float], k: int, filters):
    if engine == "qdrant":
        return search_qdrant(settings, query_vector, k, filters)
    if engine == "pgvector":
        return search_pgvector(settings, query_vector, k, filters)
    raise ValueError(f"unknown engine: {engine}")
