from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark
from .batch_experiment import run_batch_experiment
from .config import get_settings
from .data_gen import generate_dataset
from .db import load_all, recreate_qdrant_collection, reset_postgres_tables, wait_for_services
from .embeddings import embed_jsonl
from .search import parse_filter, search_pgvector, search_qdrant
from .sync_experiment import run_sync_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragdb-exp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("generate")
    p.add_argument("--documents", type=int, default=1000)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--documents-out", type=Path, default=Path("data/documents.jsonl"))
    p.add_argument("--queries-out", type=Path, default=Path("data/queries.jsonl"))

    p = subparsers.add_parser("embed-docs")
    p.add_argument("--input", type=Path, default=Path("data/documents.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/documents.embedded.jsonl"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)

    p = subparsers.add_parser("embed-queries")
    p.add_argument("--input", type=Path, default=Path("data/queries.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/queries.embedded.jsonl"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)

    p = subparsers.add_parser("wait")
    p.add_argument("--timeout", type=int, default=60)

    subparsers.add_parser("reset")

    p = subparsers.add_parser("load")
    p.add_argument("--input", type=Path, default=Path("data/documents.embedded.jsonl"))
    p.add_argument("--no-reset", action="store_true")

    p = subparsers.add_parser("search")
    p.add_argument("--engine", choices=["qdrant", "pgvector"], required=True)
    p.add_argument("--queries", type=Path, default=Path("data/queries.embedded.jsonl"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--filtered", action="store_true")

    p = subparsers.add_parser("benchmark-basic")
    p.add_argument("--documents", type=Path, default=Path("data/documents.embedded.jsonl"))
    p.add_argument("--queries", type=Path, default=Path("data/queries.embedded.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/basic.csv"))
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)

    p = subparsers.add_parser("benchmark-filter")
    p.add_argument("--documents", type=Path, default=Path("data/documents.embedded.jsonl"))
    p.add_argument("--queries", type=Path, default=Path("data/queries.embedded.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/filter.csv"))
    p.add_argument("--k", type=int, nargs="+", default=[10])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)

    p = subparsers.add_parser("sync-delay")
    p.add_argument("--documents", type=Path, default=Path("data/documents.embedded.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/sync_delay.csv"))
    p.add_argument("--delay-seconds", type=float, default=5.0)
    p.add_argument("--modify-ratio", type=float, default=0.1)
    p.add_argument("--delay-probe-interval-seconds", type=float, default=5.0)
    p.add_argument("--delay-max-probes", type=int, default=12)
    p.add_argument("--delay-probe-docs", type=int, default=20)
    p.add_argument("--mock", action="store_true")

    p = subparsers.add_parser("sync-batch")
    p.add_argument("--engine", choices=["qdrant", "pgvector"], required=True)
    p.add_argument("--input", type=Path, default=Path("data/documents.embedded.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/sync_batch.csv"))
    p.add_argument("--documents", type=int)
    p.add_argument("--batch-size", type=int, choices=[1, 10, 100, 1000], required=True)
    p.add_argument("--repeats", type=int)
    p.add_argument("--warmup", type=int)

    args = parser.parse_args()
    settings = get_settings()

    if args.command == "generate":
        generate_dataset(
            documents_path=args.documents_out,
            queries_path=args.queries_out,
            documents=args.documents,
            queries=args.queries,
            seed=args.seed,
        )
    elif args.command == "embed-docs":
        embed_jsonl(
            input_path=args.input,
            output_path=args.output,
            text_field="content",
            model_name=settings.embedding_model,
            dim=settings.embedding_dim,
            mock=args.mock,
            batch_size=args.batch_size,
        )
    elif args.command == "embed-queries":
        embed_jsonl(
            input_path=args.input,
            output_path=args.output,
            text_field="text",
            model_name=settings.embedding_model,
            dim=settings.embedding_dim,
            mock=args.mock,
            batch_size=args.batch_size,
        )
    elif args.command == "wait":
        wait_for_services(settings, args.timeout)
    elif args.command == "reset":
        reset_postgres_tables(settings)
        recreate_qdrant_collection(settings)
    elif args.command == "load":
        load_all(settings, args.input, reset=not args.no_reset)
    elif args.command == "search":
        from .io import read_jsonl

        queries = list(read_jsonl(args.queries))
        query = queries[args.index]
        filters = parse_filter(query.get("filters")) if args.filtered else None
        if args.engine == "qdrant":
            hits = search_qdrant(settings, query["embedding"], args.k, filters)
        else:
            hits = search_pgvector(settings, query["embedding"], args.k, filters)
        for rank, doc_id in enumerate(hits, start=1):
            print(f"{rank}\t{doc_id}")
    elif args.command == "benchmark-basic":
        run_benchmark(
            settings=settings,
            embedded_documents_path=args.documents,
            embedded_queries_path=args.queries,
            output_path=args.output,
            k_values=args.k,
            repeats=args.repeats,
            filtered=False,
            warmup=args.warmup,
        )
    elif args.command == "benchmark-filter":
        run_benchmark(
            settings=settings,
            embedded_documents_path=args.documents,
            embedded_queries_path=args.queries,
            output_path=args.output,
            k_values=args.k,
            repeats=args.repeats,
            filtered=True,
            warmup=args.warmup,
        )
    elif args.command == "sync-delay":
        run_sync_experiment(
            settings=settings,
            embedded_documents_path=args.documents,
            output_path=args.output,
            delay_seconds=args.delay_seconds,
            modify_ratio=args.modify_ratio,
            model_name=settings.embedding_model,
            dim=settings.embedding_dim,
            mock=args.mock,
            delay_probe_interval_seconds=args.delay_probe_interval_seconds,
            delay_max_probes=args.delay_max_probes,
            delay_probe_docs=args.delay_probe_docs,
        )
    elif args.command == "sync-batch":
        run_batch_experiment(
            settings=settings,
            engine=args.engine,
            embedded_documents_path=args.input,
            output_path=args.output,
            documents=(
                settings.experiment_documents
                if args.documents is None
                else args.documents
            ),
            batch_size=args.batch_size,
            repeats=(
                settings.experiment_repeats
                if args.repeats is None
                else args.repeats
            ),
            warmup=(
                settings.experiment_warmup
                if args.warmup is None
                else args.warmup
            ),
            dim=settings.embedding_dim,
        )
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
