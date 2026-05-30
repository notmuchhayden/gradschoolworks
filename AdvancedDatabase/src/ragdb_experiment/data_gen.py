from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .io import write_jsonl

CATEGORIES = ["news", "tech_doc", "policy", "manual"]
DOC_TYPES = ["article", "report", "manual", "notice"]

TOPICS = [
    ("vpn", "사내 VPN 접속 오류 처리 절차와 인증 로그 확인 방법"),
    ("refund", "환불 규정 변경과 고객 요청 처리 기준"),
    ("network", "네트워크 인증 오류 원인 분석 및 장애 복구 절차"),
    ("security", "제로트러스트 보안 정책과 접근 제어 점검 항목"),
    ("backup", "데이터베이스 백업 주기와 장애 복구 테스트 계획"),
    ("hr", "인사 발령 공지와 내부 승인 절차"),
    ("cloud", "폐쇄망 환경의 컨테이너 배포와 이미지 반입 절차"),
    ("manual", "장비 점검 매뉴얼과 현장 작업 안전 수칙"),
]


def make_documents(count: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    docs: list[dict] = []

    for i in range(count):
        topic_key, topic_text = rng.choice(TOPICS)
        category = rng.choice(CATEGORIES)
        doc_type = rng.choice(DOC_TYPES)
        year = rng.choice([2021, 2022, 2023, 2024, 2025, 2026])
        updated_at = base_time + timedelta(minutes=i)
        title = f"{topic_key.upper()} 문서 {i:06d}"
        content = (
            f"{topic_text}. 이 문서는 {category} 카테고리의 {doc_type} 유형 문서이며 "
            f"{year}년에 작성되었다. 담당자는 원인, 영향 범위, 조치 절차, "
            f"검증 방법을 순서대로 기록해야 한다. 참조 번호는 {i:06d}이다."
        )
        docs.append(
            {
                "doc_id": f"doc-{i:06d}",
                "title": title,
                "content": content,
                "category": category,
                "year": year,
                "doc_type": doc_type,
                "updated_at": updated_at.isoformat(),
                "embedding_version": 1,
            }
        )

    return docs


def make_queries(count: int, seed: int = 99) -> list[dict]:
    rng = random.Random(seed)
    templates = [
        "장애 복구 절차",
        "환불 규정",
        "네트워크 인증 오류",
        "VPN 접속 문제",
        "백업 복구 테스트",
        "제로트러스트 인증 실패",
        "컨테이너 배포 절차",
        "현장 장비 점검",
    ]
    queries: list[dict] = []
    for i in range(count):
        text = rng.choice(templates)
        category = rng.choice(CATEGORIES)
        doc_type = rng.choice(DOC_TYPES)
        year = rng.choice([2023, 2024, 2025, 2026])
        queries.append(
            {
                "query_id": f"q-{i:04d}",
                "text": text,
                "filters": {
                    "category": category,
                    "doc_type": doc_type,
                    "year_gte": year,
                },
            }
        )
    return queries


def generate_dataset(
    documents_path: Path,
    queries_path: Path,
    documents: int,
    queries: int,
    seed: int,
) -> None:
    write_jsonl(documents_path, make_documents(documents, seed))
    write_jsonl(queries_path, make_queries(queries, seed + 1))
