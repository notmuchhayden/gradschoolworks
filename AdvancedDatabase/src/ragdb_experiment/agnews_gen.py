# AG News 데이터셋을 사용하여 테스트 데이터 문서 생성

from datasets import load_dataset
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


out_path = Path("data/documents.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True) # 디렉토리가 없으면 생성

dataset = load_dataset("fancyzhx/ag_news", split="train") # Hugging Face에서 AG News 데이터셋 로드

# 10000 건만 사용
dataset = dataset.shuffle(seed=42).select(range(10000))

base_time = datetime(2026, 1, 1, tzinfo=timezone.utc) # 기준 시간 설정

with out_path.open("w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        label = int(row["label"]) # 레이블을 정수로 변환
        category = LABELS[label] # 레이블에 해당하는 카테고리 이름

        doc = {
            "doc_id": f"agnews-{i:06d}", # 고유한 문서 ID 생성
            "title": f"AG News {i:06d}", # 제목 생성
            "content": row["text"], # 내용 생성
            "category": category, # 카테고리 추가
            "year": 2026, # 연도 추가
            "doc_type": "article", # 문서 유형 추가
            "updated_at": (base_time + timedelta(minutes=i)).isoformat(), # 생성 시간 추가 (기준 시간에서 i분씩 증가)
            "embedding_version": 1,
        }
        f.write(json.dumps(doc, ensure_ascii=False) + "\n") # JSONL 형식으로 파일에 쓰기