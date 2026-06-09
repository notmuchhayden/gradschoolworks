# RAG 저장 구조 비교 실험 환경

이 폴더는 데이터베이스특론 텀프로젝트의 실험 환경이다. 비교 대상은 다음 두 구조다.

- `PostgreSQL + Qdrant`: PostgreSQL에 원문/메타데이터, Qdrant에 벡터 저장
- `PostgreSQL + pgvector`: PostgreSQL 내부에 원문/메타데이터/벡터 통합 저장

## 1. 구성 요소

```text
docker-compose.yml
infra/
  postgres_meta/init.sql       # Qdrant 분리형의 원문/메타데이터 DB
  postgres_pgvector/init.sql   # pgvector 통합형 DB
src/ragdb_experiment/
  data_gen.py                  # synthetic 문서/질의 생성
  embeddings.py                # 결정적 mock embedding 생성
  db.py                        # PostgreSQL/Qdrant 적재
  search.py                    # Qdrant/pgvector 검색
  benchmark.py                 # 기본/필터 검색 측정
  sync_experiment.py           # stale vector 동기화 지연 실험
  batch_experiment.py          # 구조별 1/100/1000건 배치 갱신 실험
  main.py                      # CLI 진입점
```

## 2. Python 환경

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

본 실험은 검색 품질이 아니라 배치 갱신 성능과 일관성을 측정하므로 외부 데이터셋과 실제 임베딩 모델을 사용하지 않는다. 고정 seed synthetic 문서와 결정적 mock embedding을 사용한다.

## 3. 컨테이너 실행

```bash
cp .env.example .env
docker compose up -d
ragdb-exp wait --timeout 120
```

포트와 실험 기본값은 `.env`에서 변경할 수 있다. 기본 포트는 다음과 같다.

| 서비스 | 포트 | 용도 |
|---|---:|---|
| `postgres_meta` | `5433` | Qdrant 분리형의 원문/메타데이터 저장 |
| `postgres_pgvector` | `5434` | pgvector 통합형 저장 |
| `qdrant` | `6333` | Qdrant HTTP API |

## 4. 빠른 스모크 테스트

모델 다운로드 없이 전체 흐름만 확인하는 명령이다.

```bash
ragdb-exp generate --documents 1000 --queries 20
ragdb-exp embed-docs --mock
ragdb-exp embed-queries --mock
ragdb-exp load
ragdb-exp search --engine qdrant --index 0 --k 5
ragdb-exp search --engine pgvector --index 0 --k 5
ragdb-exp benchmark-basic --repeats 3 --k 10 --output results/basic_smoke.csv
ragdb-exp benchmark-filter --repeats 3 --k 10 --output results/filter_smoke.csv
ragdb-exp sync-batch --engine qdrant --documents 100 --batch-size 100 --repeats 1 --warmup 0 --output results/qdrant_batch_smoke.csv
ragdb-exp sync-batch --engine pgvector --documents 100 --batch-size 100 --repeats 1 --warmup 0 --output results/pgvector_batch_smoke.csv
```

## 5. 구조별 배치 갱신 실험

synthetic 문서 10,000건과 mock embedding을 준비하고 두 구조에 동일하게 적재한다.

```bash
ragdb-exp generate --documents 10000 --queries 100
ragdb-exp embed-docs --mock
ragdb-exp embed-queries --mock
ragdb-exp load
```

각 조건 전에 `ragdb-exp load`로 시작 상태를 복원한다.

```bash
ragdb-exp load
ragdb-exp sync-batch --engine qdrant --batch-size 1 --output results/qdrant_batch_1.csv
ragdb-exp load
ragdb-exp sync-batch --engine qdrant --batch-size 100 --output results/qdrant_batch_100.csv
ragdb-exp load
ragdb-exp sync-batch --engine qdrant --batch-size 1000 --output results/qdrant_batch_1000.csv

ragdb-exp load
ragdb-exp sync-batch --engine pgvector --batch-size 1 --output results/pgvector_batch_1.csv
ragdb-exp load
ragdb-exp sync-batch --engine pgvector --batch-size 100 --output results/pgvector_batch_100.csv
ragdb-exp load
ragdb-exp sync-batch --engine pgvector --batch-size 1000 --output results/pgvector_batch_1000.csv
```

기본값은 문서 1,000건, 조건별 30회, warmup 1회이다. `.env`의 `EXPERIMENT_DOCUMENTS`, `EXPERIMENT_REPEATS`, `EXPERIMENT_WARMUP`으로 변경할 수 있다. 수정 임베딩 생성 시간은 측정에서 제외된다.

## 6. 자원 제한 실험

기본 측정 후 준비된 Compose override를 적용해 같은 여섯 조건을 반복 측정한다.

| 조건 | CPU | Memory |
|---|---:|---:|
| R1 | 제한 없음 | 제한 없음 |
| R2 | 2 CPU | 4 GB |
| R3 | 1 CPU | 2 GB |

측정 중 컨테이너 자원 사용량은 별도 터미널에서 확인한다.

```bash
docker stats rag_postgres_meta rag_postgres_pgvector rag_qdrant
```

## 7. 결과 파일

벤치마크 결과는 `results/*.csv`에 저장된다. 중간보고서의 결과표에는 다음 열을 옮기면 된다.

- 기본 검색: `engine`, `k`, `avg_latency_ms`, `p95_latency_ms`, `throughput_qps`, `avg_recall_at_k`
- 필터 검색: 위 항목 + `result_shortage_rate`
- 배치 갱신: `engine`, `batch_size`, `repeat`, `documents`, `update_operations`, `total_processing_time_ms`, `document_visibility_latency_p95_ms`, `consistency_error_ratio_after_update`
