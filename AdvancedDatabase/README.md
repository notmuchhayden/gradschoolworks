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
  embeddings.py                # sentence-transformers 또는 mock embedding
  db.py                        # PostgreSQL/Qdrant 적재
  search.py                    # Qdrant/pgvector 검색
  benchmark.py                 # 기본/필터 검색 측정
  sync_experiment.py           # stale vector 동기화 지연 실험
  main.py                      # CLI 진입점
```

## 2. Python 환경

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

실제 `all-MiniLM-L6-v2` 모델로 임베딩을 생성할 때만 아래 의존성을 추가로 설치한다.

```bash
pip install -r requirements-model.txt
```

폐쇄망에서 실제 모델을 사용할 경우 `sentence-transformers/all-MiniLM-L6-v2` 모델 캐시를 미리 준비해야 한다. 모델 준비 전에는 `--mock` 옵션으로 파이프라인 동작을 검증할 수 있다.

## 3. 컨테이너 실행

```bash
docker compose up -d
ragdb-exp wait --timeout 120
```

포트는 다음과 같다.

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
```

## 5. 실제 모델 기반 실험

```bash
ragdb-exp generate --documents 10000 --queries 100
ragdb-exp embed-docs --batch-size 64
ragdb-exp embed-queries --batch-size 64
ragdb-exp load
ragdb-exp benchmark-basic --repeats 20 --k 5 10 20 --output results/basic.csv
ragdb-exp benchmark-filter --repeats 20 --k 10 --output results/filter.csv
ragdb-exp sync-delay --delay-seconds 5 --modify-ratio 0.1 --output results/sync_delay_5s.csv
```

동기화 지연 조건별 측정은 다음처럼 반복한다.

```bash
ragdb-exp load
ragdb-exp sync-delay --delay-seconds 0 --output results/sync_delay_0s.csv
ragdb-exp load
ragdb-exp sync-delay --delay-seconds 30 --output results/sync_delay_30s.csv
ragdb-exp load
ragdb-exp sync-delay --delay-seconds 120 --output results/sync_delay_120s.csv
```

각 `sync-delay` 실행 전 `ragdb-exp load`를 다시 수행하면 수정 실험 시작 상태를 동일하게 맞출 수 있다.

## 6. 자원 제한 실험

기본 측정 후 `docker-compose.yml`의 각 서비스에 `deploy.resources.limits` 또는 Docker 실행 옵션을 적용해 다음 조건을 반복 측정한다.

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
- 동기화 지연: `engine`, `delay_seconds`, `stale_vector_ratio_before_sync`, `stale_retrieval_count_before_sync`, `update_latency_ms`
