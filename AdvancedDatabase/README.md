# RAG 저장 구조 비교 실험 환경

이 폴더는 데이터베이스특론 텀프로젝트의 실험 환경이다. 비교 대상은 다음 두 구조다.

- `PostgreSQL + Qdrant`: PostgreSQL에 원문/메타데이터, Qdrant에 벡터 저장
- `PostgreSQL + pgvector`: PostgreSQL 내부에 원문/메타데이터/벡터 통합 저장

## 1. 구성 요소

```text
docker-compose.yml
docker-compose.r2.yml             # R2: 서비스별 2 CPU, 4 GB 제한
docker-compose.r3.yml             # R3: 서비스별 1 CPU, 2 GB 제한
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
  batch_experiment.py          # 구조별 1/10/100/1000건 배치 갱신 실험
  main.py                      # CLI 진입점
```

## 2. Python 환경

### 사전 요구사항

- Python 3.10 이상
- Docker Engine 또는 Docker Desktop
- Docker Compose v2 (`docker compose` 명령)

모든 명령은 이 `README.md`가 있는 `AdvancedDatabase` 디렉터리에서 실행한다.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

본 실험은 검색 품질이 아니라 배치 갱신 성능과 일관성을 측정하므로 외부 데이터셋과 실제 임베딩 모델을 사용하지 않는다. 고정 seed synthetic 문서와 결정적 mock embedding을 사용한다.

## 3. 컨테이너 실행

Linux/macOS:

```bash
cp .env.example .env
docker compose up -d
ragdb-exp wait --timeout 120
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
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

최초 실행에는 `ragdb-exp load`가 필요하다. `sync-batch`는 각 warmup 및 측정 반복 전에 선택한 엔진의 대상 문서를 원래 상태로 자체 복원하므로, 각 배치 조건 사이에서 `ragdb-exp load`를 다시 실행할 필요는 없다. 다만 데이터나 설정을 변경했거나 전체 저장소를 초기 상태로 되돌리려면 다시 실행한다.

```bash
ragdb-exp sync-batch --engine qdrant --batch-size 1 --output results/qdrant_batch_1.csv
ragdb-exp sync-batch --engine qdrant --batch-size 10 --output results/qdrant_batch_10.csv
ragdb-exp sync-batch --engine qdrant --batch-size 100 --output results/qdrant_batch_100.csv
ragdb-exp sync-batch --engine qdrant --batch-size 1000 --output results/qdrant_batch_1000.csv

ragdb-exp sync-batch --engine pgvector --batch-size 1 --output results/pgvector_batch_1.csv
ragdb-exp sync-batch --engine pgvector --batch-size 10 --output results/pgvector_batch_10.csv
ragdb-exp sync-batch --engine pgvector --batch-size 100 --output results/pgvector_batch_100.csv
ragdb-exp sync-batch --engine pgvector --batch-size 1000 --output results/pgvector_batch_1000.csv
```

기본값은 문서 1,000건, 조건별 30회, warmup 1회이다. `.env`의 `EXPERIMENT_DOCUMENTS`, `EXPERIMENT_REPEATS`, `EXPERIMENT_WARMUP`으로 변경할 수 있다. 수정 임베딩 생성 시간은 측정에서 제외된다.

## 6. 자원 제한 실험

기본 측정 후 준비된 Compose override를 적용해 같은 여덟 조건을 반복 측정한다.

| 조건 | CPU | Memory |
|---|---:|---:|
| R1 | 제한 없음 | 제한 없음 |
| R2 | 2 CPU | 4 GB |
| R3 | 1 CPU | 2 GB |

각 조건은 다음 명령으로 적용한다. 조건을 바꿀 때 `--force-recreate`로 컨테이너를 다시 생성한 후 서비스 준비 상태를 확인한다. 명명된 Docker volume은 유지되므로 기존 적재 데이터는 보존된다.

R1:

```bash
docker compose up -d --force-recreate
ragdb-exp wait --timeout 120
```

R2:

```bash
docker compose -f docker-compose.yml -f docker-compose.r2.yml up -d --force-recreate
ragdb-exp wait --timeout 120
```

R3:

```bash
docker compose -f docker-compose.yml -f docker-compose.r3.yml up -d --force-recreate
ragdb-exp wait --timeout 120
```

각 자원 조건에서 5절의 여덟 명령을 다시 실행한다. 이전 결과를 덮어쓰지 않도록 `results/r1_*`, `results/r2_*`, `results/r3_*`처럼 자원 조건을 출력 파일명에 포함한다.

측정 중 컨테이너 자원 사용량은 별도 터미널에서 확인한다.

```bash
docker stats rag_postgres_meta rag_postgres_pgvector rag_qdrant
```

## 7. 결과 파일

벤치마크 결과는 `results/*.csv`에 저장된다. 중간보고서의 결과표에는 다음 열을 옮기면 된다.

- 기본 검색: `engine`, `k`, `avg_latency_ms`, `p95_latency_ms`, `throughput_qps`, `avg_recall_at_k`
- 필터 검색: 위 항목 + `result_shortage_rate`
- 배치 갱신: `engine`, `batch_size`, `repeat`, `documents`, `update_operations`, `total_processing_time_ms`, `document_visibility_latency_p95_ms`, `consistency_error_ratio_after_update`

`sync-batch`는 CSV와 함께 같은 이름의 `*.modified.jsonl` 파일을 생성한다. 이 JSONL 파일은 실험 중 사용한 수정 문서이며, 실험 재실행에는 필요하지 않다.

현재 프로젝트의 `batch_experiment_summary.csv`는 개별 배치 결과를 별도로 집계한 파일이다. 이 저장소에는 요약 파일 생성 명령이 포함되어 있지 않으므로, 원시 측정값 재현의 기준은 구조별 `*_batch_*.csv` 파일이다.

## 8. 제출용 압축 파일

실험 재연 코드와 최종 결과 근거를 함께 제출할 때는 다음 항목을 포함한다.

```text
README.md
.env.example
pyproject.toml
requirements.txt
docker-compose.yml
docker-compose.r2.yml
docker-compose.r3.yml
infra/
src/
term_project_final_report.md
results/
  batch_experiment_summary.csv
  stale_validation.csv
  qdrant_batch_1.csv
  qdrant_batch_10.csv
  qdrant_batch_100.csv
  qdrant_batch_1000.csv
  pgvector_batch_1.csv
  pgvector_batch_10.csv
  pgvector_batch_100.csv
  pgvector_batch_1000.csv
```

다음 항목은 생성 파일, 로컬 환경 또는 캐시이므로 압축에서 제외한다.

- `.venv/`, `__pycache__/`, `*.pyc`, `*.egg-info/`
- 비밀번호 등 로컬 설정이 들어갈 수 있는 `.env`
- 명령으로 다시 생성할 수 있는 `data/`
- `results/*_smoke.csv`, `results/*.modified.jsonl`
- Docker volume 데이터와 모델 캐시
- mock embedding 실험에는 사용하지 않는 `requirements-model.txt`

