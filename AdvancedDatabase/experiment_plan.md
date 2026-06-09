# 데이터베이스특론 텀프로젝트 실험 계획 문서

이 문서는 데이터베이스특론 텀프로젝트의 실험 계획과 실행 순서를 정리한 문서이다. 연구의 중심은 **벡터 저장 구조별로 어떤 갱신 배치 방법이 효율성과 최신성의 균형이 좋은가**를 확인하는 것이다.

이를 위해 1,000건의 문서를 수정하면서 `PostgreSQL + Qdrant` 분리형 구조와 `PostgreSQL + pgvector` 통합형 구조에 각각 1건, 10건, 100건, 1,000건 배치 갱신을 적용한다. 분리형 구조의 stale embedding 발생 여부는 동기화 필요성을 보여주는 사전 검증으로만 수행한다.

## 1. 프로젝트 주제

폐쇄 환경의 Qdrant 분리형과 pgvector 통합형 벡터 검색 구조에서 배치 갱신 방법 비교

## 2. 연구 배경 요약
의미 기반 검색은 문서와 질의를 임베딩 벡터로 변환한 뒤 벡터 유사도를 이용해 관련 문서를 찾는 방식이다. 이 방식은 키워드가 정확히 일치하지 않아도 의미가 비슷한 문서를 찾을 수 있다는 장점이 있다.

초기 계획은 Qdrant 분리형 구조에서 동기화를 늦출수록 stale embedding이 얼마나 오래 노출되는지 확인하는 것이었다. 그러나 의도적으로 5초, 30초, 120초 지연시키는 실험은 동기화하지 않으면 불일치가 유지된다는 사실만 반복해서 보여준다. 따라서 이 실험은 원문만 수정한 뒤 동기화 전 검색 결과를 한 번 확인하는 사전 검증으로 축소한다.

본 실험에서는 동기화 필요성이 확인되었다는 전제에서 **변경 건을 어느 크기로 묶어서 반영하는 것이 좋은가**를 비교한다. Qdrant 분리형은 PostgreSQL과 Qdrant를 별도로 갱신하고, pgvector 통합형은 본문과 벡터를 같은 트랜잭션으로 갱신한다. 작은 배치는 최신 변경을 빠르게 반영하지만 요청과 트랜잭션 횟수가 많고, 큰 배치는 처리 효율이 높을 수 있지만 변경이 배치에 쌓이는 동안 최신 상태 반영이 늦어진다.

따라서 두 구조에 동일한 1건, 10건, 100건, 1,000건 배치 정책을 적용하고 전체 처리시간과 문서별 반영 지연을 측정하여 구조와 배치 크기에 따른 차이를 확인한다.

## 3. 실험 대상 구조

### 구조 A: PostgreSQL + Qdrant 분리형 구조

- PostgreSQL: 문서 원문, 제목, 메타데이터, 수정 시각, 임베딩 버전 저장
- Qdrant: 임베딩 벡터와 검색에 필요한 payload 저장
- 특징: 원문 저장소와 벡터 저장소가 분리되어 있음
- 주요 관찰 지점: 동기화 배치 크기에 따른 전체 처리시간과 문서별 동기화 지연

PostgreSQL에서 원문과 `embedding_version`을 변경한 뒤, 변경 건을 설정된 배치 크기로 Qdrant에 반영한다. Q1, Q10, Q100, Q1000은 모두 같은 1,000건과 같은 수정 임베딩을 사용한다.

### 구조 B: PostgreSQL + pgvector 통합형 구조

- PostgreSQL: 문서 원문, 메타데이터, 임베딩 벡터를 같은 DBMS에 저장
- pgvector: PostgreSQL 내부에서 벡터 타입과 벡터 거리 검색 제공
- 특징: 원문과 벡터를 같은 저장소 안에서 관리함
- 주요 관찰 지점: 배치 크기에 따른 전체 처리시간과 문서별 반영 지연

pgvector에도 Qdrant와 동일하게 1건, 10건, 100건, 1,000건 조건을 적용한다. 다만 pgvector는 별도 저장소 간 동기화가 아니라 본문과 벡터를 같은 트랜잭션으로 갱신하므로, 비교 결과는 동기화 비용뿐 아니라 저장 구조 차이까지 포함한다.

## 4. Docker Compose 구성

두 구조를 같은 환경에서 비교하기 위해 전체 컨테이너 세 개를 모두 사용한다.

| 서비스 | 역할 | 포트 |
|---|---|---:|
| `postgres_meta` | Qdrant 분리형 구조의 원문 및 메타데이터 저장소 | 5433 |
| `postgres_pgvector` | pgvector 통합형 구조의 원문, 메타데이터, 벡터 저장소 | 5434 |
| `qdrant` | 전용 벡터DB 검색 저장소 | 6333, 6334 |

Docker Compose를 사용하는 이유는 세 컨테이너를 하나의 실험 환경으로 묶어서 관리하기 위해서이다. 이 실험은 PostgreSQL 하나만 실행하는 단순한 실험이 아니라, PostgreSQL 두 개와 Qdrant 하나를 함께 실행해야 한다. Compose 파일을 사용하면 이미지, 포트, 볼륨, 초기화 SQL, healthcheck를 고정할 수 있어 같은 환경을 반복해서 만들기 쉽다.

또한 `docker compose up -d`로 전체 실험 환경을 한 번에 실행하고, `docker compose ps`로 상태를 확인할 수 있다. 동기화 실험은 여러 번 반복해야 하므로, 환경을 쉽게 올리고 내릴 수 있는 구성이 중요하다.

## 5. 권장 운영체제

실험은 Ubuntu Linux 에서 수행한다. Docker, PostgreSQL, Qdrant, pgvector가 Linux 컨테이너 환경에서 자연스럽게 동작한다. 또한 OS와 파일시스템 차이가 실험 결과에 영향을 주지 않도록 환경을 고정할 수 있다. 실험 환경을 다른 장비로 옮길 때 Docker Compose 기반 구성이 재현하기 쉽기 때문이다.

## 6. 공통 실험 조건

| 항목 | 값 |
|---|---|
| 실행 환경 | 로컬 Linux 또는 WSL2 Ubuntu |
| 외부 API | 사용하지 않음 |
| 컨테이너 | Docker Compose |
| DBMS | PostgreSQL |
| 전용 벡터DB | Qdrant |
| RDBMS 통합 벡터 검색 | pgvector |
| 벡터 생성 방식 | 고정 seed 기반 384차원 mock embedding |
| 스모크 테스트 데이터 | 1,000건 |
| 본 실험 데이터 | 10,000건 |
| 수정 대상 | 10,000건 중 1,000건 |
| 비교 구조 | Qdrant 분리형, pgvector 통합형 |
| 비교 배치 크기 | 1건, 10건, 100건, 1,000건 |
| 배치 갱신 횟수 | 구조별 1,000회, 100회, 10회, 1회 |
| 변경 입력 방식 | 인위적 대기 없이 1,000건 연속 처리 |
| 검색 단위 | 문서 단위 |
| 비교 조건 수 | 2개 구조 × 4개 배치 크기 = 8개 |
| 반복 횟수 | 8개 조건별 30회 |

스모크 테스트와 본 실험 모두 synthetic 문서와 mock embedding을 사용한다. 이 실험은 검색 정확도가 아니라 갱신 방식의 처리시간과 일관성을 비교하므로 외부 데이터셋과 실제 임베딩 모델은 사용하지 않는다.

## 7. 데이터 설계

처음에는 문서 단위 검색으로 진행한다. 긴 문서 chunking은 현재 연구 범위에서는 제외한다.

공통 문서 필드:

| 필드 | 설명 |
|---|---|
| `doc_id` | 문서 고유 ID |
| `title` | 제목 |
| `content` | 본문 |
| `category` | 카테고리 |
| `year` | 작성 연도 |
| `doc_type` | 문서 유형 |
| `updated_at` | 수정 시각 |
| `embedding_version` | 임베딩 버전 |

`embedding_version`은 동기화 상태를 판별하는 필드이다. PostgreSQL의 최신 문서 버전과 Qdrant payload의 임베딩 버전이 다르면 아직 동기화되지 않은 stale vector로 판단한다.

예시 category:

- news
- tech_doc
- policy
- manual

예시 doc_type:

- article
- report
- manual
- notice

## 8. 본 실험: 구조별 배치 갱신 방법 비교

### 목적

1,000건의 문서 변경을 Qdrant 분리형과 pgvector 통합형 구조에 반영할 때 배치 크기에 따라 처리 효율과 최신성이 어떻게 달라지는지 비교한다. 이 실험의 목적은 분리형 벡터DB에 동기화 문제가 있다는 사실을 반복해서 증명하는 것이 아니라, 각 저장 구조에서 실제 운영에 사용할 수 있는 갱신 방법을 선택하고 두 구조의 차이를 확인하는 것이다.

### 사전 검증: 동기화 필요성 확인

stale embedding 발생 여부는 다음 절차로 한 번만 확인한다.

1. PostgreSQL의 문서 원문과 `embedding_version`을 변경한다.
2. Qdrant는 갱신하지 않은 상태로 둔다.
3. 수정 전 의미와 수정 후 의미에 대한 검색을 각각 수행한다.
4. PostgreSQL과 Qdrant의 `embedding_version`이 다른지 확인한다.
5. Qdrant를 동기화한 뒤 최신 의미가 검색에 반영되는지 확인한다.

이 검증에서는 인위적으로 5초, 30초, 120초를 기다리지 않는다. 동기화를 수행하지 않은 상태에서 원문과 벡터가 불일치하고 검색이 예전 의미를 사용할 수 있다는 사실만 확인한다.

### 수정 데이터 준비

- 전체 10,000건 중 동일한 1,000건을 수정한다.
- 수정 전후 의미가 달라지도록 본문을 변경한다.
- 여덟 조건에서 같은 문서 순서와 같은 수정 임베딩을 사용한다.
- 수정된 1,000건의 벡터는 고정된 mock embedding 함수로 실험 전에 미리 생성한다.

### 비교 조건

| 조건 | 구조 | 배치 크기 | 갱신 횟수 | 최대 미반영 문서 수 |
|---|---|---:|---:|---:|
| Q1 | Qdrant 분리형 | 1건 | 1,000회 | 1건 |
| Q10 | Qdrant 분리형 | 10건 | 100회 | 10건 |
| Q100 | Qdrant 분리형 | 100건 | 10회 | 100건 |
| Q1000 | Qdrant 분리형 | 1,000건 | 1회 | 1,000건 |
| P1 | pgvector 통합형 | 1건 | 1,000회 | 1건 |
| P10 | pgvector 통합형 | 10건 | 100회 | 10건 |
| P100 | pgvector 통합형 | 100건 | 10회 | 100건 |
| P1000 | pgvector 통합형 | 1,000건 | 1회 | 1,000건 |

### 공통 처리 흐름

1. `ragdb-exp load`로 두 구조를 동일한 초기 상태로 복원한다.
2. 미리 생성한 수정 문서와 임베딩 1,000건을 같은 순서로 준비한다.
3. 각 문서의 변경 이벤트 발생 시각을 기록하고 조건별 배치에 추가한다.
4. 설정된 배치 크기에 도달하면 구조별 갱신 작업을 실행한다.
5. 각 문서의 최신 본문과 벡터가 검색 가능한 시각을 기록한다.
6. 1,000건의 변경과 마지막 배치 갱신이 모두 끝나면 전체 처리시간을 기록한다.
7. 모든 수정 문서의 최신 본문과 벡터 버전이 일치하는지 확인한다.
8. 여덟 조건을 각각 30회 반복한다.

### 구조별 갱신 방식

Qdrant 분리형:

1. PostgreSQL의 원문과 `embedding_version`을 변경한다.
2. 배치 크기에 도달하면 해당 문서의 vector와 payload를 Qdrant에 upsert한다.
3. Qdrant 반영이 완료된 시각을 각 문서의 최신 상태 반영 시각으로 기록한다.

pgvector 통합형:

1. 변경 이벤트를 설정된 배치 크기까지 모은다.
2. 한 PostgreSQL 트랜잭션에서 해당 문서의 본문, vector, `embedding_version`을 함께 갱신한다.
3. 트랜잭션 commit 완료 시각을 각 문서의 최신 상태 반영 시각으로 기록한다.

### 측정 지표

| 지표 | 의미 |
|---|---|
| `total_processing_time_ms` | 첫 문서 변경 시작부터 1,000건의 마지막 동기화 완료까지 걸린 시간 |
| `document_visibility_latency_p95_ms` | 변경 이벤트 발생 후 최신 본문과 벡터가 검색 가능한 상태가 되기까지 걸린 시간의 p95 |
| `consistency_error_ratio_after_update` | 실험 완료 후 최신 본문과 벡터 버전이 일치하지 않는 문서 비율 |

```text
document visibility latency = 최신 상태 반영 시각 - 변경 이벤트 발생 시각
```

`total_processing_time_ms`는 갱신 작업을 묶었을 때 얻는 처리 효율을 보여준다. `document_visibility_latency_p95_ms`는 변경된 문서가 배치 완료를 기다리면서 최신 검색 상태로 반영되기까지 걸린 시간을 보여준다. `consistency_error_ratio_after_update`는 성능 비교 지표가 아니라 여덟 조건이 모두 정상적으로 갱신을 완료했는지 확인하는 검증 지표이며 0이어야 한다.

### 해석 기준

- 1건 조건은 최신성이 가장 높을 것으로 예상되지만 Qdrant 요청 또는 PostgreSQL 트랜잭션 1,000회의 비용이 발생한다.
- 10건 조건은 최신성을 비교적 높게 유지하면서 갱신 횟수를 100회로 줄이는 소규모 배치 조건이다.
- 100건 조건은 갱신 횟수를 크게 줄이면서 미반영 문서 수와 문서별 지연을 제한하는 절충안이다.
- 1,000건 조건은 전체 처리시간이 짧을 수 있지만 먼저 발생한 변경이 마지막 배치까지 오래 기다린다.
- 같은 배치 크기에서 Qdrant와 pgvector를 비교하면 저장소 분리와 단일 트랜잭션 구조의 비용 차이를 확인할 수 있다.
- 가장 좋은 방법은 전체 처리시간만 가장 짧은 조건이 아니라, 요구되는 최신성 범위 안에서 처리시간이 가장 짧은 조건이다.
- 본 결과는 1,000건이 연속으로 발생하는 조건에 대한 결과이다. 변경 빈도가 낮은 실제 운영에서는 건수 기준에 도달하지 못해 동기화가 오래 지연될 수 있으므로 최대 대기시간 조건을 함께 사용하는 방식을 추가로 고려해야 한다.

## 9. 보조 실험: 기본 검색 및 필터 검색 확인

기본 검색과 필터 검색은 본 연구의 중심 실험이 아니다. Qdrant와 pgvector가 정상적으로 적재와 검색을 수행하는지 확인하기 위한 보조 실험으로만 유지한다.

### 기본 검색 확인

- 같은 질의를 Qdrant와 pgvector에 입력한다.
- top-k 결과가 반환되는지 확인한다.
- 평균 검색 시간은 참고용으로만 기록한다.

### 필터 검색 확인

- `category`, `year`, `doc_type` 조건을 함께 적용한다.
- Qdrant는 payload filter를 사용한다.
- pgvector는 SQL `WHERE` 조건과 벡터 정렬을 함께 사용한다.
- 결과 부족률과 검색 가능 여부를 확인한다.

이 보조 실험 결과는 최종 보고서에서 검색 성능 우열을 주장하는 근거로 사용하지 않는다. 본 실험 전에 Qdrant와 pgvector의 적재 및 검색 기능이 정상적으로 동작하는지 확인하는 용도로만 사용한다.

## 10. 확장 실험: 자원 제한 환경

자원 제한 실험은 필수 실험이 아니다. 시간이 남으면 같은 여덟 조건을 제한된 CPU와 메모리에서 반복하여 구조별 권장 배치 크기가 자원 조건에 따라 달라지는지 확인한다.

### Docker 자원 조건

| 조건 | CPU | Memory |
|---|---:|---:|
| R1 | 제한 없음 | 제한 없음 |
| R2 | 2 CPU | 4 GB |
| R3 | 1 CPU | 2 GB |

이미 다음 override 파일을 준비했다.

- `docker-compose.r2.yml`
- `docker-compose.r3.yml`

실행 예시:

```bash
docker compose -f docker-compose.yml -f docker-compose.r2.yml up -d
docker compose -f docker-compose.yml -f docker-compose.r3.yml up -d
```

## 11. 추천 실행 순서

### 1단계: 환경 확인

1. Docker 설치 확인
2. Docker Compose 설치 확인
3. `docker compose up -d` 실행
4. `docker compose ps`로 세 컨테이너 상태 확인
5. Python 가상환경 활성화


### 2단계: 스모크 테스트

스모크 테스트는 이미 1,000건 mock embedding으로 수행했다. 다시 수행하려면 다음 명령을 사용한다.

```bash
ragdb-exp generate --documents 1000 --queries 20
ragdb-exp embed-docs --mock
ragdb-exp embed-queries --mock
ragdb-exp load
ragdb-exp search --engine qdrant --index 0 --k 5
ragdb-exp search --engine pgvector --index 0 --k 5
```

### 3단계: synthetic 데이터 준비

1. 고정 seed로 synthetic 문서 10,000건을 생성한다.
2. 필요한 경우 기능 확인용 질의 100개를 생성한다.
3. 문서와 질의에 mock embedding을 생성한다.
4. Qdrant 분리형과 pgvector 통합형 구조에 동일한 데이터를 적재한다.

명령 예시:

```bash
source .venv/bin/activate
ragdb-exp generate --documents 10000 --queries 100
ragdb-exp embed-docs --mock
ragdb-exp embed-queries --mock
ragdb-exp load
```

### 4단계: 동기화 필요성 사전 검증

원문만 변경하고 Qdrant를 갱신하지 않은 상태에서 stale vector와 예전 의미 검색 결과가 나타나는지 한 번 확인한다. 이 단계에서는 지연 시간을 변경하며 반복하지 않는다.

```bash
ragdb-exp load
ragdb-exp sync-delay --delay-seconds 0 --modify-ratio 0.1 --output results/stale_validation.csv
```

### 5단계: 구조별 배치 갱신 본 실험

각 조건을 실행하기 전에 `ragdb-exp load`로 시작 상태를 동일하게 복원한다. `sync-batch` 명령은 구조와 배치 크기를 지정하여 실험을 수행하고 반복별 결과를 CSV로 저장한다.

```bash
ragdb-exp load
ragdb-exp sync-batch --engine qdrant --documents 1000 --batch-size 1 --repeats 30 --output results/qdrant_batch_1.csv

ragdb-exp load
ragdb-exp sync-batch --engine qdrant --documents 1000 --batch-size 10 --repeats 30 --output results/qdrant_batch_10.csv

ragdb-exp load
ragdb-exp sync-batch --engine qdrant --documents 1000 --batch-size 100 --repeats 30 --output results/qdrant_batch_100.csv

ragdb-exp load
ragdb-exp sync-batch --engine qdrant --documents 1000 --batch-size 1000 --repeats 30 --output results/qdrant_batch_1000.csv

ragdb-exp load
ragdb-exp sync-batch --engine pgvector --documents 1000 --batch-size 1 --repeats 30 --output results/pgvector_batch_1.csv

ragdb-exp load
ragdb-exp sync-batch --engine pgvector --documents 1000 --batch-size 10 --repeats 30 --output results/pgvector_batch_10.csv

ragdb-exp load
ragdb-exp sync-batch --engine pgvector --documents 1000 --batch-size 100 --repeats 30 --output results/pgvector_batch_100.csv

ragdb-exp load
ragdb-exp sync-batch --engine pgvector --documents 1000 --batch-size 1000 --repeats 30 --output results/pgvector_batch_1000.csv
```

### 6단계: 결과 정리

1. 여덟 조건의 CSV 결과를 하나의 표로 정리한다.
2. 구조별·배치 크기별 `total_processing_time_ms`의 평균과 p95를 비교한다.
3. 구조별·배치 크기별 `document_visibility_latency_p95_ms`를 비교한다.
4. 모든 조건의 `consistency_error_ratio_after_update`가 0인지 확인한다.
5. 같은 배치 크기에서 Qdrant와 pgvector의 차이를 비교한다.
6. 처리 효율과 최신성 요구사항을 함께 고려하여 구조별 권장 배치 크기를 결정한다.

## 12. 결과표 템플릿

### 동기화 필요성 사전 검증

| 상태 | stale ratio | old meaning 검색 | new meaning 검색 |
|---|---:|---:|---:|
| Qdrant 동기화 전 |  |  |  |
| Qdrant 동기화 후 |  |  |  |

### 구조별 배치 갱신 방법 비교

| 조건 | 구조 | 배치 크기 | 갱신 횟수 | 전체 처리시간 평균 ms | 전체 처리시간 p95 ms | 문서 반영 지연 p95 ms | 최종 불일치 비율 |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1 | Qdrant | 1 | 1,000 |  |  |  |  |
| Q10 | Qdrant | 10 | 100 |  |  |  |  |
| Q100 | Qdrant | 100 | 10 |  |  |  |  |
| Q1000 | Qdrant | 1,000 | 1 |  |  |  |  |
| P1 | pgvector | 1 | 1,000 |  |  |  |  |
| P10 | pgvector | 10 | 100 |  |  |  |  |
| P100 | pgvector | 100 | 10 |  |  |  |  |
| P1000 | pgvector | 1,000 | 1 |  |  |  |  |

### 보조 검색 확인

| 구조 | 실험 | 데이터 수 | k | Avg latency | Recall@K | 비고 |
|---|---|---:|---:|---:|---:|---|
| Qdrant 분리형 | 기본 검색 |  |  |  |  | 보조 확인 |
| pgvector 통합형 | 기본 검색 |  |  |  |  | 보조 확인 |
| Qdrant 분리형 | 필터 검색 |  |  |  |  | 보조 확인 |
| pgvector 통합형 | 필터 검색 |  |  |  |  | 보조 확인 |

### 구조와 갱신 방식 비교

| 항목 | Qdrant 분리형 | pgvector 통합형 |
|---|---|---|
| 본문과 벡터 저장 위치 | 분리 | 동일 PostgreSQL |
| 갱신 방식 | PostgreSQL 변경 후 Qdrant upsert | 본문과 벡터를 한 트랜잭션으로 update |
| 중간 불일치 가능성 | 있음 | commit 전 외부 노출 없음 |
| 실패 시 재시도 대상 | Qdrant 미반영 배치 | PostgreSQL 트랜잭션 |
| 비교 배치 크기 | 1, 10, 100, 1,000 | 1, 10, 100, 1,000 |

## 13. 최종 결론에서 답해야 할 질문

최종 보고서에서는 아래 질문에 답하면 된다.

1. Qdrant와 pgvector에서 배치 크기가 전체 처리시간과 문서별 반영 지연에 어떤 영향을 주는가?
2. 같은 배치 크기에서 Qdrant 분리형과 pgvector 통합형의 처리 비용은 얼마나 다른가?
3. 10건과 100건 배치는 각 구조에서 최신성과 처리 효율 사이에 어떤 절충점을 제공하는가?
4. 구조별로 효율성과 최신성의 균형이 가장 좋은 배치 크기는 무엇인가?
5. Qdrant의 저장소 간 불일치 가능성은 실제 측정 결과와 운영 복잡성에 어떤 영향을 주는가?
6. 변경 빈도가 낮을 때 건수 기준과 최대 대기시간을 함께 적용해야 하는가?

## 14. 현재 폴더에 있는 주요 파일

| 파일 | 용도 |
|---|---|
| `term_project_final_report.md` | 수정된 최종보고서 본문 |
| `experiment_report.md` | 현재까지의 실험 진행 보고서 |
| `experiment_plan.md` | 실험 계획 및 실행 순서 |
| `docker-compose.yml` | 기본 Docker Compose 구성 |
| `docker-compose.r2.yml` | 2 CPU / 4 GB 확장 실험 조건 |
| `docker-compose.r3.yml` | 1 CPU / 2 GB 확장 실험 조건 |
| `requirements.txt` | 기본 Python 의존성 |
| `src/ragdb_experiment/` | 실험 실행 코드 |
| `infra/` | PostgreSQL 초기화 SQL |

## 15. 주의사항

- 현재 연구 중심은 Qdrant와 pgvector에서 적절한 배치 갱신 방법을 비교하는 것이다.
- 기본 검색과 필터 검색 결과는 보조 확인으로만 사용한다.
- 각 배치 조건 실험 전에는 `ragdb-exp load`로 시작 상태를 동일하게 맞춘다.
- 여덟 조건에서 수정 문서, 처리 순서, 사전 생성 임베딩을 반드시 동일하게 유지한다.
- mock embedding 생성 시간은 배치 갱신 성능 측정에서 제외한다.
- synthetic 데이터 생성 seed와 벡터 차원은 모든 조건에서 동일하게 유지한다.
- 스모크 테스트 결과를 최종 성능 결론으로 사용하지 않는다.
- 최종 보고서에서는 전체 처리시간과 문서별 반영 지연을 함께 고려하여 구조별 갱신 방법을 평가한다.
