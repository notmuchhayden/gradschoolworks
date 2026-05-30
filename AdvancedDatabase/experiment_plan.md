# 데이터베이스특론 텀프로젝트 실험 계획 문서

이 문서는 데이터베이스특론 텀프로젝트의 실험 계획과 실행 순서를 정리한 문서이다. 중간보고서 수정 이후 실험 범위는 기존의 넓은 검색 성능 비교에서 **문서 수정 후 원문 데이터와 임베딩 벡터의 동기화 불일치 문제** 중심으로 축소되었다.

따라서 본 문서는 `PostgreSQL + Qdrant` 분리형 구조와 `PostgreSQL + pgvector` 통합형 구조를 비교하되, 검색 속도 자체보다 문서 수정 후 stale embedding이 어떻게 발생하고 검색 결과에 어떤 영향을 주는지 확인하는 데 초점을 둔다.

## 1. 프로젝트 주제

폐쇄 환경에서 의미 기반 문서 검색 시스템을 구성할 때, 전용 벡터DB 분리형 구조와 RDBMS 통합형 구조의 데이터 동기화 관리 방식 비교

## 2. 연구 배경 요약

의미 기반 검색은 문서와 질의를 임베딩 벡터로 변환한 뒤 벡터 유사도를 이용해 관련 문서를 찾는 방식이다. 이 방식은 키워드가 정확히 일치하지 않아도 의미가 비슷한 문서를 찾을 수 있다는 장점이 있다.

초기 계획은 키워드 검색과 벡터 검색의 차이, Qdrant와 pgvector의 검색 성능, 필터 검색, 자원 제한 실험까지 넓게 비교하는 것이었다. 그러나 중간보고서 수정 이후 실험 범위를 줄였다. 현재 연구의 중심은 **문서가 수정되었을 때 원문과 임베딩 벡터가 서로 맞지 않는 상황**을 확인하는 것이다.

특히 `PostgreSQL + Qdrant` 분리형 구조에서는 원문은 PostgreSQL에 있고 벡터는 Qdrant에 저장된다. 이때 원문 수정 후 Qdrant 벡터 갱신이 늦어지면 최신 문서 내용과 검색용 벡터가 서로 다른 의미를 가질 수 있다. 이를 stale embedding 또는 의미적 불일치 문제로 본다.

반대로 `PostgreSQL + pgvector` 통합형 구조에서는 원문과 벡터를 같은 PostgreSQL 안에 저장하므로, 새 임베딩이 준비된 뒤 본문과 벡터를 같은 update 흐름으로 반영할 수 있다. 본 연구는 이 두 구조의 동기화 관리 차이를 비교한다.

## 3. 비교 대상 구조

### 구조 A: PostgreSQL + Qdrant 분리형 구조

- PostgreSQL: 문서 원문, 제목, 메타데이터, 수정 시각, 임베딩 버전 저장
- Qdrant: 임베딩 벡터와 검색에 필요한 payload 저장
- 특징: 원문 저장소와 벡터 저장소가 분리되어 있음
- 주요 관찰 지점: 동기화 지연, stale embedding, old meaning retrieval, new meaning retrieval

이 구조는 문서 수정 후 불일치 상황을 재현하기 위해 필요하다. PostgreSQL의 원문을 먼저 수정하고, Qdrant의 벡터 갱신을 일부러 지연시키면 원문과 벡터가 어긋난 상태를 만들 수 있다.

### 구조 B: PostgreSQL + pgvector 통합형 구조

- PostgreSQL: 문서 원문, 메타데이터, 임베딩 벡터를 같은 DBMS에 저장
- pgvector: PostgreSQL 내부에서 벡터 타입과 벡터 거리 검색 제공
- 특징: 원문과 벡터를 같은 저장소 안에서 관리함
- 주요 관찰 지점: 본문과 벡터를 같은 update 흐름으로 반영할 수 있는지

이 구조는 Qdrant 분리형 구조의 비교 기준이다. 임베딩 계산은 DB 밖에서 수행되지만, 새 임베딩이 준비된 뒤에는 본문과 벡터를 같은 PostgreSQL 갱신 작업으로 묶을 수 있다.

## 4. Docker Compose 구성

비교 구조는 두 개이지만 실제 컨테이너는 총 세 개이다.

| 서비스 | 역할 | 포트 |
|---|---|---:|
| `postgres_meta` | Qdrant 분리형 구조의 원문 및 메타데이터 저장소 | 5433 |
| `postgres_pgvector` | pgvector 통합형 구조의 원문, 메타데이터, 벡터 저장소 | 5434 |
| `qdrant` | 전용 벡터DB 검색 저장소 | 6333, 6334 |

Docker Compose를 사용하는 이유는 세 컨테이너를 하나의 실험 환경으로 묶어서 관리하기 위해서이다. 이 실험은 PostgreSQL 하나만 실행하는 단순한 실험이 아니라, PostgreSQL 두 개와 Qdrant 하나를 함께 실행해야 한다. Compose 파일을 사용하면 이미지, 포트, 볼륨, 초기화 SQL, healthcheck를 고정할 수 있어 같은 환경을 반복해서 만들기 쉽다.

또한 `docker compose up -d`로 전체 실험 환경을 한 번에 실행하고, `docker compose ps`로 상태를 확인할 수 있다. 동기화 실험은 여러 번 반복해야 하므로, 환경을 쉽게 올리고 내릴 수 있는 구성이 중요하다.

## 5. 권장 운영체제

실험 환경은 가능하면 Linux로 통일한다.

추천 순서:

1. Ubuntu Linux 직접 설치 환경
2. Windows + WSL2 Ubuntu + Docker Desktop
3. 순수 Windows 직접 실행은 비추천

이유:

- Docker, PostgreSQL, Qdrant, pgvector가 Linux 컨테이너 환경에서 자연스럽게 동작한다.
- OS와 파일시스템 차이가 실험 결과에 영향을 주지 않도록 환경을 고정할 수 있다.
- 실험 환경을 다른 장비로 옮길 때 Docker Compose 기반 구성이 재현하기 쉽다.

## 6. 공통 실험 조건

| 항목 | 값 |
|---|---|
| 실행 환경 | 로컬 Linux 또는 WSL2 Ubuntu |
| 외부 API | 사용하지 않음 |
| 컨테이너 | Docker Compose |
| DBMS | PostgreSQL |
| 전용 벡터DB | Qdrant |
| RDBMS 통합 벡터 검색 | pgvector |
| 임베딩 모델 | sentence-transformers/all-MiniLM-L6-v2 |
| 스모크 테스트 데이터 | 1,000건 |
| 본 실험 데이터 | 10,000건 |
| 수정 대상 | 전체 문서의 약 10% |
| 검색 단위 | 문서 단위 |

현재 스모크 테스트는 1,000건 mock embedding으로 완료했다. 본 실험에서는 10,000건 데이터와 실제 `all-MiniLM-L6-v2` 임베딩을 사용할 예정이다.

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

`embedding_version`은 동기화 불일치 실험에서 가장 중요한 필드이다. PostgreSQL의 최신 문서 버전과 Qdrant payload의 임베딩 버전이 다르면 stale vector로 판단할 수 있다.

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

## 8. 본 실험: 문서 수정과 동기화 불일치 실험

### 목적

문서 원문이 수정된 뒤 임베딩 벡터 갱신이 늦어질 때, stale embedding 문제가 검색 결과에 어떤 영향을 주는지 확인한다.

본 실험은 현재 프로젝트의 핵심 실험이다. 기본 검색 성능이나 필터 검색 성능은 보조적으로만 사용하고, 최종 결론은 동기화 불일치 결과를 중심으로 작성한다.

### 수정 대상

- 전체 문서 중 약 10%를 수정한다.
- 예: 10,000건 중 1,000건 수정
- 수정 전후 의미가 달라지도록 본문을 바꾼다.

예시:

- 수정 전: "사내 VPN 접속 오류 처리 절차"
- 수정 후: "제로트러스트 인증 실패 대응 절차"

이렇게 수정하면 예전 의미와 새 의미가 분명히 달라진다. 따라서 Qdrant에 예전 벡터가 남아 있을 때 old meaning query로는 검색되고, new meaning query로는 검색되지 않는 상황을 확인할 수 있다.

### Qdrant 분리형 지연 조건

| 조건 | 임베딩 갱신 지연 |
|---|---:|
| A1 | 0초 |
| A2 | 5초 |
| A3 | 30초 |
| A4 | 120초 |

### Qdrant 분리형 처리 흐름

1. PostgreSQL의 원문을 먼저 수정한다.
2. PostgreSQL의 `embedding_version`을 증가시킨다.
3. Qdrant의 기존 vector와 payload는 일정 시간 유지한다.
4. 지연 시간 동안 검색을 수행한다.
5. Qdrant payload의 `embedding_version`이 최신인지 확인한다.
6. old meaning query로 수정 문서가 검색되는지 확인한다.
7. new meaning query로 수정 문서가 검색되는지 확인한다.
8. 지연 시간이 지난 뒤 Qdrant vector와 payload를 갱신한다.
9. 갱신 후 다시 old meaning retrieval과 new meaning retrieval을 측정한다.

### pgvector 통합형 처리 흐름

1. 수정 본문을 준비한다.
2. 새 embedding을 생성한다.
3. 본문, embedding, embedding_version을 같은 PostgreSQL update 흐름으로 반영한다.
4. 검색 결과를 측정한다.

정확히 말하면 embedding 계산은 DB 밖에서 수행된다. 하지만 계산 완료 후에는 본문과 embedding을 같은 저장 작업으로 묶을 수 있다는 점을 Qdrant 분리형 구조와 비교한다.

### 측정 지표

| 지표 | 의미 |
|---|---|
| `stale_vector_ratio` | 수정된 문서 중 최신 원문과 Qdrant embedding version이 불일치한 비율 |
| `old_meaning_retrieval_count` | 예전 의미의 벡터로 검색했을 때 수정 문서가 검색 결과에 남아 있는 횟수 |
| `new_meaning_retrieval_count` | 새 의미의 벡터로 검색했을 때 수정 문서가 검색 결과에 나타나는 횟수 |
| `update_latency_ms` | 문서 수정과 벡터 갱신 처리에 걸린 시간 |

stale vector ratio 계산:

```text
stale vector ratio =
최신 원문과 embedding_version이 불일치한 문서 수
/ 수정 대상 문서 수
```

해석 기준:

- Qdrant 갱신 전 `stale_vector_ratio`가 높으면 원문과 벡터가 어긋난 상태가 존재한다는 뜻이다.
- Qdrant 갱신 전 `old_meaning_retrieval_count`가 높으면 검색이 아직 예전 의미를 기준으로 작동한다는 뜻이다.
- Qdrant 갱신 전 `new_meaning_retrieval_count`가 낮으면 최신 원문 의미가 검색에 반영되지 않았다는 뜻이다.
- Qdrant 갱신 후에는 `stale_vector_ratio`가 0에 가까워지고, `new_meaning_retrieval_count`가 증가해야 한다.
- pgvector 통합형 구조는 본문과 벡터를 같은 update 흐름으로 반영하므로, 저장 이후 stale 상태가 남지 않는 것을 기대한다.

## 9. 보조 실험: 기본 검색 및 필터 검색 확인

기본 검색과 필터 검색은 본 연구의 중심 실험이 아니다. 다만 두 구조가 정상적으로 적재와 검색을 수행하는지 확인하기 위한 보조 실험으로 유지한다.

### 기본 검색 확인

- 같은 질의를 Qdrant와 pgvector에 입력한다.
- top-k 결과가 반환되는지 확인한다.
- 평균 검색 시간은 참고용으로만 기록한다.

### 필터 검색 확인

- `category`, `year`, `doc_type` 조건을 함께 적용한다.
- Qdrant는 payload filter를 사용한다.
- pgvector는 SQL `WHERE` 조건과 벡터 정렬을 함께 사용한다.
- 결과 부족률과 검색 가능 여부를 확인한다.

이 보조 실험 결과는 최종 보고서에서 성능 우열을 주장하는 근거로 사용하지 않는다. 동기화 실험을 수행하기 전에 두 구조가 정상적으로 동작하는지 확인하는 용도로만 사용한다.

## 10. 확장 실험: 자원 제한 환경

초기 계획에는 자원 제한 실험도 포함되어 있었다. 현재 연구 범위가 축소되었으므로 자원 제한 실험은 필수 실험이 아니라 시간이 남을 때 수행하는 확장 실험으로 둔다.

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
6. 기본 의존성 설치 확인

### 2단계: 스모크 테스트

스모크 테스트는 이미 1,000건 mock embedding으로 수행했다. 다시 수행하려면 다음 명령을 사용한다.

```bash
ragdb-exp generate --documents 1000 --queries 20
ragdb-exp embed-docs --mock
ragdb-exp embed-queries --mock
ragdb-exp load
ragdb-exp search --engine qdrant --index 0 --k 5
ragdb-exp search --engine pgvector --index 0 --k 5
ragdb-exp sync-delay --delay-seconds 0 --modify-ratio 0.02 --mock --output results/sync_delay_smoke.csv
```

### 3단계: 실제 모델 기반 데이터 준비

1. 실제 모델 실행용 의존성 설치
2. 10,000건 문서 생성 또는 외부 데이터셋 다운로드 후 변환
3. 질의 100개 준비
4. 문서 임베딩 생성
5. 질의 임베딩 생성
6. 두 구조에 데이터 적재

명령 예시:

```bash
source .venv/bin/activate
pip install -r requirements-model.txt
ragdb-exp generate --documents 10000 --queries 100
ragdb-exp embed-docs --batch-size 64
ragdb-exp embed-queries --batch-size 64
ragdb-exp load
```

### 4단계: 동기화 불일치 본 실험

각 지연 조건마다 시작 상태를 동일하게 맞추기 위해 `ragdb-exp load`를 다시 실행한 뒤 sync-delay를 수행한다.

```bash
ragdb-exp load
ragdb-exp sync-delay --delay-seconds 0 --modify-ratio 0.1 --output results/sync_delay_0s.csv

ragdb-exp load
ragdb-exp sync-delay --delay-seconds 5 --modify-ratio 0.1 --output results/sync_delay_5s.csv

ragdb-exp load
ragdb-exp sync-delay --delay-seconds 30 --modify-ratio 0.1 --output results/sync_delay_30s.csv

ragdb-exp load
ragdb-exp sync-delay --delay-seconds 120 --modify-ratio 0.1 --output results/sync_delay_120s.csv
```

### 5단계: 결과 정리

1. 각 지연 조건의 CSV 결과를 하나의 표로 정리한다.
2. Qdrant의 갱신 전후 `stale_vector_ratio` 변화를 확인한다.
3. Qdrant의 old meaning retrieval과 new meaning retrieval 변화를 확인한다.
4. pgvector 통합형 구조의 update 방식과 비교한다.
5. 최종 보고서에서는 검색 속도 우열보다 동기화 관리 방식의 차이를 중심으로 해석한다.

## 12. 결과표 템플릿

### 동기화 불일치 실험

| 구조 | 지연 조건 | stale vector ratio before | stale vector ratio after | old meaning before | old meaning after | new meaning before | new meaning after | update latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qdrant 분리형 | 0초 |  |  |  |  |  |  |  |
| Qdrant 분리형 | 5초 |  |  |  |  |  |  |  |
| Qdrant 분리형 | 30초 |  |  |  |  |  |  |  |
| Qdrant 분리형 | 120초 |  |  |  |  |  |  |  |
| pgvector 통합형 | 기준 |  |  |  |  |  |  |  |

### 보조 검색 확인

| 구조 | 실험 | 데이터 수 | k | Avg latency | Recall@K | 비고 |
|---|---|---:|---:|---:|---:|---|
| Qdrant 분리형 | 기본 검색 |  |  |  |  | 보조 확인 |
| pgvector 통합형 | 기본 검색 |  |  |  |  | 보조 확인 |
| Qdrant 분리형 | 필터 검색 |  |  |  |  | 보조 확인 |
| pgvector 통합형 | 필터 검색 |  |  |  |  | 보조 확인 |

### 운영 구조 비교

| 항목 | PostgreSQL + Qdrant | PostgreSQL + pgvector |
|---|---|---|
| 컨테이너 수 | PostgreSQL + Qdrant, 총 2개 | PostgreSQL 1개 |
| 전체 실험 환경 컨테이너 | 두 구조 합산 총 3개 | 두 구조 합산 총 3개 |
| 원문과 벡터 저장 위치 | 분리 | 통합 |
| 동기화 로직 | 필요 | 상대적으로 단순 |
| stale embedding 발생 가능성 | 있음 | 낮음 |
| 불일치 확인 대상 | PostgreSQL 문서 버전 + Qdrant payload 버전 | PostgreSQL 내부 상태 |
| SQL 통합성 | 낮음 | 높음 |

## 13. 최종 결론에서 답해야 할 질문

최종 보고서에서는 아래 질문에 답하면 된다.

1. 문서 수정 후 Qdrant 분리형 구조에서 stale embedding이 실제로 발생하는가?
2. Qdrant의 벡터 갱신 전에는 검색 결과가 예전 의미를 기준으로 남아 있는가?
3. Qdrant의 벡터 갱신 후에는 새 의미가 검색 결과에 반영되는가?
4. pgvector 통합형 구조는 원문과 벡터를 더 단순하게 함께 관리할 수 있는가?
5. 폐쇄망 환경에서 동기화 로직을 별도로 운영하는 부담은 어떤 의미를 가지는가?
6. 문서 수정이 자주 발생하는 환경에서는 어떤 구조가 더 관리하기 쉬운가?

## 14. 현재 폴더에 있는 주요 파일

| 파일 | 용도 |
|---|---|
| `term_project_mid_report.md` | 수정된 중간보고서 본문 |
| `experiment_report.md` | 현재까지의 실험 진행 보고서 |
| `experiment_plan.md` | 실험 계획 및 실행 순서 |
| `docker-compose.yml` | 기본 Docker Compose 구성 |
| `docker-compose.r2.yml` | 2 CPU / 4 GB 확장 실험 조건 |
| `docker-compose.r3.yml` | 1 CPU / 2 GB 확장 실험 조건 |
| `requirements.txt` | 기본 Python 의존성 |
| `requirements-model.txt` | 실제 임베딩 모델 실행용 의존성 |
| `src/ragdb_experiment/` | 실험 실행 코드 |
| `infra/` | PostgreSQL 초기화 SQL |

## 15. 주의사항

- 현재 연구 중심은 성능 최적화가 아니라 동기화 불일치 실험이다.
- 기본 검색과 필터 검색 결과는 보조 확인으로만 사용한다.
- 각 지연 조건 실험 전에는 `ragdb-exp load`로 시작 상태를 동일하게 맞춘다.
- 데이터셋, 질의 집합, 임베딩 모델은 두 구조에서 반드시 동일하게 유지한다.
- 스모크 테스트 결과를 최종 성능 결론으로 사용하지 않는다.
- 최종 보고서에서는 검색 속도보다 stale embedding 발생 여부와 관리 방식 차이를 중심으로 해석한다.
