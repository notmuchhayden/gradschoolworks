# 통합형 및 분리형 벡터 저장 구조의 데이터 일관성 비교 연구

## 문서 수정 후 stale embedding이 의미 기반 검색 결과에 미치는 영향

> 본 문서는 최종 보고서 작성을 위한 초안이다. 문장 표현, 참고문헌 형식, 그림 번호, 표 번호, 실험 환경 세부값은 최종 제출 전에 다시 교정한다.

## 초록

본 연구는 의미 기반 문서 검색 시스템에서 원문 데이터와 임베딩 벡터를 함께 관리하는 방식에 따라 데이터 일관성 문제가 어떻게 달라지는지 비교한다. 비교 대상은 `PostgreSQL + Qdrant` 분리형 구조와 `PostgreSQL + pgvector` 통합형 구조이다. 분리형 구조에서는 원문 문서가 PostgreSQL에 저장되고 임베딩 벡터가 Qdrant에 저장되므로, 문서 수정 후 벡터 갱신이 지연되면 최신 원문과 검색용 벡터가 서로 다른 의미를 갖는 stale embedding 상태가 발생할 수 있다. 반면 pgvector 통합형 구조에서는 원문, 메타데이터, 벡터를 같은 PostgreSQL 안에서 관리할 수 있으므로 새 임베딩이 준비된 뒤 본문과 벡터를 같은 갱신 흐름으로 반영할 수 있다.

실험은 AG News 데이터셋 10,000건과 질의 100개를 사용하여 수행하였다. 전체 문서의 10%인 1,000건을 의미가 다른 문장으로 수정하고, Qdrant 벡터 갱신 지연을 0초, 5초, 30초로 설정하였다. 실험 결과 Qdrant 분리형 구조에서는 모든 지연 조건에서 갱신 전 `stale_vector_ratio_before_sync`가 1.0으로 나타났고, 수정 전 의미 검색은 100건 모두 검색된 반면 수정 후 의미 검색은 0건으로 나타났다. Qdrant 벡터 갱신 후에는 stale 비율이 0.0으로 감소하고 수정 후 의미 검색 결과가 100건으로 회복되었다. 지연 시간 동안 반복 검색한 결과, 5초 지연에서는 stale 검색 결과가 20회, 30초 지연에서는 100회 누적 관찰되었다. 이는 지연 시간이 길어질수록 stale 상태가 사용자 검색에 더 오래 노출됨을 보여준다. 본 연구는 폐쇄망 또는 제한된 운영 환경에서 분리형 벡터DB 구조가 별도 동기화 관리 부담을 갖는다는 점을 실험적으로 확인하였다.

**키워드:** 벡터 데이터베이스, pgvector, Qdrant, stale embedding, 데이터 일관성, 의미 기반 검색

## 1. 서론

최근 문서 검색 시스템에서는 키워드 일치 기반 검색뿐 아니라 문장이나 문서의 의미를 벡터로 표현하여 유사한 문서를 찾는 의미 기반 검색이 널리 사용된다. 이러한 시스템에서는 문서 본문을 임베딩 모델로 변환한 벡터를 저장하고, 사용자의 질의 역시 벡터로 변환하여 가까운 문서 벡터를 검색한다.

벡터 검색 시스템을 구성하는 방식은 크게 두 가지로 나눌 수 있다. 첫째는 Qdrant와 같은 전용 벡터 데이터베이스를 별도로 두고, 원문 데이터는 기존 RDBMS에 저장하는 분리형 구조이다. 둘째는 PostgreSQL의 pgvector 확장을 사용하여 원문과 벡터를 같은 RDBMS 안에서 관리하는 통합형 구조이다. 전용 벡터DB는 벡터 검색 기능에 특화되어 있지만, 원문 저장소와 벡터 저장소가 분리되므로 두 저장소의 동기화 문제가 발생할 수 있다.

본 연구는 단순 검색 속도 비교가 아니라, 문서 수정 후 원문과 임베딩 벡터가 일시적으로 불일치하는 stale embedding 문제에 초점을 둔다. 특히 문서 원문은 최신 상태로 바뀌었지만 Qdrant에 저장된 벡터가 이전 문서 의미를 유지하는 경우, 검색 결과가 최신 원문이 아니라 과거 의미를 기준으로 반환될 수 있다. 본 연구의 목적은 이 현상을 제한된 실험 환경에서 재현하고, 통합형 구조와 분리형 구조의 데이터 일관성 관리 차이를 비교하는 것이다.

## 2. 관련 배경

벡터DB는 텍스트, 이미지, 음성 등 비정형 데이터를 고차원 벡터로 변환하여 유사도 검색을 수행하는 시스템이다. 문서 검색에서는 일반적으로 원문 문서를 임베딩 모델로 벡터화하고, 질의 문장도 같은 임베딩 모델로 벡터화한 뒤 두 벡터 사이의 거리를 계산한다.

Qdrant는 전용 벡터DB로서 벡터와 payload를 point 단위로 저장한다. payload에는 문서 ID, 카테고리, 작성 연도, 문서 유형, 임베딩 버전 같은 메타데이터를 둘 수 있다. 그러나 원문 문서는 일반적으로 별도의 RDBMS에 저장되므로, 문서 수정 시 RDBMS와 Qdrant 양쪽을 모두 갱신해야 한다.

pgvector는 PostgreSQL 내부에서 벡터 타입과 벡터 거리 검색을 제공하는 확장이다. 이 구조에서는 문서 원문, 메타데이터, 임베딩 벡터를 같은 PostgreSQL 테이블 안에서 관리할 수 있다. 임베딩 계산 자체는 여전히 DB 외부에서 수행되지만, 계산된 벡터를 본문과 함께 같은 update 흐름으로 저장할 수 있다는 점에서 관리 구조가 단순하다.

## 3. 연구 질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

1. 문서 수정 후 Qdrant 분리형 구조에서 stale embedding이 실제로 발생하는가?
2. Qdrant의 벡터 갱신 전에는 검색 결과가 예전 의미를 기준으로 남아 있는가?
3. Qdrant의 벡터 갱신 후에는 새 의미가 검색 결과에 반영되는가?
4. pgvector 통합형 구조는 원문과 벡터를 더 단순하게 함께 관리할 수 있는가?
5. 폐쇄망 환경에서 동기화 로직을 별도로 운영하는 부담은 어떤 의미를 가지는가?
6. 문서 수정이 자주 발생하는 환경에서는 어떤 구조가 더 관리하기 쉬운가?

## 4. 실험 환경 및 데이터

### 4.1 실험 구조

실험 환경은 Docker Compose로 구성하였다. 비교 구조는 두 가지이지만 실제 컨테이너는 세 개이다.

| 서비스 | 역할 | 포트 |
|---|---|---:|
| `postgres_meta` | Qdrant 분리형 구조의 원문 및 메타데이터 저장소 | 5433 |
| `postgres_pgvector` | pgvector 통합형 구조의 원문, 메타데이터, 벡터 저장소 | 5434 |
| `qdrant` | 전용 벡터DB 검색 저장소 | 6333, 6334 |

![실험 구조도 placeholder](graph1.png)

**그림 1.** PostgreSQL + Qdrant 분리형 구조와 PostgreSQL + pgvector 통합형 구조 비교.

### 4.2 데이터셋

실험 데이터는 Hugging Face의 AG News 데이터셋을 사용하였다. 학습 데이터 중 10,000건을 선택하여 프로젝트 스키마에 맞게 변환하였다. 각 문서는 `doc_id`, `title`, `content`, `category`, `year`, `doc_type`, `updated_at`, `embedding_version` 필드를 가진다.

| 항목 | 값 |
|---|---:|
| 문서 수 | 10,000 |
| 질의 수 | 100 |
| 수정 대상 비율 | 10% |
| 수정 문서 수 | 1,000 |
| 임베딩 모델 | `sentence-transformers/all-MiniLM-L6-v2` |
| 임베딩 차원 | 384 |
| 문서 유형 | AG News 기사 |
| 검색 top-k | 10 |

질의는 AG News의 네 범주인 World, Sports, Business, Sci/Tech에 대해 각 25개씩 총 100개를 작성하였다. 문서와 질의는 같은 임베딩 모델로 벡터화하였다.

### 4.3 수정 문서 생성

동기화 불일치 실험에서는 전체 문서 중 앞쪽 10%를 수정 대상으로 선택하였다. 수정 대상 문서는 기존 AG News 기사 본문을 다음과 같은 새로운 의미의 문장으로 교체하였다.

```text
Zero trust authentication failure response procedure. This document replaces the previous access issue article and requires checking authentication policy, device trust, and access control logs.
```

이 수정은 기존 뉴스 기사 의미와 다른 의미를 갖도록 설계하였다. 수정 후에는 `embedding_version`을 1 증가시키고, 수정된 본문으로 새 임베딩을 생성하였다.

## 5. 실험 방법

### 5.1 Qdrant 분리형 구조 실험

Qdrant 분리형 구조에서는 다음 순서로 실험을 수행하였다.

1. PostgreSQL의 문서 원문을 먼저 수정한다.
2. PostgreSQL의 `embedding_version`을 증가시킨다.
3. Qdrant의 기존 vector와 payload는 갱신하지 않고 유지한다.
4. Qdrant 갱신 전 stale 상태와 검색 결과를 측정한다.
5. 설정한 지연 시간 동안 반복 검색을 수행한다.
6. 지연 시간이 끝나면 Qdrant vector와 payload를 갱신한다.
7. Qdrant 갱신 후 stale 상태와 검색 결과를 다시 측정한다.

Qdrant 갱신 지연 조건은 0초, 5초, 30초로 설정하였다.

### 5.2 pgvector 통합형 구조 실험

pgvector 통합형 구조에서는 새 임베딩이 준비된 뒤 PostgreSQL 안에서 `content`, `updated_at`, `embedding_version`, `embedding`을 같은 update 흐름으로 반영하였다. 이 구조는 Qdrant처럼 원문 저장소와 벡터 저장소가 분리되어 있지 않으므로, 저장 이후 stale 상태가 남지 않는 기준 구조로 사용하였다.

### 5.3 측정 지표

| 지표 | 의미 |
|---|---|
| `stale_vector_ratio_before_sync` | Qdrant 갱신 전 수정 문서 중 버전 불일치 문서 비율 |
| `stale_vector_ratio_after_sync` | Qdrant 갱신 후 수정 문서 중 버전 불일치 문서 비율 |
| `old_meaning_retrieval_count_before_sync` | 갱신 전 수정 전 의미로 검색했을 때 수정 문서가 검색된 횟수 |
| `new_meaning_retrieval_count_before_sync` | 갱신 전 수정 후 의미로 검색했을 때 수정 문서가 검색된 횟수 |
| `delay_old_meaning_retrieval_total` | 지연 시간 동안 수정 전 의미 검색이 stale 문서를 반환한 누적 횟수 |
| `delay_new_meaning_retrieval_total` | 지연 시간 동안 수정 후 의미 검색이 수정 문서를 반환한 누적 횟수 |
| `qdrant_vector_update_latency_ms` | Qdrant vector와 payload를 실제 갱신하는 데 걸린 시간 |
| `update_latency_ms` | 전체 실험 처리 시간 |

## 6. 실험 결과

### 6.1 Qdrant 분리형 구조 결과

| 지연 시간 | stale before | stale after | old before | old after | new before | new after | probe 수 | 지연 중 old 누적 | 지연 중 new 누적 | Qdrant 갱신 ms | 전체 처리 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0초 | 1.0 | 0.0 | 100 | 0 | 0 | 100 | 0 | 0 | 0 | 472.54 | 26770.84 |
| 5초 | 1.0 | 0.0 | 100 | 0 | 0 | 100 | 1 | 20 | 0 | 481.08 | 31806.97 |
| 30초 | 1.0 | 0.0 | 100 | 0 | 0 | 100 | 5 | 100 | 0 | 479.34 | 59218.70 |

![Qdrant 지연 조건별 stale 상태 변화 placeholder](graph2.png)

**그림 2.** Qdrant 갱신 전후 stale vector ratio 변화.

![지연 시간 동안 stale 검색 누적 횟수 placeholder](graph3.png)

**그림 3.** Qdrant 갱신 지연 시간 동안 old meaning 검색 결과 누적 횟수.

실험 결과 모든 지연 조건에서 Qdrant 갱신 전 `stale_vector_ratio_before_sync`는 1.0으로 나타났다. 이는 수정된 1,000개 문서 전체가 PostgreSQL의 최신 문서 버전과 Qdrant의 임베딩 버전이 불일치했음을 의미한다. Qdrant 갱신 후에는 `stale_vector_ratio_after_sync`가 0.0으로 감소하였다.

검색 결과에서도 같은 경향이 나타났다. 갱신 전에는 수정 전 의미의 벡터로 검색했을 때 100개 확인 문서가 모두 검색되었고, 수정 후 의미의 벡터로 검색했을 때는 0개가 검색되었다. 반대로 Qdrant 갱신 후에는 수정 전 의미 검색 결과가 0개로 감소하고, 수정 후 의미 검색 결과가 100개로 증가하였다.

지연 시간 동안의 반복 검색 결과도 중요하다. 5초 지연 조건에서는 1회 probe가 수행되었고, probe 대상 문서 20개 모두 old meaning 검색 결과에 나타났다. 30초 지연 조건에서는 5회 probe가 수행되어 old meaning 검색 결과가 총 100회 누적되었다. new meaning 검색 결과는 두 조건 모두 0회였다. 이는 지연 시간이 길어질수록 stale 상태가 더 오래 검색 결과에 노출됨을 보여준다.

### 6.2 pgvector 통합형 구조 결과

| 조건 | stale before | stale after | old before | old after | new before | new after | update ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0초 실험 기준 | 0.0 | 0.0 | 0 | 0 | 100 | 100 | 1715.06 |
| 5초 실험 기준 | 0.0 | 0.0 | 0 | 0 | 100 | 100 | 1692.08 |
| 30초 실험 기준 | 0.0 | 0.0 | 0 | 0 | 100 | 100 | 1767.27 |

![Qdrant와 pgvector의 stale 상태 비교 placeholder](graph4.png)

**그림 4.** Qdrant 분리형 구조와 pgvector 통합형 구조의 stale 상태 비교.

pgvector 통합형 구조에서는 stale 상태가 관찰되지 않았다. 이는 원문과 벡터가 같은 PostgreSQL 안에서 관리되고, 새 임베딩이 준비된 뒤 같은 update 흐름으로 반영되기 때문이다. 단, 임베딩 계산 자체가 PostgreSQL 내부에서 수행된다는 뜻은 아니며, 본 실험에서는 Python 코드가 임베딩을 생성한 뒤 PostgreSQL에 저장하였다.

### 6.3 갱신 시간 해석

Qdrant의 `qdrant_vector_update_latency_ms`는 0초, 5초, 30초 조건에서 각각 약 472.54ms, 481.08ms, 479.34ms로 거의 유사하였다. 이는 지연 조건이 Qdrant의 실제 upsert 성능을 변화시킨 것이 아니라, Qdrant 갱신 전 stale 상태가 유지되는 시간을 변화시킨 것임을 의미한다.

반면 `update_latency_ms`는 end-to-end 처리 시간이다. 이 값에는 PostgreSQL 원문 수정, 갱신 전 측정, 지연 시간 동안 반복 검색, Qdrant 갱신, 갱신 후 측정이 포함된다. 따라서 이 값은 순수 Qdrant 벡터 갱신 시간으로 해석하면 안 된다.

## 7. 논의

### 7.1 Qdrant 분리형 구조의 stale embedding 문제

실험 결과 Qdrant 분리형 구조에서는 문서 수정 후 Qdrant 벡터가 갱신되기 전까지 stale embedding 상태가 명확히 발생하였다. 이 상태에서 검색 결과는 최신 원문 의미가 아니라 수정 전 의미를 기준으로 반환되었다. 이는 원문 저장소와 벡터 저장소가 분리된 구조에서 별도 동기화 로직이 필수임을 보여준다.

### 7.2 지연 시간과 stale 노출

갱신 전후의 상태값은 0초, 5초, 30초 조건에서 동일하게 나타났다. 그러나 지연 시간 동안 반복 검색한 결과에서는 차이가 나타났다. 5초 조건보다 30초 조건에서 stale 검색 결과가 더 많이 누적되었다. 따라서 지연 시간은 stale 상태의 발생 여부보다 stale 결과가 사용자에게 노출되는 시간과 횟수에 영향을 준다.

### 7.3 pgvector 통합형 구조의 관리 단순성

pgvector 통합형 구조는 원문과 벡터가 같은 PostgreSQL 안에 저장되므로, 새 임베딩이 준비된 뒤 본문과 벡터를 같은 update 흐름으로 반영할 수 있다. 이 점에서 문서 수정이 자주 발생하는 환경에서는 관리 구조가 단순하다. 다만 대규모 벡터 검색 성능이나 고동시성 검색에서는 전용 벡터DB가 장점을 가질 수 있으므로, 본 연구의 결론은 성능 우열이 아니라 데이터 일관성과 동기화 관리 측면의 비교로 제한된다.

### 7.4 폐쇄망 환경에서의 의미

폐쇄망 환경에서는 외부 관리형 큐, 클라우드 모니터링, 자동 복구 서비스를 쓰기 어렵거나 제한될 수 있다. Qdrant 분리형 구조를 사용하려면 PostgreSQL 변경 이벤트를 Qdrant 갱신으로 연결하는 별도 worker, 재시도 로직, 버전 불일치 감시, 장애 복구 절차가 필요하다. 이러한 운영 부담은 검색 결과의 정확성과 직접 연결된다.

## 8. 한계 및 향후 연구

본 실험은 10,000건 규모의 AG News 데이터셋과 로컬 Docker 환경에서 수행되었다. 따라서 대규모 운영 환경의 검색 성능이나 고동시성 상황을 대표하지는 않는다. 또한 문서 수정은 실험적으로 같은 의미의 영어 문장으로 일괄 교체하는 방식으로 구성하였다. 실제 서비스에서는 문서마다 수정 폭과 의미 변화가 다를 수 있다.

향후 연구에서는 다음을 추가로 고려할 수 있다.

1. 1초 단위 지연 조건을 추가하여 stale 노출량 증가 경향을 더 세밀하게 측정한다.
2. 검색 요청 빈도를 모델링하여 초당 검색 요청 수에 따른 stale 결과 노출률을 계산한다.
3. PostgreSQL outbox pattern, CDC, queue 기반 worker 구조를 구현하여 동기화 지연을 줄이는 방안을 비교한다.
4. Qdrant 검색 결과 반환 시 PostgreSQL의 최신 `embedding_version`과 비교하여 stale 결과를 제외하는 보정 방식을 실험한다.
5. 더 큰 데이터셋과 반복 실험을 통해 결과의 안정성을 검증한다.

## 9. 결론

본 연구는 통합형 및 분리형 벡터 저장 구조에서 문서 수정 후 데이터 일관성 관리 방식의 차이를 비교하였다. 실험 결과 `PostgreSQL + Qdrant` 분리형 구조에서는 문서 원문이 수정된 뒤 Qdrant 벡터 갱신이 이루어지기 전까지 stale embedding 상태가 발생하였다. 이 상태에서는 검색 결과가 최신 원문이 아니라 수정 전 의미를 기준으로 반환되었다. Qdrant 벡터 갱신 후에는 stale 상태가 해소되고 새 의미가 검색 결과에 반영되었다.

반면 `PostgreSQL + pgvector` 통합형 구조에서는 원문과 벡터를 같은 PostgreSQL 안에서 관리할 수 있어, 새 임베딩이 준비된 뒤 같은 update 흐름으로 반영할 수 있었다. 따라서 문서 수정이 자주 발생하고 운영 환경이 제한된 상황에서는 pgvector 통합형 구조가 데이터 일관성 관리 측면에서 더 단순하다.

결론적으로 Qdrant와 같은 전용 벡터DB는 벡터 검색 기능과 확장성 측면에서 장점을 가질 수 있지만, 원문 저장소와 벡터 저장소가 분리되는 만큼 동기화 로직을 별도로 설계하고 운영해야 한다. 특히 폐쇄망 환경이나 운영 인력이 제한된 환경에서는 stale embedding이 단순한 내부 상태 불일치가 아니라 사용자 검색 결과의 의미적 오류로 이어질 수 있다. 따라서 벡터DB 구조 선택 시 검색 성능뿐 아니라 문서 수정 빈도, 동기화 지연 허용 범위, 운영 복잡도를 함께 고려해야 한다.

## 참고문헌 초안

[1] T. Taipalus, "Vector database management systems: Fundamental concepts, use-cases, and current challenges," arXiv:2309.11322, 2023.

[2] pgvector, "Open-source vector similarity search for Postgres," https://github.com/pgvector/pgvector.

[3] Qdrant Documentation, "Qdrant vector database documentation," https://qdrant.tech/documentation/.

[4] Hugging Face, "fancyzhx/ag_news dataset," https://huggingface.co/datasets/fancyzhx/ag_news.

[5] Sentence Transformers Documentation, "SentenceTransformers Documentation," https://www.sbert.net/.
