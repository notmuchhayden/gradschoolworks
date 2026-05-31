# 실험 준비

## Python 가상 환경 설정
1. 실험 스크립트 환경을 위하여 Python 가상 환경을 설정한다. 다음 명령어를 사용하여 가상 환경을 만들고 필요한 패키지를 설치한다.
``` shell
cd AdvancedDatabase
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
2. 위 shell 명령어를 실행하는 도중에 .venv 가 제대로 작동하지 않는 문제가 있어서 해결함.

3. 임베딩 모델 설치
- sentence-transformers : 임베딩 모델을 불러오고 실행하는 Python 라이브러리
- all-MiniLM-L6-v2 : Hugging Face Hub에 있는 사전학습 모델 <= AI 추천

>> **폐쇄망 환경을 가정한다면 염두해 두어야 할 점.**
>> * 인터넷이 되는 환경에서 모델을 미리 다운로드한다.
>> * Hugging Face 캐시 디렉터리를 폐쇄망 환경으로 옮긴다.
>> * 또는 모델 파일을 프로젝트 외부의 로컬 경로에 저장하고 그 경로를 EMBEDDING_MODEL로 지정한다.

4. 임베딩 모델 다운로드 및 설치
- sentence-transformers 라이브러리를 사용하여 Hugging Face Hub에서 all-MiniLM-L6-v2 모델을 다운로드한다. 이 모델은 384차원 문장 임베딩을 생성하며, 비교적 가볍고 빠르게 실행할 수 있다. 다음 Python 코드를 실행하여 모델을 다운로드한다.

``` python
from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
```
- 모델이 다운로드되고 캐시에 저장되면, 이후에는 인터넷 연결 없이도 모델을 사용할 수 있다. 모델이 저장된 경로는 Hugging Face Hub의 캐시 디렉터리에 위치한다. 일반적으로 `~/.cache/huggingface/transformers` 경로에 저장된다. 이 경로를 폐쇄망 환경으로 옮기거나, 모델 파일을 프로젝트 외부의 로컬 경로에 저장하여 EMBEDDING_MODEL로 지정할 수 있다.

5. 모델 실행용 의존성 설치
``` shell
# 모델 실행용 의존성 설치 (requirements-model.txt에 torch + sentence-transformers 포함)
pip install -r requirements-model.txt
```

6. Docker 설치 및 설정
- Docker, Compose 설치 확인
    * 다음 명령어로 Docker가 설치되어 있고 실행 중인지 확인한다.
``` shell
docker --version
docker compose version
```

- `docker compose up -d` 실행
    * 프로젝트 루트 디렉터리에서 다음 명령어를 실행하여 Docker 컨테이너를 백그라운드에서 시작한다.

``` shell
(.venv) seaeast2@seaeast2-Ubuntu:~/works/gradschoolworks/AdvancedDatabase$ docker compose up -d
[+] Running 7/7
 ✔ Network advanceddatabase_default                Created                                                                                       0.0s 
 ✔ Volume advanceddatabase_qdrant_data             Created                                                                                       0.0s 
 ✔ Volume advanceddatabase_postgres_meta_data      Created                                                                                       0.0s 
 ✔ Volume advanceddatabase_postgres_pgvector_data  Created                                                                                       0.0s 
 ✔ Container rag_qdrant                            Started                                                                                       0.5s 
 ✔ Container rag_postgres_meta                     Started                                                                                       0.5s 
 ✔ Container rag_postgres_pgvector                 Started                                                                                       0.5s 
```

- `docker compose ps`로 세 컨테이너 상태 확인

``` shell
(.venv) seaeast2@seaeast2-Ubuntu:~/works/gradschoolworks/AdvancedDatabase$ docker compose ps
NAME                    IMAGE                    COMMAND                   SERVICE             CREATED          STATUS                    PORTS
rag_postgres_meta       postgres:16              "docker-entrypoint.s…"   postgres_meta       14 seconds ago   Up 13 seconds (healthy)   0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp
rag_postgres_pgvector   pgvector/pgvector:pg16   "docker-entrypoint.s…"   postgres_pgvector   14 seconds ago   Up 13 seconds (healthy)   0.0.0.0:5434->5432/tcp, [::]:5434->5432/tcp
rag_qdrant              qdrant/qdrant:v1.12.4    "./entrypoint.sh"         qdrant              14 seconds ago   Up 13 seconds (healthy)   0.0.0.0:6333-6334->6333-6334/tcp, [::]:6333-6334->6333-6334/tcp
```

7. Python 가상환경 활성화 및 실험 명령어 ragdb-exp 등록
- radgdb-exp 는 pyproject.toml에 등록된 명령어로, 실험 스크립트를 실행하는 명령어이다. 다음 명령어로 가상환경을 활성화하고 ragdb-exp 명령어를 사용할 수 있다.


8. 실험 데이터 다운로드 및 준비
- Hugging Face ag_news 데이터셋 다운로드 및 준비
    * https://huggingface.co/datasets/fancyzhx/ag_news

** 의존성 설치 **
    - venv 안에서 다음 명령어 입력
    - datasets 라이브러리는 Hugging Face Hub에서 데이터셋을 다운로드하고 로드하는 데 사용된다.

``` shell
pip install datasets
```
8. agnews_gen.py 에서 AG News 데이터를 jsonl 형식으로 변환하는 스크립트 작성
``` shell
(.venv) seaeast2@seaeast2-Ubuntu:~/works/gradschoolworks/AdvancedDatabase$ python src/ragdb_experiment/agnews_gen.py 
README.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8.07k/8.07k [00:00<00:00, 10.2MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
data/train-00000-of-00001.parquet: 100%|█████████████████████████████████████████████████████████████████████████| 18.6M/18.6M [00:05<00:00, 3.46MB/s]
data/test-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████| 1.23M/1.23M [00:01<00:00, 961kB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████| 120000/120000 [00:00<00:00, 489399.43 examples/s]
```
- data/documents.jsonl 에 AG News 데이터셋 생성됨

9. test_queries_gen.py 에서 테스트 쿼리를 생성
``` shell
(.venv) seaeast2@seaeast2-Ubuntu:~/works/gradschoolworks/AdvancedDatabase$ python src/ragdb_experiment/test_queries_gen.py  
```

10. 문서 임베딩 생성
- '--batch-size 64' 을 사용하면 모델이 한 번에 64개의 문서를 처리하여 임베딩을 생성한다. 배치 크기를 조절하여 메모리 사용량과 처리 속도를 최적화할 수 있다.
``` shell
ragdb-exp embed-docs --batch-size 64

(.venv) seaeast2@seaeast2-Ubuntu:~/works/gradschoolworks/AdvancedDatabase$ ragdb-exp embed-docs --batch-size 64
modules.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<00:00, 1.17MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.0:00<?, ?B/s]
config_sentence_transformers.json: 100%|██████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 402kB/s]
README.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10.5k/10.5k [00:00<00:00, 18.2MB/s]
sentence_bert_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<00:00, 189kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 612/612 [00:00<00:00, 2.28MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 90.9M/90.9M [00:35<00:00, 2.54MB/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2669.90it/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 350/350 [00:00<00:00, 1.20MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 1.47MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 6.19MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 398kB/s]
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 677kB/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.63s/it]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.09s/it]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.44s/it]
```
- data/documents.embedded.jsonl 에 문서 임베딩이 생성됨


11. 질의 임베딩 생성
- 벡터DB 는 질의와 문서 간의 유사도를 계산하여 관련 문서를 검색하는데, 이를 위해 질의도 임베딩으로 변환해야 한다. 다음 명령어로 test_queries_gen.py에서 생성된 테스트 쿼리를 임베딩으로 변환한다.

``` shell
ragdb-exp embed-queries --batch-size 64

(.venv) seaeast2@seaeast2-Ubuntu:~/works/gradschoolworks/AdvancedDatabase$ ragdb-exp embed-queries --batch-size 64
embedding batches:   0%|                                                                                                        | 0/2 [00:00<?, ?it/s]Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3761.61it/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.42it/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.91it/s]

```
- data/queries.embedded.jsonl 에 질의 임베딩이 생성됨

12. 두 구조에 데이터 적재 (DB에 적재)
``` shell
ragdb-exp load
```

# 실험 실행
13. 동기화 불일치 본 실험
- 각 지연 조건마다 시작 상태를 동일하게 맞추기 위해 `ragdb-exp load`를 다시 실행한 뒤 sync-delay를 수행한다.
- export ATEN_CPU_CAPABILITY=default 는 PyTorch가 CPU에서 사용할 수 있는 최적의 명령어 집합을 자동으로 감지하도록 하는 환경 변수 설정이다. 이 설정은 CPU에서 PyTorch가 최적의 성능을 발휘할 수 있도록 도와준다. 실험을 실행하기 전에 이 환경 변수를 설정하여 CPU에서 PyTorch가 최적의 성능을 낼 수 있도록 한다.

```bash
export ATEN_CPU_CAPABILITY=default

ragdb-exp load
ragdb-exp sync-delay --delay-seconds 0 --modify-ratio 0.1 --output results/sync_delay_0s.csv

ragdb-exp load
ragdb-exp sync-delay --delay-seconds 5 --modify-ratio 0.1 --output results/sync_delay_5s.csv

ragdb-exp load
ragdb-exp sync-delay --delay-seconds 30 --modify-ratio 0.1 --output results/sync_delay_30s.csv
```

# 결과 정리
14. 실험 결과 분석
1) 각 지연 조건의 CSV 결과를 하나의 표로 정리한다.
2) Qdrant의 갱신 전후 `stale_vector_ratio` 변화를 확인한다.
3) Qdrant의 old meaning retrieval과 new meaning retrieval 변화를 확인한다.
4) pgvector 통합형 구조의 update 방식과 비교한다.
5) 최종 보고서에서는 검색 속도 우열보다 동기화 관리 방식의 차이를 중심으로 해석한다.


```
- 항목 : 의미
- engine : 실험 대상 구조. qdrant_split은 PostgreSQL + Qdrant 분리형, pgvector_integrated는 PostgreSQL + pgvector 통합형
- delay_seconds : Qdrant 벡터 갱신을 의도적으로 지연한 시간
- modified_docs : 전체 문서 중 수정한 문서 수
- checked_docs : old/new meaning 검색 측정에 사용한 수정 문서 수. 현재 최대 100개
- stale_vector_ratio_before_sync : Qdrant 갱신 전, 수정 문서 중 Qdrant의 embedding_version이 최신 PostgreSQL 문서 버전보다 뒤처진 비율
- stale_vector_ratio_after_sync : Qdrant 갱신 후, 수정 문서 중 여전히 stale 상태인 비율
- old_meaning_retrieval_count_before_sync : Qdrant 갱신 전, 수정 전 의미의 embedding으로 검색했을 때 수정 문서가 top-k 안에 다시 검색된 횟수
- old_meaning_retrieval_count_after_sync : Qdrant 갱신 후, 수정 전 의미의 embedding으로 검색했을 때 수정 문서가 top-k 안에 검색된 횟수
- new_meaning_retrieval_count_before_sync : Qdrant 갱신 전, 수정 후 의미의 embedding으로 검색했을 때 수정 문서가 top-k 안에 검색된 횟수
- new_meaning_retrieval_count_after_sync : Qdrant 갱신 후, 수정 후 의미의 embedding으로 검색했을 때 수정 문서가 top-k 안에 검색된 횟수
- delay_probe_interval_seconds : 지연 시간 동안 반복 검색을 몇 초 간격으로 수행할지 설정한 값
- delay_probe_docs : 지연 시간 동안 각 반복 검색에서 확인한 수정 문서 수
- delay_probe_count : 지연 시간 동안 실제 수행한 반복 검색 횟수
- delay_observed_seconds : 반복 검색으로 실제 관찰한 지연 구간 시간
- delay_old_meaning_retrieval_total : 지연 시간 동안 old meaning 검색에서 stale 문서가 검색된 누적 횟수
- delay_new_meaning_retrieval_total : 지연 시간 동안 new meaning 검색에서 수정 문서가 검색된 누적 횟수
- qdrant_vector_update_latency_ms : Qdrant의 벡터와 payload를 실제로 갱신하는 데 걸린 시간
- update_latency_ms : 전체 처리 시간. PostgreSQL 원문 수정, 갱신 전 측정, 지연 구간 probe, Qdrant 벡터 갱신 등을 포함한 end-to-end 시간
```