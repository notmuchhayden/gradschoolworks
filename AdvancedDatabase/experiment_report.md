# 1. 실험 준비

## 실험 환경 확인

1. Python 가상 환경 설정
- 실험 스크립트 환경을 위하여 Python 가상 환경을 설정한다. 다음 명령어를 사용하여 가상 환경을 만들고 필요한 패키지를 설치한다.
``` shell
cd AdvancedDatabase
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
- 위 shell 명령어를 실행하는 도중에 .venv 가 제대로 작동하지 않는 문제가 있어서 해결함.
- 임베딩 모델 설치
    - sentence-transformers : 임베딩 모델을 불러오고 실행하는 Python 라이브러리
    - all-MiniLM-L6-v2 : Hugging Face Hub에 있는 사전학습 모델 <= AI 추천

>> **폐쇄망 환경을 가정한다면 염두해 두어야 할 점.**
>> * 인터넷이 되는 환경에서 모델을 미리 다운로드한다.
>> * Hugging Face 캐시 디렉터리를 폐쇄망 환경으로 옮긴다.
>> * 또는 모델 파일을 프로젝트 외부의 로컬 경로에 저장하고 그 경로를 EMBEDDING_MODEL로 지정한다.


2. 임베딩 모델 다운로드 및 설치

2.1. 사전 설명
- sentence-transformers 라이브러리를 사용하여 Hugging Face Hub에서 all-MiniLM-L6-v2 모델을 다운로드한다. 이 모델은 384차원 문장 임베딩을 생성하며, 비교적 가볍고 빠르게 실행할 수 있다. 다음 Python 코드를 실행하여 모델을 다운로드한다.

``` python
from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
```
- 모델이 다운로드되고 캐시에 저장되면, 이후에는 인터넷 연결 없이도 모델을 사용할 수 있다. 모델이 저장된 경로는 Hugging Face Hub의 캐시 디렉터리에 위치한다. 일반적으로 `~/.cache/huggingface/transformers` 경로에 저장된다. 이 경로를 폐쇄망 환경으로 옮기거나, 모델 파일을 프로젝트 외부의 로컬 경로에 저장하여 EMBEDDING_MODEL로 지정할 수 있다.

2.2 모델 실행용 의존성 설치

``` shell
# 모델 실행용 의존성 설치 (requirements-model.txt에 torch + sentence-transformers 포함)
pip install -r requirements-model.txt
```

3. Docker 설치 및 설정
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
- Python 가상환경 활성화 및 실험 명령어 ragdb-exp 등록
    * radgdb-exp 는 pyproject.toml에 등록된 명령어로, 실험 스크립트를 실행하는 명령어이다. 다음 명령어로 가상환경을 활성화하고 ragdb-exp 명령어를 사용할 수 있다.


4. 실험 데이터 다운로드 및 준비
- Hugging Face ag_news 데이터셋 다운로드 및 준비
    * https://huggingface.co/datasets/fancyzhx/ag_news

** 의존성 설치 **
    - venv 안에서 다음 명령어 입력
    - datasets 라이브러리는 Hugging Face Hub에서 데이터셋을 다운로드하고 로드하는 데 사용된다.
``` shell
pip install datasets
```
- agnews_gen.py 에서 AG News 데이터를 jsonl 형식으로 변환하는 스크립트 작성