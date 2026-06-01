# LLM Refactoring Experiment

Java 코드 스멜에 대한 LLM 자동 리팩토링 실험 환경이다. `test_plan.md`의 절차에 맞춰 Maven, JUnit 5, PMD, Checkstyle, CPD, CSV 결과 기록 파일을 기준으로 구성했다.

## 구조

```text
llm-refactoring-experiment/
  pom.xml
  samples.csv
  config/
    pmd-ruleset.xml
    checkstyle.xml
  prompts/
    basic_prompt.txt
    constrained_prompt.txt
  results/
    raw_outputs/
    refactored_sources/
    metrics_before.csv
    metrics_after.csv
    human_evaluation.csv
    final_summary.csv
  scripts/
    compile_main.sh
    collect_simple_metrics.py
    make_prompt.py
  src/main/java/experiment/
  src/test/java/experiment/
```

## 기본 실행

Maven이 설치되어 있으면 다음 명령으로 원본 샘플을 검증한다.

```bash
mvn test
mvn pmd:pmd
mvn pmd:cpd
mvn checkstyle:checkstyle
```

현재 머신처럼 Maven이 없을 때는 최소한 main 소스 컴파일과 단순 지표 수집을 먼저 수행할 수 있다.

```bash
./scripts/compile_main.sh
python3 scripts/collect_simple_metrics.py
```

## LLM 프롬프트 생성

샘플 코드를 프롬프트 템플릿에 삽입한다.

```bash
python3 scripts/make_prompt.py --template prompts/constrained_prompt.txt --source src/main/java/experiment/longmethod/sample01/OrderProcessor.java
```

응답 원문은 `results/raw_outputs/{model}/{sample_id}.md`에 저장하고, 추출한 Java 코드는 `results/refactored_sources/{model}/{sample_id}/` 아래에 둔다.

## 샘플 ID

- `LM01`: Long Method
- `DC01`: Duplicated Code
- `LC01`: Large Class/Data Class

최소 실험 규모를 채우려면 `LM02`-`LM10`, `DC02`-`DC10`을 같은 패키지 규칙으로 추가한다.
