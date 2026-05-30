# Java 기반 LLM 자동 리팩토링 실험 계획서

## 1. 실험 목적

본 실험의 목적은 LLM 이 Java 코드의 대표적인 코드 스멜을 얼마나 안전하고 효과적으로 리팩토링할 수 있는지 확인하는 것이다. 평가는 단순히 코드가 보기 좋아졌는지가 아니라, 다음 조건을 동시에 만족하는지를 기준으로 한다.

1. 리팩토링 결과가 Java 컴파일을 통과하는가?
2. 기존 JUnit 테스트를 모두 통과하는가?
3. 대상 코드 스멜이 실제로 완화되었는가?
4. 복잡도, 중복도, 정적분석 위반 수가 개선되었는가?
5. 사람이 보기에 유지보수성과 가독성이 개선되었는가?

## 2. 실험 범위

### 2.1 대상 언어

실험 언어는 Java 로 고정한다. Java 를 선택한 이유는 정적 타입, 클래스 기반 객체지향 구조, JUnit 테스트, Maven/Gradle 빌드, SonarQube/PMD/Checkstyle 등 정적분석 도구가 잘 갖춰져 있어 리팩토링 평가에 적합하기 때문이다.

### 2.2 대상 코드 스멜

본 실험은 다음 세 가지 코드 스멜을 대상으로 한다.

| 코드 스멜 | 설명 | 대표 리팩토링 |
| --- | --- | --- |
| Long Method | 하나의 메서드가 여러 책임을 수행하고 길이가 긴 경우 | Extract Method, 조건문 분리, 변수명 개선 |
| Duplicated Code | 유사한 코드가 여러 메서드 또는 클래스에 반복되는 경우 | 공통 메서드 추출, 파라미터화, 템플릿 메서드 |
| Large Class/Data Class | 하나의 클래스가 과도한 필드와 책임을 가지거나 데이터만 보유하는 경우 | Extract Class, 책임 분리, 캡슐화 |

### 2.3 최소 실험 규모와 확장 규모

| 구분 | 최소 실험 | 확장 실험 |
| --- | ---: | ---: |
| 코드 스멜 유형 | 2종 | 3종 |
| 유형별 샘플 수 | 10개 | 10개 |
| 총 샘플 수 | 20개 | 30개 |
| LLM 모델 수 | 2개 | 3개 이상 |
| 프롬프트 조건 | 제약 프롬프트 1개 | 기본/제약 프롬프트 2개 |

최소 실험에서는 Long Method 와 Duplicated Code 를 우선 수행한다. 시간이 허락하면 Large Class/Data Class 를 추가한다.

## 3. 실험 도구

### 3.1 기본 개발 환경

| 항목 | 권장 선택 | 목적 |
| --- | --- | --- |
| JDK | JDK 17 또는 JDK 21 | Java 컴파일 및 실행 |
| 빌드 도구 | Maven | 샘플별 빌드와 테스트 표준화 |
| 테스트 프레임워크 | JUnit 5 | 기능 동등성 검증 |
| 정적분석 | PMD | 복잡도, 코드 스멜성 규칙 위반 측정 |
| 스타일 검사 | Checkstyle | 기본 스타일 위반 측정 |
| 중복 검사 | PMD CPD 또는 jscpd | 중복 코드 측정 |
| 결과 기록 | CSV 또는 스프레드시트 | 실험 결과 정리 |

Maven 을 기본으로 추천한다. Gradle 도 가능하지만, 텀프로젝트에서는 Maven 의 디렉터리 구조와 명령이 단순해서 반복 실험에 더 적합하다.

### 3.2 권장 Maven 명령

```bash
mvn test
mvn -DskipTests compile
mvn pmd:pmd
mvn pmd:cpd
mvn checkstyle:checkstyle
```

실험에서 가장 중요한 명령은 `mvn test`이다. 이 명령은 컴파일과 테스트를 함께 수행하므로 기능 보존 여부를 판단하는 1차 기준으로 사용한다.

## 4. 실험 프로젝트 구조

실험은 하나의 Maven 프로젝트 안에 샘플별 패키지를 나누는 방식으로 구성한다.

```text
llm-refactoring-experiment/
  pom.xml
  README.md
  samples.csv
  results/
    raw_outputs/
    metrics_before.csv
    metrics_after.csv
    human_evaluation.csv
    final_summary.csv
  src/
    main/java/
      experiment/
        longmethod/
          sample01/
            OrderProcessor.java
          sample02/
            ...
        duplicatedcode/
          sample01/
            ReportFormatter.java
          sample02/
            ...
        largeclass/
          sample01/
            CustomerAccount.java
          sample02/
            ...
    test/java/
      experiment/
        longmethod/
          sample01/
            OrderProcessorTest.java
        duplicatedcode/
          sample01/
            ReportFormatterTest.java
        largeclass/
          sample01/
            CustomerAccountTest.java
```

각 샘플은 독립적으로 이해 가능한 작은 코드로 만든다. 한 샘플이 다른 샘플에 의존하지 않도록 한다. 그래야 LLM 결과를 교체하거나 테스트할 때 오류 원인을 쉽게 분리할 수 있다.

## 5. 데이터셋 설계

### 5.1 샘플 작성 원칙

각 샘플은 다음 조건을 만족해야 한다.

1. 명확한 코드 스멜이 하나 이상 포함되어야 한다.
2. 원본 코드는 컴파일되고 JUnit 테스트를 모두 통과해야 한다.
3. 테스트는 정상 입력뿐 아니라 경계값과 예외 상황을 일부 포함해야 한다.
4. public API 는 실험 중 유지되어야 한다.
5. 샘플은 너무 작지 않아야 한다. LLM 이 실제로 구조 개선을 할 여지가 있어야 한다.
6. 샘플은 너무 크지 않아야 한다. 한 번의 프롬프트에 코드와 테스트를 함께 넣을 수 있어야 한다.

### 5.2 샘플 난이도 기준

| 난이도 | 기준 |
| --- | --- |
| Easy | 하나의 클래스와 1~2개 메서드로 구성되며 코드 스멜이 명확함 |
| Medium | 여러 메서드 또는 보조 클래스가 있고 리팩토링 선택지가 2개 이상 있음 |
| Hard | 여러 책임이 섞여 있으며 잘못 리팩토링하면 테스트가 실패하기 쉬움 |

### 5.3 샘플 메타데이터 양식

`samples.csv`에 다음 정보를 기록한다.

| 컬럼 | 예시 | 설명 |
| --- | --- | --- |
| sample_id | LM01 | 샘플 고유 ID |
| smell_type | Long Method | 코드 스멜 유형 |
| difficulty | Medium | 난이도 |
| main_file | OrderProcessor.java | 원본 파일 |
| test_file | OrderProcessorTest.java | 테스트 파일 |
| public_api | process(Order order) | 유지해야 할 API |
| expected_refactoring | Extract Method | 기대 리팩토링 방향 |
| note | 할인/배송/세금 계산 책임이 섞여 있음 | 평가 참고 메모 |

## 6. LLM 비교 조건

### 6.1 비교 모델

실험 시점에 접근 가능한 모델을 사용한다. 최소 2개 모델, 가능하면 3개 모델을 비교한다.

| 모델 구분 | 예시 | 비고 |
| --- | --- | --- |
| OpenAI 계열 | ChatGPT 계열 | 기준 모델 |
| Anthropic 계열 | Claude 계열 | 비교 모델 |
| Google 계열 | Gemini 계열 | 비교 모델 |
| 로컬/오픈소스 | 선택 사항 | 시간과 환경이 허락할 때만 수행 |

상용 모델의 정확한 버전명은 실험 당시 사용한 이름을 결과표에 기록한다. 모델 버전은 시간이 지나면 바뀔 수 있으므로 보고서에는 실험 날짜와 함께 적는다.

### 6.2 프롬프트 조건

최소 실험에서는 제약 프롬프트만 사용한다. 확장 실험에서는 기본 프롬프트와 제약 프롬프트를 비교한다.

#### 기본 프롬프트

```text
다음 Java 코드의 기능을 유지하면서 코드 스멜을 제거하도록 리팩토링하라.
리팩토링된 Java 코드만 제시하라.
```

#### 제약 프롬프트

```text
다음 Java 코드를 리팩토링하라.

목표:
- 기능을 변경하지 않고 코드 스멜을 줄인다.
- 가독성과 유지보수성을 개선한다.

제약 조건:
1. 기존 public 클래스명, public 메서드명, 인자, 반환 형식을 변경하지 말 것.
2. 기존 JUnit 테스트가 모두 통과해야 한다.
3. 새로운 외부 라이브러리나 빌드 설정 변경을 추가하지 말 것.
4. 기능 추가나 삭제를 하지 말 것.
5. 요구되지 않은 아키텍처 변경을 하지 말 것.
6. 리팩토링된 전체 Java 코드만 제시할 것.

입력 코드:
[여기에 Java 코드 삽입]
```

### 6.3 LLM 실행 규칙

1. 같은 샘플에는 모든 모델에 동일한 프롬프트를 사용한다.
2. 한 샘플당 모델별 응답은 1회만 사용한다. 재시도는 별도 실험으로 기록한다.
3. LLM 응답의 로직은 사람이 수정하지 않는다.
4. 코드 블록 제거, 파일명 맞춤, package 선언 조정처럼 실행을 위한 최소한의 형식 수정만 허용한다.
5. 형식 수정을 한 경우 `manual_fix_note`에 기록한다.

## 7. 실험 절차

### 7.1 사전 준비

1. Maven 프로젝트를 생성한다.
2. JUnit 5, PMD, Checkstyle 플러그인을 `pom.xml`에 설정한다.
3. 코드 스멜 유형별 샘플 코드를 작성한다.
4. 각 샘플에 대한 JUnit 테스트를 작성한다.
5. `mvn test`를 실행하여 원본 코드가 모두 통과하는지 확인한다.
6. `samples.csv`에 샘플 정보를 기록한다.

### 7.2 원본 코드 지표 측정

각 샘플에 대해 리팩토링 전 지표를 측정하고 `results/metrics_before.csv`에 기록한다.

| 컬럼 | 설명 |
| --- | --- |
| sample_id | 샘플 ID |
| smell_type | 코드 스멜 유형 |
| loc_before | 리팩토링 전 LOC |
| method_count_before | 리팩토링 전 메서드 수 |
| complexity_before | 리팩토링 전 복잡도 |
| pmd_violations_before | PMD 위반 수 |
| checkstyle_violations_before | Checkstyle 위반 수 |
| duplication_before | 중복 코드 지표 |
| test_result_before | 원본 테스트 결과 |

### 7.3 LLM 리팩토링 수행

각 샘플마다 다음 순서로 수행한다.

1. 원본 Java 코드를 프롬프트에 삽입한다.
2. 모델에 리팩토링을 요청한다.
3. 응답 원문을 `results/raw_outputs/{model}/{sample_id}.md`에 저장한다.
4. 리팩토링된 Java 코드만 추출하여 실험 프로젝트의 별도 작업 디렉터리에 반영한다.
5. 형식 수정이 필요하면 수정 사유를 기록한다.

권장 결과 디렉터리 구조는 다음과 같다.

```text
results/refactored_sources/
  chatgpt/
    LM01/
      OrderProcessor.java
    LM02/
      ...
  claude/
    LM01/
      OrderProcessor.java
  gemini/
    LM01/
      OrderProcessor.java
```

### 7.4 리팩토링 결과 검증

각 결과에 대해 다음 검증을 수행한다.

1. 컴파일 확인: `mvn -DskipTests compile`
2. 테스트 확인: `mvn test`
3. 정적분석: `mvn pmd:pmd`, `mvn checkstyle:checkstyle`
4. 중복 검사: `mvn pmd:cpd`
5. 결과 기록: `results/metrics_after.csv`

검증 실패 시에도 결과를 버리지 말고 실패 유형을 기록한다. 실패한 결과가 오히려 LLM 리팩토링의 한계를 설명하는 중요한 사례가 될 수 있다.

## 8. 평가 지표

### 8.1 자동 평가 지표

| 지표 | 계산 방법 |
| --- | --- |
| 컴파일 가능률 | 컴파일 성공 결과 수 / 전체 결과 수 |
| 테스트 통과율 | JUnit 테스트 성공 결과 수 / 전체 결과 수 |
| 리팩토링 성공률 | 성공 기준 만족 결과 수 / 전체 결과 수 |
| 복잡도 변화 | complexity_after - complexity_before |
| 정적분석 위반 변화 | pmd_violations_after - pmd_violations_before |
| 코드 크기 변화 | loc_after - loc_before |
| 중복도 변화 | duplication_after - duplication_before |

### 8.2 인간 평가 루브릭

각 항목은 1~5점으로 평가한다.

| 항목 | 1점 | 3점 | 5점 |
| --- | --- | --- | --- |
| 코드 스멜 제거 | 거의 제거되지 않음 | 일부 개선됨 | 명확히 개선됨 |
| 행동 보존 신뢰도 | 의미 변경 가능성이 큼 | 대체로 유지됨 | 의미 보존이 명확함 |
| 가독성 | 읽기 어려움 | 보통 | 명확하고 이해하기 쉬움 |
| 설계 적절성 | 부적절한 구조 변경 | 일부 적절함 | 책임 분리가 적절함 |
| 변경 최소성 | 불필요한 변경이 많음 | 일부 불필요한 변경 있음 | 필요한 변경만 수행 |

### 8.3 최종 성공 기준

다음 조건을 모두 만족하면 해당 결과를 성공으로 판단한다.

```text
성공 = 컴파일 성공
    + JUnit 테스트 전체 통과
    + 코드 스멜 제거 점수 4점 이상
    + 가독성 점수 4점 이상
```

## 9. 실패 유형 분류

실패 결과는 다음 기준으로 분류한다.

| 실패 유형 | 판단 기준 |
| --- | --- |
| Compile Error | 문법 오류, 타입 오류, import 오류 등으로 컴파일 실패 |
| Test Failure | 컴파일은 되지만 JUnit 테스트 실패 |
| API Breakage | public 클래스명, 메서드명, 인자, 반환 형식 변경 |
| Hallucination | 존재하지 않는 라이브러리, 클래스, 메서드 사용 |
| Over-refactoring | 요구하지 않은 구조 변경, 과도한 클래스 분리, 불필요한 패턴 적용 |
| Under-refactoring | 코드 스멜이 거의 개선되지 않음 |
| Style-only Change | 이름이나 포맷만 바꾸고 구조적 개선이 없음 |

하나의 결과가 여러 실패 유형에 해당할 수 있다. 이 경우 가장 직접적인 실패 원인을 `primary_failure_type`으로 기록하고, 나머지는 `secondary_failure_type`에 기록한다.

## 10. 결과 기록 파일 양식

### 10.1 `metrics_after.csv`

| 컬럼 | 설명 |
| --- | --- |
| sample_id | 샘플 ID |
| model | 모델명 |
| prompt_type | basic 또는 constrained |
| compile_success | true/false |
| test_success | true/false |
| loc_after | 리팩토링 후 LOC |
| method_count_after | 리팩토링 후 메서드 수 |
| complexity_after | 리팩토링 후 복잡도 |
| pmd_violations_after | PMD 위반 수 |
| checkstyle_violations_after | Checkstyle 위반 수 |
| duplication_after | 중복 지표 |
| primary_failure_type | 대표 실패 유형 |
| manual_fix_note | 형식 수정 내용 |

### 10.2 `human_evaluation.csv`

| 컬럼 | 설명 |
| --- | --- |
| sample_id | 샘플 ID |
| model | 모델명 |
| prompt_type | basic 또는 constrained |
| smell_removal_score | 코드 스멜 제거 점수 |
| behavior_preservation_score | 행동 보존 신뢰도 점수 |
| readability_score | 가독성 점수 |
| design_score | 설계 적절성 점수 |
| minimal_change_score | 변경 최소성 점수 |
| success | 최종 성공 여부 |
| comment | 평가 메모 |

### 10.3 `final_summary.csv`

| 컬럼 | 설명 |
| --- | --- |
| model | 모델명 |
| prompt_type | 프롬프트 조건 |
| smell_type | 코드 스멜 유형 |
| total_cases | 전체 케이스 수 |
| compile_success_rate | 컴파일 가능률 |
| test_success_rate | 테스트 통과율 |
| refactoring_success_rate | 리팩토링 성공률 |
| avg_complexity_delta | 평균 복잡도 변화 |
| avg_pmd_violation_delta | 평균 PMD 위반 변화 |
| avg_readability_score | 평균 가독성 점수 |

## 11. 분석 방법

### 11.1 모델별 분석

모델별로 컴파일 가능률, 테스트 통과율, 리팩토링 성공률을 비교한다. 이 분석은 어떤 모델이 Java 리팩토링을 더 안정적으로 수행하는지 보여준다.

### 11.2 코드 스멜 유형별 분석

Long Method, Duplicated Code, Large Class/Data Class 별로 성공률과 실패 유형을 비교한다. 이를 통해 어떤 코드 스멜이 LLM 에게 상대적으로 쉬운지 또는 어려운지 확인한다.

### 11.3 프롬프트 조건별 분석

기본 프롬프트와 제약 프롬프트의 결과를 비교한다. 제약 프롬프트가 API 파괴, 기능 변경, 환각을 줄이는지 확인한다.

### 11.4 대표 사례 분석

최종 보고서에는 다음 사례를 포함한다.

1. 가장 성공적인 리팩토링 사례 1개
2. 컴파일은 되지만 테스트가 실패한 사례 1개
3. 코드 스멜이 거의 제거되지 않은 사례 1개
4. 가능하면 모델별 차이가 뚜렷한 사례 1개

각 사례는 원본 코드 일부, LLM 리팩토링 결과 일부, 테스트/정적분석 결과, 인간 평가 메모를 함께 제시한다.

## 12. 최종 보고서 반영 계획

실험 후 최종 보고서에는 다음 내용을 추가한다.

1. 실제 사용한 모델명과 실험 날짜
2. 데이터셋 구성표
3. 모델별 전체 결과표
4. 코드 스멜 유형별 결과표
5. 프롬프트 조건별 결과표
6. 대표 성공 사례와 실패 사례
7. LLM 기반 자동 리팩토링의 실무 적용 가능성 논의
8. 연구 한계와 후속 연구 방향

## 13. 실험 일정

| 기간 | 작업 |
| --- | --- |
| 1일차 | Maven 프로젝트 생성, JUnit/PMD/Checkstyle 설정 |
| 2~3일차 | Long Method 샘플 10개와 테스트 작성 |
| 4~5일차 | Duplicated Code 샘플 10개와 테스트 작성 |
| 6일차 | Large Class/Data Class 샘플 작성 여부 결정 |
| 7일차 | 원본 코드 테스트 및 정적 지표 측정 |
| 8~9일차 | LLM 모델별 리팩토링 수행 |
| 10일차 | 컴파일, 테스트, 정적분석 결과 수집 |
| 11일차 | 인간 평가 루브릭 적용 |
| 12일차 | 결과표 작성 및 실패 유형 분석 |
| 13일차 | 최종 보고서 결과/분석 장 반영 |

## 14. 체크리스트

### 실험 전

- [ ] Java 버전 결정
- [ ] Maven 프로젝트 생성
- [ ] JUnit 5 설정
- [ ] PMD 설정
- [ ] Checkstyle 설정
- [ ] 샘플 ID 규칙 결정
- [ ] `samples.csv` 작성
- [ ] 모든 원본 코드가 `mvn test` 통과

### LLM 실행 중

- [ ] 모든 모델에 동일한 프롬프트 사용
- [ ] 원본 응답 저장
- [ ] 리팩토링 코드 저장
- [ ] 형식 수정 여부 기록
- [ ] 재시도 여부 기록

### 실험 후

- [ ] 컴파일 성공 여부 기록
- [ ] 테스트 성공 여부 기록
- [ ] 정적분석 결과 기록
- [ ] 인간 평가 점수 기록
- [ ] 실패 유형 분류
- [ ] 모델별 요약표 작성
- [ ] 대표 사례 선정
