# 알고리즘 과제 빌드 및 실행

Windows PowerShell과 Visual Studio 2022 기준이다. 모든 명령은 프로젝트 루트(`D:\Works\gradschoolworks`)에서 실행한다.

## 준비

- CMake 3.20 이상
- Visual Studio 2022 또는 Build Tools 2022의 **C++를 사용한 데스크톱 개발** 구성 요소(MSVC 및 Windows SDK)

프로젝트 루트로 이동한다.

```powershell
cd D:\Works\gradschoolworks
```

## HW1

### 빌드

CMake 프로젝트를 구성한 뒤 Release 모드로 빌드한다. 두 실행 파일이 함께 생성된다.

```powershell
cmake -S Algorithm/HW1 -B Algorithm/HW1/build -G "Visual Studio 17 2022" -A x64
cmake --build Algorithm/HW1/build --config Release
```

| 소스 | 실행 파일 | 기능 |
|---|---|---|
| [HW1_3_3.cpp](HW1/HW1_3_3.cpp) | `Algorithm/HW1/build/Release/HW1_3_3.exe` | 역전쌍 개수 계산 |
| [HW1_4_2.cpp](HW1/HW1_4_2.cpp) | `Algorithm/HW1/build/Release/HW1_4_2.exe` | 최대 상담 수와 선택한 상담 ID 출력 |

특정 프로그램만 빌드하려면 대상 이름을 지정한다.

```powershell
cmake --build Algorithm/HW1/build --config Release --target HW1_3_3
cmake --build Algorithm/HW1/build --config Release --target HW1_4_2
```

### 실행

역전쌍 프로그램:

```powershell
.\Algorithm\HW1\build\Release\HW1_3_3.exe
```

실행 후 원소 개수 `n`과 `n`개의 정수를 입력한다.

상담 선택 프로그램:

```powershell
.\Algorithm\HW1\build\Release\HW1_4_2.exe
```

실행 후 상담 개수 `n`과 각 상담의 `ID 시작시간 종료시간`을 입력한다.

## HW2

### 빌드

```powershell
cmake -S Algorithm/HW2 -B Algorithm/HW2/build -G "Visual Studio 17 2022" -A x64
cmake --build Algorithm/HW2/build --config Release
```

[HW2_2_3.cpp](HW2/HW2_2_3.cpp)와 [HW2_4_2.cpp](HW2/HW2_4_2.cpp)를 빌드하여
`Algorithm/HW2/build/Release/` 아래에 각각의 실행 파일을 생성한다.

### 실행

```powershell
.\Algorithm\HW2\build\Release\HW2_2_3.exe
```

첫 줄에 문자열 `A`, 둘째 줄에 문자열 `B`, 셋째 줄에 교체 비용 `c`를 입력한다.
문자열 길이는 각각 0~200, `c`는 1~100의 정수이다. 빈 문자열은 빈 줄로 입력한다.
삽입·삭제 비용은 1이며, 문자는 대소문자를 구분하지 않고 비교한다.
`-`는 편집 과정의 공백 표시로 사용하므로 입력 문자열에는 사용할 수 없다.

입력 예시:

```text
abc
Yabd
3
```

출력 예시:

```text
3
-ab-c
Yabd-
```

첫 줄은 최소 비용, 다음 두 줄은 `-`로 길이를 맞춘 최적 편집 과정 하나이다.
빈 편집 문자열도 빈 줄로 출력한다. 출력 전에 원래 문자열 복원, 양쪽 모두 `-`인 열의 부재,
열별 비용 합과 최소 비용의 일치를 검증한다.

### 해싱 실험 (HW2_4_2)

입력 파일이 있는 폴더로 이동한 뒤 실행한다. 네 파일에 선형 탐사와 이중 해싱을 각각 적용한다.

```powershell
Push-Location Algorithm/HW2
.\build\Release\HW2_4_2.exe
Pop-Location
```

Windows에서는 프로그램이 콘솔 출력 인코딩을 UTF-8로 설정하므로 한글 결과를 그대로 표시한다.
MSVC의 `/utf-8` 빌드 옵션과 함께 사용하며, 별도로 `chcp` 명령을 실행할 필요는 없다.

실행하면 전체 결과를 콘솔에 출력하고, 현재 작업 폴더의 `result.txt`에도 UTF-8로 저장한다.
위 실행 방법에서는 `Algorithm/HW2/result.txt`가 생성되며, 기존 파일이 있으면 덮어쓴다.

각 파일의 실제 삽입 키(성공 탐색과 같은 순서), 실패 질의 순서와 두 방식의 결과를 출력한다.
결과 항목은 삽입 평균, 성공 탐색 평균, 실패 탐색 평균 및 최대 탐사 횟수이다.
처음 확인한 칸과 마지막 빈칸도 탐사 횟수에 포함한다. 빈칸은 `-1`로 표시하여 키 `0`과 구분한다.

문제 주석의 이중 해싱 식에 있는 `m mod m`은 오기로 보고, 첨부 강의록의
`h1(k) = k mod m`을 적용했다. 두 번째 해시 함수는 문제에서 지정한 `1 + k mod (m - 1)`이다.

## 수정 후 다시 빌드

소스를 수정한 뒤에는 해당 과제의 빌드 명령을 다시 실행한다.

```powershell
cmake --build Algorithm/HW1/build --config Release
cmake --build Algorithm/HW2/build --config Release
```

두 프로젝트는 C++17을 사용하며, MSVC에서는 한글 소스를 읽기 위한 `/utf-8` 옵션을 적용한다.
