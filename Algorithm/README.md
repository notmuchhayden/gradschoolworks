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

[HW2_2_3.cpp](HW2/HW2_2_3.cpp)를 빌드하여 `Algorithm/HW2/build/Release/HW2_2_3.exe`를 생성한다.

### 실행

```powershell
.\Algorithm\HW2\build\Release\HW2_2_3.exe
```

추가 입력이나 실행 옵션 없이 다음 5쌍을 교체 비용 `c=1`, `c=3`에서 각각 실행해 총 10개 결과를 출력한다.

| 검증 항목 | A | B |
|---|---|---|
| 수기 결과 대조 | `abc` | `Yabd` |
| 삽입만 필요 | 빈 문자열 | `Abc` |
| 삭제만 필요 | `abc` | 빈 문자열 |
| 변경 불필요 | `same` | `Same` |
| 순서가 다른 경우 | `ab` | `Ba` |

결과표에는 최소 비용, 최적 편집 과정, 열별 비용 합과 검증 결과가 표시된다.

## 수정 후 다시 빌드

소스를 수정한 뒤에는 해당 과제의 빌드 명령을 다시 실행한다.

```powershell
cmake --build Algorithm/HW1/build --config Release
cmake --build Algorithm/HW2/build --config Release
```

두 프로젝트는 C++17을 사용하며, MSVC에서는 한글 소스를 읽기 위한 `/utf-8` 옵션을 적용한다.
