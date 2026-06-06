# ARX 스트림 암호 멀티미디어 암호화/복호화 실험 계획

## 1. 실험 개요

본 실험은 ARX(Add-Rotate-Xor) 구조를 사용하는 대표적인 스트림 암호인 Salsa20과 ChaCha20을 C++로 직접 구현하고, 동일한 입력 데이터에 대해 암호화 및 복호화 성능을 비교하는 것을 목표로 한다. 실험에서는 두 알고리즘을 각각 독립된 헤더 파일과 구현 파일로 분리하고, 별도의 테스트 실행 파일인 `TestMain.cpp`에서 정확성 검증과 성능 측정을 수행한다.

실험 대상 알고리즘은 구현 복잡도를 줄이고 두 구조의 차이를 명확히 비교하기 위해 Salsa20/8과 ChaCha8을 기준으로 한다. 두 알고리즘 모두 256비트 키, 64비트 논스, 64바이트 키스트림 블록을 사용하며, 생성된 키스트림과 입력 데이터를 XOR하여 암호화와 복호화를 수행한다.

## 2. 실험 목표

1. Salsa20과 ChaCha20의 핵심 라운드 구조를 C++ 코드로 구현한다.
2. 두 알고리즘의 파일 구조를 분리하여 각 구현을 독립적으로 테스트할 수 있게 한다.
3. BMP 이미지 파일을 입력 데이터로 사용하여 암호화와 복호화를 수행한다.
4. 복호화 결과가 원본 BMP 파일과 완전히 동일한지 검증한다.
5. 동일한 조건에서 Salsa20/8과 ChaCha8의 암호화 및 복호화 시간을 측정한다.
6. 측정 결과를 바탕으로 두 알고리즘의 구현상 차이가 성능에 미치는 영향을 분석한다.

## 3. 구현 파일 구성

실험에 사용할 C++ 파일은 다음과 같이 구성한다.

| 파일명 | 역할 |
| --- | --- |
| `Salsa20.h` | Salsa20 클래스 또는 함수 선언, 키/논스 설정 함수, 암호화/복호화 인터페이스 선언 |
| `Salsa20.cpp` | Salsa20 quarter-round, row/column round, block function, 키스트림 생성, XOR 처리 구현 |
| `ChaCha20.h` | ChaCha20 클래스 또는 함수 선언, 키/논스 설정 함수, 암호화/복호화 인터페이스 선언 |
| `ChaCha20.cpp` | ChaCha quarter-round, column/diagonal round, block function, 키스트림 생성, XOR 처리 구현 |
| `TestMain.cpp` | 테스트 데이터 로드, 암호화/복호화 실행, 결과 파일 저장, 정확성 검증, 시간 측정 수행 |

## 4. 구현 세부 계획

### 4.1 Salsa20 구현

`Salsa20.h`와 `Salsa20.cpp`에는 Salsa20/8 구현을 작성한다. 내부 상태는 16개의 32비트 워드로 구성하며, 상수, 256비트 키, 64비트 논스, 64비트 블록 카운터를 배치한다. 라운드 함수는 Salsa20 논문에서 제시한 ARX 구조를 따른다.

구현해야 할 주요 기능은 다음과 같다.

1. 32비트 왼쪽 회전 함수 구현
2. Salsa20 quarter-round 구현
3. column round와 row round 구현
4. 8라운드 block function 구현
5. 64바이트 키스트림 블록 생성
6. 임의 길이 바이트 배열에 대한 XOR 기반 암호화/복호화 함수 구현

### 4.2 ChaCha20 구현

`ChaCha20.h`와 `ChaCha20.cpp`에는 ChaCha8 구현을 작성한다. ChaCha는 Salsa20과 동일하게 ARX 연산을 사용하지만, quarter-round의 연산 순서와 내부 상태 배치가 다르다. 구현에서는 상수, 256비트 키, 64비트 카운터/논스 입력을 4 x 4 상태 배열에 배치하고, column round와 diagonal round를 반복 수행한다.

구현해야 할 주요 기능은 다음과 같다.

1. 32비트 왼쪽 회전 함수 구현
2. ChaCha quarter-round 구현
3. column round와 diagonal round 구현
4. 8라운드 block function 구현
5. 64바이트 키스트림 블록 생성
6. 임의 길이 바이트 배열에 대한 XOR 기반 암호화/복호화 함수 구현

## 5. 테스트 시나리오

### 5.1 테스트 입력 준비

테스트 입력으로 `data_samples` 디렉터리에 BMP 이미지 파일을 준비한다. BMP 파일은 압축되지 않은 바이너리 데이터이므로 암호화 전후의 바이트 변화를 확인하기 쉽고, 복호화 결과가 원본과 동일한지 파일 단위로 비교하기에 적합하다.

입력 파일 예시는 다음과 같다.

```text
Cryptography_Security/data_samples/input.bmp
```

### 5.2 TestMain.cpp 실행 흐름

`TestMain.cpp`는 다음 순서로 테스트를 수행한다.

1. 테스트용 BMP 파일을 바이너리 모드로 읽어 `std::vector<uint8_t>`에 저장한다.
2. 고정된 256비트 키와 64비트 논스를 설정한다.
3. Salsa20 객체 또는 함수를 초기화한다.
4. Salsa20로 원본 데이터를 암호화하고 암호화 결과를 `results/salsa20_encrypted.bmp`에 저장한다.
5. Salsa20 암호문을 다시 복호화하고 복호화 결과를 `results/salsa20_decrypted.bmp`에 저장한다.
6. 원본 데이터와 Salsa20 복호화 결과를 바이트 단위로 비교한다.
7. ChaCha20 객체 또는 함수를 초기화한다.
8. ChaCha20으로 원본 데이터를 암호화하고 암호화 결과를 `results/chacha20_encrypted.bmp`에 저장한다.
9. ChaCha20 암호문을 다시 복호화하고 복호화 결과를 `results/chacha20_decrypted.bmp`에 저장한다.
10. 원본 데이터와 ChaCha20 복호화 결과를 바이트 단위로 비교한다.
11. 각 알고리즘의 암호화 시간과 복호화 시간을 `std::chrono`로 측정한다.
12. 측정 결과와 검증 성공 여부를 콘솔에 출력한다.

### 5.3 정확성 검증 기준

정확성 검증은 다음 조건을 모두 만족해야 성공으로 판단한다.

1. Salsa20 복호화 결과가 원본 BMP 데이터와 바이트 단위로 완전히 동일해야 한다.
2. ChaCha20 복호화 결과가 원본 BMP 데이터와 바이트 단위로 완전히 동일해야 한다.
3. 암호화 결과는 원본 BMP 데이터와 달라야 한다.
4. 동일한 키와 논스를 사용할 경우 같은 알고리즘의 암호화 결과는 재실행 시 동일해야 한다.
5. 서로 다른 알고리즘인 Salsa20과 ChaCha20의 암호문은 일반적으로 서로 달라야 한다.

### 5.4 성능 측정 기준

성능 측정은 암호화와 복호화를 분리하여 수행한다. 단일 실행의 측정 오차를 줄이기 위해 동일한 입력 파일에 대해 각 알고리즘을 여러 번 반복 실행하고 평균 시간을 계산한다.

측정 항목은 다음과 같다.

| 항목 | 설명 |
| --- | --- |
| Salsa20 암호화 시간 | 원본 BMP 데이터를 Salsa20로 암호화하는 데 걸린 시간 |
| Salsa20 복호화 시간 | Salsa20 암호문을 복호화하는 데 걸린 시간 |
| ChaCha20 암호화 시간 | 원본 BMP 데이터를 ChaCha20으로 암호화하는 데 걸린 시간 |
| ChaCha20 복호화 시간 | ChaCha20 암호문을 복호화하는 데 걸린 시간 |
| 처리량 | 입력 파일 크기를 실행 시간으로 나누어 MB/s 단위로 계산한 값 |

## 6. 빌드 및 실행 계획

C++17 이상을 기준으로 컴파일한다. 예시 빌드 명령은 다음과 같다.

```bash
g++ -std=c++17 -O2 TestMain.cpp Salsa20.cpp ChaCha20.cpp -o arx_stream_test
```

실행 명령은 다음과 같다.

```bash
./arx_stream_test
```

실행 후 콘솔에는 다음과 같은 정보가 출력되도록 한다.

```text
Input file: data_samples/input.bmp
Input size: 000000 bytes

[Salsa20/8]
Encryption time: 0.000 ms
Decryption time: 0.000 ms
Verification: PASS

[ChaCha8]
Encryption time: 0.000 ms
Decryption time: 0.000 ms
Verification: PASS
```

## 7. 결과 파일 계획

테스트 실행 후 `results` 디렉터리에 다음 파일을 생성한다.

| 파일명 | 설명 |
| --- | --- |
| `salsa20_encrypted.bmp` | Salsa20으로 암호화된 BMP 데이터 |
| `salsa20_decrypted.bmp` | Salsa20 암호문을 복호화한 BMP 데이터 |
| `chacha20_encrypted.bmp` | ChaCha20으로 암호화된 BMP 데이터 |
| `chacha20_decrypted.bmp` | ChaCha20 암호문을 복호화한 BMP 데이터 |

암호화된 BMP 파일은 정상 이미지로 표시되지 않아야 하며, 복호화된 BMP 파일은 원본 이미지와 동일하게 열려야 한다. 최종 보고서에는 원본 이미지, 암호화된 결과, 복호화된 결과, 실행 시간 표를 포함한다.

## 8. 예상 분석 방향

실험 결과 분석에서는 Salsa20과 ChaCha20의 성능 차이를 단순 실행 시간뿐 아니라 내부 구조 차이와 연결하여 설명한다. Salsa20은 column round와 row round를 통해 상태를 확산시키며, ChaCha20은 quarter-round에서 각 워드를 두 번 갱신하고 column/diagonal round를 수행하므로 라운드당 확산 특성이 다르다. 따라서 두 알고리즘의 연산 수가 유사하더라도 구현 방식, 메모리 접근 패턴, 컴파일러 최적화 결과에 따라 실행 시간이 달라질 수 있다.

최종적으로 본 실험은 두 알고리즘이 모두 XOR 기반 스트림 암호로서 복호화 정확성을 만족하는지 확인하고, 동일한 입력 조건에서 ChaCha20의 구조적 개선이 실제 C++ 구현 성능에 어떤 영향을 주는지 비교하는 데 초점을 둔다.
