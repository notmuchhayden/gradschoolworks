# Salsa20/8 및 ChaCha8 C++ 실험

이 프로젝트는 Salsa20/8과 ChaCha8 스트림 암호를 C++17로 구현하고, BMP 파일의 암호화·복호화 정확성과 처리 성능을 비교한다.

## 요구 사항

- CMake 3.16 이상
- C++17을 지원하는 C++ 컴파일러
  - GCC
  - Clang
  - Microsoft Visual C++

설치 여부는 다음 명령으로 확인할 수 있다.

```bash
cmake --version
c++ --version
```

## 프로젝트 구성

```text
Cryptography_Security/
├── CMakeLists.txt
├── README.md
├── data_samples/
│   ├── gradient_512x512.bmp
│   └── pattern_512x512.bmp
├── results/
└── src/
    ├── Salsa20.h
    ├── Salsa20.cpp
    ├── ChaCha20.h
    ├── ChaCha20.cpp
    └── TestMain.cpp
```

파일명과 클래스명은 `Salsa20`, `ChaCha20`이지만 현재 실험의 라운드 수는 각각 8회이다. 따라서 실제 비교 대상은 Salsa20/8과 ChaCha8이다.

## 빌드

저장소 루트에서 프로젝트 디렉터리로 이동한다.

```bash
cd Cryptography_Security
```

Release 모드의 CMake 빌드 파일을 생성한다.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

프로그램을 빌드한다.

```bash
cmake --build build --parallel
```

빌드가 완료되면 다음 실행 파일이 생성된다.

```text
build/arx_stream_test
```

## 자동 테스트

CMake에 등록된 검증 테스트는 다음 명령으로 실행한다.

```bash
ctest --test-dir build --output-on-failure
```

테스트는 `gradient_512x512.bmp`를 대상으로 Salsa20/8과 ChaCha8의 암호화·복호화 및 정확성 검사를 수행한다. 성공하면 다음과 유사한 결과가 출력된다.

```text
100% tests passed, 0 tests failed out of 1
```

## 실험 실행

### 기본 실행

인자를 생략하면 `data_samples/gradient_512x512.bmp`를 사용하고, 각 암호화·복호화 연산을 30회 반복한다.

```bash
./build/arx_stream_test
```

### 입력 파일 지정

첫 번째 인자로 테스트할 파일 경로를 지정할 수 있다.

```bash
./build/arx_stream_test data_samples/pattern_512x512.bmp
```

BMP 파일을 기본 실험 대상으로 사용하지만 프로그램은 파일 전체를 바이트 배열로 처리하므로 다른 바이너리 파일도 입력할 수 있다.

### 반복 횟수 지정

두 번째 인자로 성능 측정 반복 횟수를 지정한다. 반복 횟수는 0보다 큰 정수여야 한다.

```bash
./build/arx_stream_test data_samples/gradient_512x512.bmp 100
```

명령 형식은 다음과 같다.

```text
./build/arx_stream_test [입력 파일] [반복 횟수]
```

## 검증 항목

실행 프로그램은 다음 조건을 검사한다.

1. 암호화한 데이터를 다시 처리했을 때 원본이 복원되는지 확인한다.
2. 암호문이 원본과 다른지 확인한다.
3. 같은 키, 논스, 카운터에서 같은 암호문이 생성되는지 확인한다.
4. 블록 카운터 변경 시 키스트림과 암호문이 달라지는지 확인한다.
5. Salsa20/8과 ChaCha8의 암호문이 서로 다른지 확인한다.
6. ChaCha8의 첫 키스트림 블록이 알려진 테스트 벡터와 일치하는지 확인한다.

모든 검사를 통과하면 마지막에 다음 결과가 출력된다.

```text
Overall result: PASS
```

## 성능 출력

각 알고리즘에 대해 다음 값이 출력된다.

- 평균 암호화 시간(ms)
- 암호화 처리량(MiB/s)
- 평균 복호화 시간(ms)
- 복호화 처리량(MiB/s)
- 각 정확성 검사 결과

성능 측정 구간에는 입력 파일을 읽거나 결과 파일을 저장하는 시간은 포함되지 않는다. 메모리에 적재된 데이터에 대한 키스트림 생성과 XOR 처리 시간만 측정한다.

## 결과 파일

실행 후 프로젝트의 `results` 디렉터리에 다음 파일이 생성된다.

```text
results/
├── salsa20_encrypted.bmp
├── salsa20_decrypted.bmp
├── chacha20_encrypted.bmp
└── chacha20_decrypted.bmp
```

암호화된 파일은 BMP 헤더까지 암호화되므로 일반 이미지 뷰어에서 정상 이미지로 열리지 않는다. 복호화된 파일은 원본과 바이트 단위로 동일해야 한다.

## 전체 재빌드

기존 빌드 파일을 삭제하고 처음부터 빌드하려면 다음 명령을 사용한다.

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
