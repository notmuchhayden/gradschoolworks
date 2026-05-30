# ARX 기반 스트림 암호의 멀티미디어 데이터 암호화 특성 비교

## Salsa20과 ChaCha20의 사진 및 영상 암호화 실험 최종 보고서

### 요약
본 프로젝트는 ARX(Addition-Rotation-XOR) 기반 스트림 암호인 Salsa20과 ChaCha20을 사진 및 영상형 데이터에 적용하고, 처리 성능, 키스트림 확산, nonce 재사용 오용 시 누출 특성을 비교한다. 실험은 Python 표준 라이브러리만 사용한 동일 구현 환경에서 Salsa20/8, Salsa20/12, Salsa20/20, ChaCha8, ChaCha12, ChaCha20을 대상으로 수행하도록 구성하였다. 성능은 MiB/s 처리량으로 측정하고, 확산은 키 또는 nonce 한 비트 변경 후 생성 키스트림의 블록별 Hamming distance로 평가한다. 또한 같은 키와 nonce를 재사용하면 두 암호문의 XOR가 두 평문의 XOR와 같아지는 스트림 암호의 구조적 위험을 이미지형 데이터 쌍으로 확인한다.

## Ⅰ. 서론
스트림 암호는 키와 nonce로부터 생성한 키스트림을 평문과 XOR하여 암호문을 만든다. 이 방식은 대용량 데이터를 순차 처리하기 쉽기 때문에 사진, 영상, 로그, 네트워크 패킷처럼 길이가 큰 데이터의 기밀성 제공에 적합하다. 그러나 같은 키스트림을 두 번 사용하면 `C1 XOR C2 = P1 XOR P2` 관계가 그대로 성립하므로 nonce 관리가 잘못될 때 치명적인 정보 누출이 발생한다.

본 연구는 Salsa20과 ChaCha20의 구조 차이가 멀티미디어 데이터 암호화 실험에서 어떻게 나타나는지 비교한다. 단순히 암호화와 복호화가 성공하는지를 확인하는 데서 그치지 않고, 라운드 수 변화에 따른 처리량, 키와 nonce 변화에 대한 키스트림 반응, nonce 재사용 오용 시 관찰 가능한 누출을 함께 다룬다.

연구 질문은 다음 세 가지이다.

1. 동일 구현 조건에서 Salsa20과 ChaCha20의 처리량은 라운드 수와 입력 데이터 종류에 따라 어떻게 달라지는가.
2. 키 또는 nonce 한 비트 변경은 생성 키스트림에 어느 정도의 Hamming distance를 유발하는가.
3. 동일 key/nonce 재사용 시 사진형 데이터에서 평문 간 구조 관계가 암호문 조합을 통해 어떻게 드러나는가.

## Ⅱ. 관련 연구
Salsa20은 Bernstein이 제안한 ARX 기반 스트림 암호로, 512비트 내부 상태에 대해 quarter-round, row-round, column-round를 반복 적용한다. ChaCha는 Salsa20의 변형으로 quarter-round 연산 순서와 diagonal round 구성을 조정하여 라운드당 확산을 개선하려는 목적을 가진다. 두 알고리즘은 S-box나 테이블 조회에 의존하지 않고 덧셈, 회전, XOR만 사용하므로 소프트웨어 구현이 간결하고 일정한 연산 경로를 만들기 쉽다.

ChaCha20은 Poly1305와 결합되어 TLS, QUIC 등 현대 프로토콜에서 인증 암호 형태로 널리 쓰인다. 다만 본 실험의 범위는 스트림 암호 키스트림과 XOR 암호화 경로의 비교이다. 따라서 무결성, 인증, 재전송 공격 방어는 ChaCha20-Poly1305와 같은 AEAD 모드가 담당해야 할 별도 보안 목표로 분리한다.

통계적 무작위성 지표는 구현 이상 여부를 점검하는 보조 자료로 사용할 수 있다. 그러나 바이트 엔트로피나 Hamming distance가 양호하다는 사실만으로 암호학적 안전성이 증명되지는 않는다. 본 보고서의 확산 실험은 알고리즘 증명이 아니라 구현 비교와 오용 위험 설명을 위한 관찰 실험이다.

## Ⅲ. 실험 설계

### 3.1 대상 알고리즘
실험 대상은 `chacha8`, `chacha12`, `chacha20`, `salsa8`, `salsa12`, `salsa20`이다. 두 알고리즘 모두 256비트 키를 사용한다. ChaCha 계열은 RFC 8439 방식의 96비트 nonce와 32비트 block counter를 사용하고, Salsa20 계열은 64비트 nonce와 64비트 block counter를 사용한다.

### 3.2 데이터셋
실험 스크립트는 외부 파일 의존성을 줄이기 위해 다음 샘플을 결정적으로 생성한다.

| 파일 | 설명 |
|---|---|
| `gradient_128x128.ppm` | RGB gradient 원시 이미지 |
| `pattern_128x128.ppm` | 반복 패턴이 포함된 RGB 원시 이미지 |
| `gradient_128x128.bmp` | BMP 헤더와 row padding을 포함한 이미지 |
| `pattern_128x128.bmp` | 반복 패턴이 포함된 BMP 이미지 |
| `video_like_64x36x12.rgb` | 12프레임의 raw video 유사 바이트열 |
| `random_128KiB.bin` | 비교용 결정적 난수 바이트열 |

압축 이미지 대신 PPM, BMP, raw RGB 형태를 포함한 이유는 평문 구조가 파일 내부에 비교적 직접적으로 남아 nonce 재사용 오용의 영향을 설명하기 쉽기 때문이다. 실제 JPEG, PNG, MP4는 압축과 컨테이너 구조의 영향이 커서 별도 실험으로 분리하는 것이 적절하다.

### 3.3 정확성 검증
정확성은 다음 방법으로 확인한다.

1. ChaCha20 block function은 RFC 8439 block test vector와 대조한다.
2. 모든 샘플에 대해 `decrypt(encrypt(P)) = P`가 성립하는지 확인한다.
3. 평문과 암호문의 SHA-256, 암호문 엔트로피를 기록한다.

### 3.4 성능 실험
성능은 각 샘플에 대해 암호화와 복호화를 반복 수행한 뒤 1회 암호화에 해당하는 평균 시간을 계산한다. 보고 지표는 평균 시간, 표준편차, 처리량(MiB/s)이다. 본 구현은 교육용 순수 Python 구현이므로 절대 성능을 라이브러리 최적화 구현과 비교하지 않는다. 대신 동일 코드 조건에서 알고리즘과 라운드 수 간 상대 차이를 관찰한다.

실행 명령은 다음과 같다.

```bash
python3 run_arx_experiments.py --repeats 2
```

### 3.5 확산 실험
확산 실험은 기준 키스트림 4096바이트를 만들고, 키 또는 nonce의 한 비트를 변경한 뒤 동일 길이의 키스트림을 다시 생성한다. 이후 64바이트 블록 단위 Hamming distance를 측정한다. 이상적인 무작위 독립 바이트열이라면 비트 차이 비율은 평균적으로 0.5 근처에 위치한다.

변경 조건은 다음 네 가지이다.

| 조건 | 설명 |
|---|---|
| `key_bit_0` | key 첫 번째 비트 변경 |
| `key_bit_127` | key 중간 영역 비트 변경 |
| `nonce_bit_0` | nonce 첫 번째 비트 변경 |
| `nonce_last_bit` | nonce 마지막 비트 변경 |

### 3.6 Nonce 재사용 오용 실험
같은 key와 nonce로 서로 다른 이미지형 평문 두 개를 암호화하면 다음 관계가 성립한다.

```text
C1 = P1 XOR KS
C2 = P2 XOR KS
C1 XOR C2 = P1 XOR P2
```

실험은 PPM 쌍과 BMP 쌍을 대상으로 이 관계가 실제로 성립하는지 확인하고, `P1 XOR P2`와 `C1 XOR C2`의 해시와 엔트로피를 기록한다. 이 결과는 Salsa20 또는 ChaCha20을 정상 사용했을 때의 취약점이 아니라, 스트림 암호에서 고유 nonce 사용 규칙을 위반했을 때 발생하는 오용 위험이다.

## Ⅳ. 실험 결과
실험은 `python3 run_arx_experiments.py --repeats 1`로 1회 스모크 측정을 수행하였다. 실행 환경은 Python 3.12.3, Linux 6.8.0-117-generic x86_64, CPU 8개로 기록되었다. 반복 횟수가 1회이므로 아래 성능 수치는 최종 제출 전 5회 이상 반복 측정으로 갱신하는 것이 바람직하다.

### 4.1 정확성 검증
ChaCha20 block function은 RFC 8439 test vector와 일치하였다. 또한 모든 샘플에서 Salsa20과 ChaCha20 모두 `decrypt(encrypt(P)) = P`가 성립하였다. 각 평문과 암호문의 SHA-256 및 암호문 엔트로피는 `results/correctness.csv`에 저장하였다.

### 4.2 성능 결과
순수 Python 구현에서 측정한 알고리즘별 평균 처리량은 다음과 같다.

| 알고리즘 | 평균 처리량(MiB/s) |
|---|---:|
| ChaCha8 | 0.89 |
| ChaCha12 | 0.64 |
| ChaCha20 | 0.40 |
| Salsa8 | 0.74 |
| Salsa12 | 0.59 |
| Salsa20 | 0.40 |

라운드 수가 낮을수록 처리량이 높아지는 경향이 확인되었다. 20라운드 조건에서는 ChaCha20과 Salsa20이 모두 약 0.40 MiB/s로 유사하게 측정되었다. 8라운드 조건에서는 ChaCha8이 Salsa8보다 높게 측정되었지만, 본 구현은 교육용 Python 코드이므로 이 차이를 최적화 구현의 일반적인 성능 우위로 해석해서는 안 된다.

### 4.3 확산 결과
키 또는 nonce 한 비트 변경 후 4096바이트 키스트림의 블록별 Hamming ratio 평균은 대부분 0.5 근처에 위치하였다.

| 알고리즘 | key bit 0 | key bit 127 | nonce bit 0 | nonce last bit |
|---|---:|---:|---:|---:|
| ChaCha8 | 0.496033 | 0.498901 | 0.495544 | 0.495911 |
| ChaCha12 | 0.500854 | 0.500275 | 0.500946 | 0.496521 |
| ChaCha20 | 0.497040 | 0.500153 | 0.499481 | 0.499908 |
| Salsa8 | 0.502838 | 0.500702 | 0.500671 | 0.499390 |
| Salsa12 | 0.500671 | 0.497803 | 0.500366 | 0.496857 |
| Salsa20 | 0.503998 | 0.498596 | 0.498169 | 0.503235 |

이 결과는 실험한 조건에서 키와 nonce의 작은 변화가 출력 키스트림에 넓게 반영되었음을 보여준다. 다만 Hamming ratio가 0.5에 가깝다는 사실은 통계적 관찰일 뿐이며, 감소 라운드 변형의 실제 보안성을 보장하지 않는다.

### 4.4 Nonce 재사용 결과
PPM 이미지 쌍과 BMP 이미지 쌍에서 같은 key/nonce를 재사용했을 때 두 알고리즘 모두 `C1 XOR C2 = P1 XOR P2` 관계가 참으로 확인되었다.

| 알고리즘 | 데이터 쌍 | 비교 바이트 | XOR 관계 |
|---|---|---:|---|
| ChaCha20 | gradient/pattern PPM | 49,167 | True |
| ChaCha20 | gradient/pattern BMP | 49,206 | True |
| Salsa20 | gradient/pattern PPM | 49,167 | True |
| Salsa20 | gradient/pattern BMP | 49,206 | True |

특히 BMP 쌍의 XOR 엔트로피는 7.277637로 PPM 쌍의 7.996170보다 낮게 나타났다. 이는 BMP 헤더, padding, 생성 이미지 패턴처럼 구조가 더 강한 바이트 영역에서는 XOR 결과에도 구조적 편향이 남을 수 있음을 시사한다.

### 4.5 결과 파일
실험 결과는 `results/` 디렉터리에 CSV로 저장된다.

| 파일 | 내용 |
|---|---|
| `environment.csv` | Python, OS, CPU 개수 |
| `correctness.csv` | 복호화 일치, 해시, 암호문 엔트로피 |
| `performance.csv` | 샘플별 평균 시간과 처리량 |
| `diffusion.csv` | 블록별 Hamming distance |
| `nonce_reuse.csv` | nonce 재사용 XOR 관계 확인 |
| `summary_throughput.csv` | 알고리즘별 평균 처리량 요약 |
| `summary_diffusion.csv` | 알고리즘/변경조건별 평균 Hamming ratio 요약 |

## Ⅴ. 논의
예상되는 성능 결과는 라운드 수가 낮은 `8`, `12` 변형이 `20`라운드보다 빠르게 나타나는 것이다. 다만 순수 Python 구현에서는 함수 호출, 바이트열 생성, XOR 루프 비용이 함께 섞이므로 실제 C 또는 SIMD 최적화 구현의 결과와 동일하게 일반화할 수 없다. 따라서 본 실험의 성능 결론은 "동일 교육용 구현 내 상대 비교"로 제한한다.

확산 실험에서 각 블록의 Hamming ratio가 0.5에 가까우면 키 또는 nonce의 작은 변화가 출력 키스트림 전반에 충분히 반영된 것으로 해석할 수 있다. 그러나 이 지표는 안전성 증명이 아니며, 감소 라운드 변형을 실제 보안 목적으로 사용해도 된다는 결론으로 이어지지 않는다.

Nonce 재사용 실험은 스트림 암호 운용 규칙의 중요성을 보여준다. 같은 key/nonce에서 생성된 키스트림이 반복되면 공격자는 두 암호문만 XOR해도 두 평문의 XOR를 얻는다. 원시 이미지나 반복 패턴 데이터에서는 이 값이 구조 정보를 포함할 수 있고, 알려진 평문 일부가 있으면 다른 평문 일부를 복구하는 단서가 된다. 따라서 실제 시스템에서는 nonce 고유성 보장, 인증 암호 사용, 키 관리 정책이 알고리즘 선택만큼 중요하다.

## Ⅵ. 결론
본 프로젝트는 Salsa20과 ChaCha20을 멀티미디어형 데이터에 적용하는 실험 환경을 구성하고, 성능, 확산, nonce 재사용 오용을 비교할 수 있는 재현 가능한 절차를 제시하였다. 최종 분석의 핵심은 ChaCha와 Salsa의 ARX 구조 차이 자체뿐 아니라, 라운드 수, 구현 언어, 입력 데이터 구조, nonce 관리 방식이 결과 해석에 함께 영향을 준다는 점이다.

실제 제출 전에는 `run_arx_experiments.py` 실행 결과를 본문 표와 그래프로 반영하고, 사용 장비의 CPU 정보, 반복 횟수, Python 버전, 샘플 크기를 명시해야 한다. 또한 본 실험이 기밀성 중심의 스트림 암호 비교이며 무결성 보장은 다루지 않는다는 한계를 분명히 적어야 한다.

## 참고 문헌
[1] D. J. Bernstein, "The Salsa20 family of stream ciphers," 2007.

[2] D. J. Bernstein, "ChaCha, a variant of Salsa20," 2008.

[3] Y. Nir and A. Langley, "ChaCha20 and Poly1305 for IETF Protocols," RFC 8439, 2018.

[4] NIST, "A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications," SP 800-22 Rev. 1a, 2010.
