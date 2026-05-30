# ARX 스트림 암호 멀티미디어 암호화 실험 계획

## 1. 실험 목적
본 실험의 목적은 Salsa20과 ChaCha20을 사진 및 영상형 데이터에 적용했을 때 다음 세 가지 특성이 어떻게 나타나는지 비교하는 것이다.

1. 동일 구현 조건에서 라운드 수와 알고리즘에 따른 처리 성능 차이를 측정한다.
2. 키 또는 nonce 한 비트 변경이 키스트림에 얼마나 넓게 반영되는지 Hamming distance로 관찰한다.
3. 스트림 암호에서 금지되는 nonce 재사용이 멀티미디어형 데이터에서 어떤 구조적 누출을 만드는지 확인한다.

이 실험은 암호 알고리즘의 새로운 안전성 증명이 아니라, ARX 기반 스트림 암호의 구조와 사용상 주의점을 이해하기 위한 재현 가능한 교육용 실험이다.

## 2. 연구 질문과 가설

### RQ1. 처리 성능
질문: Salsa20과 ChaCha20은 같은 Python 구현 조건에서 라운드 수에 따라 처리량이 어떻게 달라지는가.

가설: 8라운드, 12라운드, 20라운드 순으로 라운드 수가 증가할수록 처리량은 낮아진다. 같은 20라운드 조건에서는 두 알고리즘의 처리량이 큰 차이 없이 비슷하게 나타날 수 있다.

### RQ2. 키스트림 확산
질문: 키 또는 nonce의 한 비트만 바꾸면 생성 키스트림은 기준 키스트림과 얼마나 달라지는가.

가설: 충분한 라운드 조건에서는 블록별 Hamming ratio가 0.5 근처에 분포한다. 이는 두 출력이 비트 단위로 약 절반 정도 달라진다는 뜻이다.

### RQ3. Nonce 재사용 오용
질문: 같은 key와 nonce를 서로 다른 평문에 재사용하면 암호문 간 XOR에서 어떤 관계가 드러나는가.

가설: 스트림 암호의 XOR 구조 때문에 `C1 XOR C2 = P1 XOR P2`가 정확히 성립한다. 특히 BMP, PPM처럼 원시 구조가 강한 이미지형 데이터에서는 평문 간 구조 관계가 XOR 결과에 남을 수 있다.

## 3. 실험 범위

### 3.1 대상 알고리즘
실험 대상은 다음 6개 조합이다.

| 구분 | 라운드 |
|---|---|
| ChaCha | ChaCha8, ChaCha12, ChaCha20 |
| Salsa | Salsa8, Salsa12, Salsa20 |

ChaCha 계열은 256비트 key, 96비트 nonce, 32비트 counter를 사용한다. Salsa 계열은 256비트 key, 64비트 nonce, 64비트 counter를 사용한다.

### 3.2 구현 범위
구현 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `arx_streams.py` | Salsa20/ChaCha20 block function, keystream 생성, XOR 암복호화 |
| `run_arx_experiments.py` | 샘플 데이터 생성, 실험 실행, CSV 결과 저장 |
| `final_report.md` | 실험 결과를 보고서 형식으로 정리하는 최종 문서 |

외부 암호 라이브러리는 사용하지 않는다. Python 표준 라이브러리만 사용하여 구현을 읽고 재현하기 쉽게 한다. 단, 이 때문에 절대 성능은 최적화된 C/Rust/OpenSSL 구현보다 매우 낮게 나온다. 따라서 성능 결과는 같은 코드 안에서의 상대 비교로만 해석한다.

## 4. 데이터셋 계획

### 4.1 기본 샘플
스크립트는 `data_samples_small/`에 다음 샘플을 자동 생성한다.

| 파일 | 크기/형식 | 목적 |
|---|---|---|
| `gradient_128x128.ppm` | 128x128 RGB PPM | 원시 픽셀 gradient 구조 관찰 |
| `pattern_128x128.ppm` | 128x128 RGB PPM | 반복 패턴 구조 관찰 |
| `gradient_128x128.bmp` | 128x128 BMP | BMP 헤더와 row padding 포함 |
| `pattern_128x128.bmp` | 128x128 BMP | 반복 패턴과 BMP 구조 포함 |
| `video_like_64x36x12.rgb` | 64x36, 12프레임 raw RGB 유사 데이터 | 짧은 영상형 바이트열 |
| `random_128KiB.bin` | 128 KiB 결정적 난수 | 구조가 약한 비교군 |

### 4.2 작은 샘플을 쓰는 이유
순수 Python 구현은 느리기 때문에 초기 실험은 작은 샘플로 수행한다. 작은 샘플은 실험 절차 검증, CSV 구조 확인, 보고서 표 구성에 적합하다.

최종 제출 전에 시간이 허용되면 다음 확장 실험을 추가할 수 있다.

| 확장 방향 | 예시 |
|---|---|
| 이미지 크기 증가 | 512x512 PPM/BMP |
| 영상 데이터 증가 | 더 큰 해상도 또는 프레임 수 |
| 실제 파일 사용 | PNG, JPEG, MP4를 별도 그룹으로 추가 |
| 반복 횟수 증가 | `--repeats 5` 또는 `--repeats 10` |

압축 포맷인 PNG, JPEG, MP4는 압축 알고리즘과 컨테이너 구조의 영향이 크기 때문에 원시 PPM/BMP 결과와 분리해서 해석한다.

## 5. 실험 절차

### 5.1 전체 실행
기본 실행 명령은 다음과 같다.

```bash
python3 run_arx_experiments.py --repeats 2
```

빠른 확인만 할 때는 다음처럼 1회 반복으로 실행한다.

```bash
python3 run_arx_experiments.py --repeats 1
```

최종 보고서용 수치를 안정화하려면 최소 5회 이상 반복한다.

```bash
python3 run_arx_experiments.py --repeats 5
```

### 5.2 실행 순서
스크립트 내부 실행 순서는 다음과 같다.

1. 샘플 데이터가 없으면 `data_samples_small/`에 생성한다.
2. ChaCha20 RFC 8439 block test vector를 확인한다.
3. 모든 샘플에 대해 암호화 후 복호화가 원문과 같은지 확인한다.
4. 알고리즘별, 라운드별, 샘플별 처리 시간을 측정한다.
5. 키/nonce 한 비트 변경에 따른 키스트림 Hamming distance를 측정한다.
6. 동일 key/nonce 재사용 조건에서 `C1 XOR C2 = P1 XOR P2`가 성립하는지 확인한다.
7. 모든 결과를 `results/` 아래 CSV 파일로 저장한다.

## 6. 세부 실험 설계

### 6.1 정확성 검증
목적: 구현이 기본적으로 올바르게 동작하는지 확인한다.

검증 항목은 다음과 같다.

| 항목 | 판단 기준 |
|---|---|
| ChaCha20 test vector | RFC 8439 block output과 일치해야 함 |
| 암복호화 round-trip | `decrypt(encrypt(P)) == P`가 모든 샘플에서 참이어야 함 |
| 해시 기록 | 평문/암호문 SHA-256을 남겨 재현성 확인 |
| 암호문 엔트로피 | 구현 이상 여부를 보는 보조 지표 |

출력 파일: `results/correctness.csv`

주의할 점: 암호문 엔트로피가 높다는 사실만으로 보안성이 증명되지는 않는다. 이는 구현이 명백히 이상한 출력을 내지 않는지 확인하는 보조 지표이다.

### 6.2 성능 실험
목적: 알고리즘과 라운드 수에 따른 처리량 차이를 측정한다.

측정 대상은 다음 조합이다.

| 알고리즘 | 라운드 |
|---|---|
| ChaCha | 8, 12, 20 |
| Salsa | 8, 12, 20 |

각 조합에 대해 모든 샘플을 암호화하고 복호화한다. 시간 측정은 `time.perf_counter()`를 사용한다. 스크립트는 암호화와 복호화를 한 번씩 수행한 총 시간을 2로 나누어 1회 암호화에 해당하는 평균 시간으로 기록한다.

기록 지표는 다음과 같다.

| 지표 | 의미 |
|---|---|
| `mean_seconds` | 반복 실행 평균 시간 |
| `stdev_seconds` | 반복 실행 표준편차 |
| `throughput_MiB_s` | 초당 처리한 MiB |
| `bytes` | 입력 파일 크기 |
| `repeats` | 반복 횟수 |

출력 파일:

| 파일 | 내용 |
|---|---|
| `results/performance.csv` | 샘플별 상세 성능 |
| `results/summary_throughput.csv` | 알고리즘별 평균 처리량 |

해석 기준:

1. 같은 알고리즘에서는 8라운드가 20라운드보다 빨라야 자연스럽다.
2. 파일 크기가 작은 경우 Python 호출 비용과 바이트열 처리 오버헤드가 상대적으로 크게 반영된다.
3. 이 결과는 최적화 구현의 절대 성능 비교가 아니라 동일 교육용 구현 안에서의 상대 비교이다.

### 6.3 키스트림 확산 실험
목적: key 또는 nonce의 작은 변화가 출력 키스트림에 어느 정도 반영되는지 관찰한다.

절차는 다음과 같다.

1. 기준 key와 nonce로 4096바이트 키스트림을 생성한다.
2. key 또는 nonce의 특정 비트 하나만 뒤집는다.
3. 같은 길이의 키스트림을 다시 생성한다.
4. 64바이트 블록 단위로 두 키스트림의 Hamming distance를 계산한다.
5. 64바이트는 512비트이므로 `hamming_bits / 512`를 Hamming ratio로 기록한다.

변경 조건은 다음 네 가지이다.

| mutation | 의미 |
|---|---|
| `key_bit_0` | key 첫 번째 비트 변경 |
| `key_bit_127` | key 중간 영역 비트 변경 |
| `nonce_bit_0` | nonce 첫 번째 비트 변경 |
| `nonce_last_bit` | nonce 마지막 비트 변경 |

출력 파일:

| 파일 | 내용 |
|---|---|
| `results/diffusion.csv` | 블록별 Hamming distance |
| `results/summary_diffusion.csv` | 알고리즘/변경조건별 평균 Hamming ratio |

해석 기준:

1. Hamming ratio가 0.5에 가까우면 두 키스트림이 비트 단위로 약 절반 다르다는 뜻이다.
2. 특정 라운드나 특정 mutation에서 0.5에서 크게 벗어나면 추가 확인이 필요하다.
3. 이 지표는 확산 관찰 지표이지 암호학적 안전성 증명이 아니다.

### 6.4 Nonce 재사용 오용 실험
목적: 스트림 암호에서 같은 key/nonce를 재사용하면 왜 위험한지 직접 확인한다.

비교 대상 쌍은 다음과 같다.

| 쌍 | 설명 |
|---|---|
| `gradient_128x128.ppm` / `pattern_128x128.ppm` | 원시 PPM 이미지 2개 |
| `gradient_128x128.bmp` / `pattern_128x128.bmp` | BMP 이미지 2개 |

절차는 다음과 같다.

1. 같은 key와 같은 nonce로 두 평문 `P1`, `P2`를 각각 암호화한다.
2. 두 암호문 `C1`, `C2`를 XOR한다.
3. 두 평문 `P1`, `P2`도 XOR한다.
4. `C1 XOR C2`와 `P1 XOR P2`가 같은지 확인한다.
5. XOR 결과의 SHA-256과 Shannon entropy를 기록한다.

출력 파일: `results/nonce_reuse.csv`

해석 기준:

1. `cipher_xor_equals_plain_xor`는 반드시 `True`가 된다. 이것은 구현 오류가 아니라 스트림 암호 XOR 구조의 직접적인 결과이다.
2. BMP처럼 헤더, padding, 반복 구조가 있는 포맷은 XOR 결과에도 구조적 편향이 남을 수 있다.
3. 이 실험은 Salsa20/ChaCha20 자체의 정상 사용 취약점이 아니라 nonce 재사용이라는 사용 규칙 위반의 위험을 보여준다.

## 7. 결과 파일 목록

실험 후 `results/`에 다음 파일이 생성된다.

| 파일 | 역할 |
|---|---|
| `environment.csv` | Python 버전, OS, CPU 개수 |
| `correctness.csv` | 테스트 벡터, round-trip, 해시, 엔트로피 확인 |
| `performance.csv` | 알고리즘/라운드/샘플별 성능 상세 |
| `summary_throughput.csv` | 알고리즘별 평균 처리량 요약 |
| `diffusion.csv` | 블록별 Hamming distance 상세 |
| `summary_diffusion.csv` | 확산 결과 평균 요약 |
| `nonce_reuse.csv` | nonce 재사용 오용 실험 결과 |

## 8. 보고서 반영 계획

최종 보고서에는 다음 순서로 결과를 반영한다.

1. 실험 환경: `environment.csv`의 Python, OS, CPU 정보를 기재한다.
2. 정확성: ChaCha20 test vector 통과 여부와 모든 샘플 round-trip 성공 여부를 쓴다.
3. 성능: `summary_throughput.csv`를 표로 넣고, 라운드 수별 처리량 차이를 해석한다.
4. 확산: `summary_diffusion.csv`를 표로 넣고, Hamming ratio가 0.5 근처인지 확인한다.
5. Nonce 재사용: `nonce_reuse.csv`에서 XOR 관계가 성립했음을 보이고, 이것이 사용 규칙 위반의 위험임을 설명한다.
6. 한계: 순수 Python 구현, 작은 샘플, 반복 횟수, 압축 포맷 미포함 여부를 명시한다.

권장 그래프는 다음과 같다.

| 그래프 | 원본 CSV | 표현 |
|---|---|---|
| 라운드 수별 처리량 | `summary_throughput.csv` | 막대그래프 |
| mutation별 Hamming ratio | `summary_diffusion.csv` | grouped bar 또는 line plot |
| 샘플별 처리량 | `performance.csv` | 샘플별 막대그래프 |

## 9. 유의사항과 한계

1. 이 구현은 교육용 순수 Python 구현이다. 실제 보안 제품이나 고성능 라이브러리의 성능을 대표하지 않는다.
2. Hamming ratio와 entropy는 보조 지표이다. 이 값들이 좋아도 암호학적 안전성이 증명되는 것은 아니다.
3. Salsa8, Salsa12, ChaCha8, ChaCha12는 비교 관찰을 위한 감소 라운드 변형이다. 실제 사용 권고로 해석하지 않는다.
4. 본 실험은 기밀성 중심의 스트림 암호 비교이다. 메시지 무결성이나 인증은 ChaCha20-Poly1305 같은 AEAD 구성을 별도로 다뤄야 한다.
5. Nonce 재사용 실험은 통제된 오용 사례이다. 정상적인 스트림 암호 사용에서는 같은 key/nonce 쌍을 재사용하면 안 된다.

## 10. 최종 실행 체크리스트

최종 보고서용 실험 전에 다음을 확인한다.

| 체크 | 항목 |
|---|---|
| [ ] | `python3 -m compileall arx_streams.py run_arx_experiments.py` 통과 |
| [ ] | `python3 run_arx_experiments.py --repeats 5` 실행 |
| [ ] | ChaCha20 RFC 8439 block vector `PASS` 확인 |
| [ ] | `correctness.csv`의 `decrypt_matches`가 모두 `True`인지 확인 |
| [ ] | `summary_throughput.csv`를 최종 보고서 표에 반영 |
| [ ] | `summary_diffusion.csv`를 최종 보고서 표에 반영 |
| [ ] | `nonce_reuse.csv`의 XOR 관계 결과를 보고서에 반영 |
| [ ] | 반복 횟수, 실행 환경, 샘플 크기를 보고서에 명시 |
