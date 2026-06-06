#include "ChaCha20.h"
#include "Salsa20.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// 명령행에서 반복 횟수를 생략했을 때 사용하는 기본 측정 횟수.
// 여러 번 반복한 총 시간을 나누어 짧은 단일 실행의 측정 오차를 줄인다.
constexpr std::size_t DefaultIterations = 30;

// 알고리즘 하나의 출력 데이터, 평균 성능, 정확성 검사 결과를 묶는다.
// 암호화/복호화 데이터도 보관하여 결과 파일 저장과 교차 검증에 사용한다.
struct BenchmarkResult {
    std::vector<std::uint8_t> encrypted;
    std::vector<std::uint8_t> decrypted;
    double encryptionMilliseconds{};
    double decryptionMilliseconds{};
    double encryptionMegabytesPerSecond{};
    double decryptionMegabytesPerSecond{};
    bool roundTripPassed{};
    bool ciphertextChanged{};
    bool deterministic{};
    bool counterChangesOutput{};
};

// 파일 전체를 바이너리 모드로 읽는다.
// 텍스트 모드의 줄바꿈 변환이 BMP 바이트를 변경하지 않도록 반드시
// std::ios::binary를 사용한다.
std::vector<std::uint8_t> readBinaryFile(const fs::path& path) {
    // 파일 끝에서 시작하면 tellg()로 전체 크기를 먼저 얻을 수 있다.
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open input file: " + path.string());
    }

    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine input size: " + path.string());
    }

    // 크기가 확정된 벡터를 한 번 할당한 뒤 파일 처음부터 한 번에 읽는다.
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    input.seekg(0);
    if (size > 0 &&
        !input.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Cannot read input file: " + path.string());
    }
    return data;
}

// 결과 벡터를 변경 없이 바이너리 파일로 저장한다.
void writeBinaryFile(const fs::path& path, const std::vector<std::uint8_t>& data) {
    // results 디렉터리가 없더라도 실행기가 직접 생성하도록 한다.
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create output file: " + path.string());
    }
    if (!data.empty()) {
        output.write(
            reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    }
    if (!output) {
        throw std::runtime_error("Cannot write output file: " + path.string());
    }
}

// Salsa20과 ChaCha20이 동일한 process() 인터페이스를 제공하므로
// 템플릿 함수 하나로 정확성 검사와 성능 측정을 공통 수행한다.
template <typename Cipher>
BenchmarkResult benchmarkCipher(
    const Cipher& cipher,
    const std::vector<std::uint8_t>& plaintext,
    std::size_t iterations) {
    BenchmarkResult result;

    // 스트림 암호는 같은 키스트림을 두 번 XOR하면 원문으로 돌아온다.
    result.encrypted = cipher.process(plaintext);
    result.decrypted = cipher.process(result.encrypted);

    // 계획서에서 정의한 네 가지 기본 동작 조건을 먼저 검증한다.
    result.roundTripPassed = result.decrypted == plaintext;
    result.ciphertextChanged = result.encrypted != plaintext;
    result.deterministic = cipher.process(plaintext) == result.encrypted;
    // 카운터가 키스트림 생성에 실제로 반영되는지도 별도로 확인한다.
    // 빈 입력은 비교할 출력이 없으므로 통과로 취급한다.
    result.counterChangesOutput =
        plaintext.empty() || cipher.process(plaintext, 1) != result.encrypted;

    // 파일 읽기/쓰기와 초기 검증은 측정 구간에서 제외한다.
    // 아래 구간에는 메모리 입력에 대한 키스트림 생성과 XOR 비용만 포함된다.
    std::vector<std::uint8_t> buffer;
    const auto encryptionStart = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        buffer = cipher.process(plaintext);
    }
    const auto encryptionEnd = Clock::now();

    const auto decryptionStart = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        buffer = cipher.process(result.encrypted);
    }
    const auto decryptionEnd = Clock::now();

    // 마지막 복호화 반복 결과도 원문과 같은지 확인하여 컴파일러가 계산을
    // 불필요한 코드로 판단하거나 측정 중 오류가 숨는 것을 방지한다.
    if (buffer != plaintext) {
        throw std::runtime_error("Benchmark decryption produced invalid output");
    }

    // 각 구간의 총 시간을 반복 횟수로 나누어 1회 평균 밀리초를 계산한다.
    const double encryptionTotalMs =
        std::chrono::duration<double, std::milli>(
            encryptionEnd - encryptionStart).count();
    const double decryptionTotalMs =
        std::chrono::duration<double, std::milli>(
            decryptionEnd - decryptionStart).count();
    result.encryptionMilliseconds =
        encryptionTotalMs / static_cast<double>(iterations);
    result.decryptionMilliseconds =
        decryptionTotalMs / static_cast<double>(iterations);

    // 처리량은 이진 메가바이트(MiB = 1024 * 1024바이트)를 기준으로 한다.
    const double mebibytes =
        static_cast<double>(plaintext.size()) / (1024.0 * 1024.0);
    result.encryptionMegabytesPerSecond =
        result.encryptionMilliseconds > 0.0
            ? mebibytes / (result.encryptionMilliseconds / 1000.0)
            : 0.0;
    result.decryptionMegabytesPerSecond =
        result.decryptionMilliseconds > 0.0
            ? mebibytes / (result.decryptionMilliseconds / 1000.0)
            : 0.0;
    return result;
}

// ChaCha8 핵심 블록 함수가 알려진 영 키/영 논스 테스트 벡터와
// 일치하는지 검사한다. 단순 왕복 검사는 같은 오류가 암호화와 복호화에
// 동시에 존재해도 통과할 수 있으므로, 외부 기준값 검증을 추가한다.
bool verifyChaCha8KnownVector() {
    const ChaCha20::Key key{};
    const ChaCha20::Nonce nonce{};
    const ChaCha20 cipher(key, nonce);

    // 256비트 영 키, 64비트 영 논스, 카운터 0인 ChaCha8의 첫 출력 블록.
    const ChaCha20::Block expected{
        0x3e, 0x00, 0xef, 0x2f, 0x89, 0x5f, 0x40, 0xd6,
        0x7f, 0x5b, 0xb8, 0xe8, 0x1f, 0x09, 0xa5, 0xa1,
        0x2c, 0x84, 0x0e, 0xc3, 0xce, 0x9a, 0x7f, 0x3b,
        0x18, 0x1b, 0xe1, 0x88, 0xef, 0x71, 0x1a, 0x1e,
        0x98, 0x4c, 0xe1, 0x72, 0xb9, 0x21, 0x6f, 0x41,
        0x9f, 0x44, 0x53, 0x67, 0x45, 0x6d, 0x56, 0x19,
        0x31, 0x4a, 0x42, 0xa3, 0xda, 0x86, 0xb0, 0x01,
        0x38, 0x7b, 0xfd, 0xb8, 0x0e, 0x0c, 0xfe, 0x42,
    };
    return cipher.generateBlock(0) == expected;
}

// 알고리즘별 필수 검증 항목을 하나의 최종 조건으로 결합한다.
bool allChecksPassed(const BenchmarkResult& result) {
    return result.roundTripPassed &&
           result.ciphertextChanged &&
           result.deterministic &&
           result.counterChangesOutput;
}

// 알고리즘별 평균 시간, 처리량, 검증 결과를 동일한 형식으로 출력한다.
void printResult(const std::string& name, const BenchmarkResult& result) {
    std::cout << '\n' << '[' << name << "]\n"
              << "Encryption average: " << result.encryptionMilliseconds
              << " ms (" << result.encryptionMegabytesPerSecond << " MiB/s)\n"
              << "Decryption average: " << result.decryptionMilliseconds
              << " ms (" << result.decryptionMegabytesPerSecond << " MiB/s)\n"
              << "Round-trip verification: "
              << (result.roundTripPassed ? "PASS" : "FAIL") << '\n'
              << "Ciphertext differs from input: "
              << (result.ciphertextChanged ? "PASS" : "FAIL") << '\n'
              << "Deterministic output: "
              << (result.deterministic ? "PASS" : "FAIL") << '\n'
              << "Block counter changes output: "
              << (result.counterChangesOutput ? "PASS" : "FAIL") << '\n';
}

// 두 번째 명령행 인자를 0보다 큰 반복 횟수로 변환한다.
// 숫자 뒤에 다른 문자가 붙은 값도 오류로 처리한다.
std::size_t parseIterations(const char* text) {
    const std::string value(text);
    std::size_t parsedCharacters = 0;
    const unsigned long parsed = std::stoul(value, &parsedCharacters);
    if (parsedCharacters != value.size() || parsed == 0) {
        throw std::invalid_argument("Iteration count must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}
}

int main(int argc, char* argv[]) {
    try {
#ifdef ARX_PROJECT_DIR
        // CMake가 주입한 소스 루트를 사용하면 어느 작업 디렉터리에서
        // 실행하더라도 기본 입력과 results 경로를 찾을 수 있다.
        const fs::path projectDirectory = ARX_PROJECT_DIR;
#else
        // CMake 외부에서 직접 빌드한 경우에는 현재 디렉터리를 기준으로 한다.
        const fs::path projectDirectory = fs::current_path();
#endif
        // 첫 번째 인자는 입력 경로, 두 번째 인자는 반복 횟수이다.
        // 인자가 없으면 저장소에 포함된 기본 gradient BMP를 사용한다.
        const fs::path inputPath =
            argc >= 2 ? fs::path(argv[1])
                      : projectDirectory / "data_samples" / "gradient_512x512.bmp";
        const std::size_t iterations =
            argc >= 3 ? parseIterations(argv[2]) : DefaultIterations;
        const fs::path resultDirectory = projectDirectory / "results";

        const std::vector<std::uint8_t> plaintext = readBinaryFile(inputPath);

        // 두 알고리즘을 같은 입력 조건에서 비교하기 위해 동일한 바이트
        // 패턴의 256비트 키와 64비트 논스를 사용한다.
        Salsa20::Key salsaKey{};
        ChaCha20::Key chaChaKey{};
        Salsa20::Nonce salsaNonce{};
        ChaCha20::Nonce chaChaNonce{};

        // 키는 00, 01, ..., 1f로 구성한다.
        for (std::size_t i = 0; i < salsaKey.size(); ++i) {
            salsaKey[i] = static_cast<std::uint8_t>(i);
            chaChaKey[i] = static_cast<std::uint8_t>(i);
        }

        // 논스는 a0, a1, ..., a7로 구성한다.
        for (std::size_t i = 0; i < salsaNonce.size(); ++i) {
            salsaNonce[i] = static_cast<std::uint8_t>(0xa0 + i);
            chaChaNonce[i] = static_cast<std::uint8_t>(0xa0 + i);
        }

        const Salsa20 salsa20(salsaKey, salsaNonce);
        const ChaCha20 chaCha20(chaChaKey, chaChaNonce);

        // 두 알고리즘을 같은 입력 크기와 반복 횟수로 순차 측정한다.
        const BenchmarkResult salsaResult =
            benchmarkCipher(salsa20, plaintext, iterations);
        const BenchmarkResult chaChaResult =
            benchmarkCipher(chaCha20, plaintext, iterations);

        // 왕복 검사 외에 ChaCha8 외부 기준값과 알고리즘 간 출력 차이를 확인한다.
        const bool chaChaVectorPassed = verifyChaCha8KnownVector();
        const bool ciphertextsDiffer =
            salsaResult.encrypted != chaChaResult.encrypted;

        // 암호문과 복호화 결과를 보고서 확인용 파일로 저장한다.
        writeBinaryFile(
            resultDirectory / "salsa20_encrypted.bmp",
            salsaResult.encrypted);
        writeBinaryFile(
            resultDirectory / "salsa20_decrypted.bmp",
            salsaResult.decrypted);
        writeBinaryFile(
            resultDirectory / "chacha20_encrypted.bmp",
            chaChaResult.encrypted);
        writeBinaryFile(
            resultDirectory / "chacha20_decrypted.bmp",
            chaChaResult.decrypted);

        // 시간은 소수점 셋째 자리까지, 처리량은 MiB/s 단위로 출력한다.
        std::cout << std::fixed << std::setprecision(3)
                  << "Input file: " << inputPath << '\n'
                  << "Input size: " << plaintext.size() << " bytes\n"
                  << "Benchmark iterations: " << iterations << '\n'
                  << "ChaCha8 known vector: "
                  << (chaChaVectorPassed ? "PASS" : "FAIL") << '\n';
        printResult("Salsa20/8", salsaResult);
        printResult("ChaCha8", chaChaResult);
        std::cout << "\nDifferent algorithm ciphertexts: "
                  << (ciphertextsDiffer ? "PASS" : "FAIL") << '\n'
                  << "Result directory: " << resultDirectory << '\n';

        // 어느 한 항목이라도 실패하면 비정상 종료 코드 1을 반환하여
        // CTest에서도 테스트 실패로 인식할 수 있게 한다.
        const bool passed =
            chaChaVectorPassed &&
            allChecksPassed(salsaResult) &&
            allChecksPassed(chaChaResult) &&
            ciphertextsDiffer;
        std::cout << "Overall result: " << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        // 파일 접근, 인자 변환, 카운터 오버플로 등의 오류를 한곳에서 보고한다.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
