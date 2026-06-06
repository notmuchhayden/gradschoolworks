#ifndef CRYPTOGRAPHY_SECURITY_CHACHA20_H
#define CRYPTOGRAPHY_SECURITY_CHACHA20_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Bernstein의 원 논문 형식인 256비트 키, 64비트 카운터,
// 64비트 논스를 사용하는 ChaCha8 스트림 암호 구현.
//
// 파일명과 클래스명은 프로젝트 계획에 따라 ChaCha20으로 유지하지만,
// 실험에서 비교하는 변형은 Rounds가 8인 ChaCha8이다.
class ChaCha20 {
public:
    // ChaCha의 키, 논스, 출력 블록 크기를 바이트 단위로 나타낸다.
    static constexpr std::size_t KEY_SIZE = 32;
    static constexpr std::size_t NONCE_SIZE = 8;
    static constexpr std::size_t BLOCK_SIZE = 64;
    static constexpr unsigned Rounds = 8;

    // 컴파일 시점에 키, 논스, 블록 길이를 고정한다.
    using Key = std::array<std::uint8_t, KEY_SIZE>;
    using Nonce = std::array<std::uint8_t, NONCE_SIZE>;
    using Block = std::array<std::uint8_t, BLOCK_SIZE>;

    // 상수, 키, 논스를 초기 상태에 저장한다.
    // 64비트 블록 카운터는 generateBlock() 호출 시 설정한다.
    ChaCha20(const Key& key, const Nonce& nonce);

    // 지정된 블록 카운터의 ChaCha8 키스트림 64바이트를 생성한다.
    Block generateBlock(std::uint64_t block_counter) const;

    // 입력과 키스트림을 XOR하여 암호화 또는 복호화한다.
    std::vector<std::uint8_t> process(
        const std::vector<std::uint8_t>& input,
        std::uint64_t initial_block_counter = 0) const;

private:
    // 4 x 4 상태 행렬. 12번과 13번 카운터 워드는 블록마다 갱신된다.
    std::array<std::uint32_t, 16> initial_state_{};

    // ChaCha ARX 연산에서 사용하는 32비트 순환 왼쪽 회전.
    static std::uint32_t rotateLeft(std::uint32_t value, unsigned bits);

    // 외부 바이트 표현과 내부 32비트 워드 표현을 리틀 엔디언으로 변환한다.
    static std::uint32_t load32(const std::uint8_t* input);
    static void store32(std::uint8_t* output, std::uint32_t value);

    // 네 상태 워드를 각각 두 번 갱신하는 ChaCha quarter-round.
    static void quarterRound(
        std::uint32_t& a,
        std::uint32_t& b,
        std::uint32_t& c,
        std::uint32_t& d);
};

#endif
