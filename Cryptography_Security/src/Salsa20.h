#ifndef CRYPTOGRAPHY_SECURITY_SALSA20_H
#define CRYPTOGRAPHY_SECURITY_SALSA20_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// 256비트 키와 64비트 논스를 사용하는 Salsa20/8 스트림 암호 구현.
//
// 클래스 이름은 프로젝트 파일 구성에 맞춰 Salsa20으로 정했지만, 실제
// 라운드 수는 Rounds 상수에 지정된 8라운드이다. 암호화와 복호화는 모두
// 입력 데이터와 동일한 키스트림을 XOR하는 process() 함수로 수행한다.
class Salsa20 {
public:
    // Salsa20의 키, 논스, 출력 블록 크기를 바이트 단위로 나타낸다.
    static constexpr std::size_t KEY_SIZE = 32;
    static constexpr std::size_t NONCE_SIZE = 8;
    static constexpr std::size_t BLOCK_SIZE = 64;
    static constexpr unsigned ROUNDS = 8;

    // 고정 크기 배열을 사용해 잘못된 길이의 키나 논스가 전달되지 않도록 한다.
    using Key = std::array<std::uint8_t, KEY_SIZE>;
    using Nonce = std::array<std::uint8_t, NONCE_SIZE>;
    using Block = std::array<std::uint8_t, BLOCK_SIZE>;

    // 키와 논스를 Salsa20 초기 상태에 배치한다.
    // 블록 카운터는 블록마다 달라지므로 생성자에서 설정하지 않는다.
    Salsa20(const Key& key, const Nonce& nonce);

    // 지정된 64비트 블록 카운터에 대응하는 64바이트 키스트림을 생성한다.
    Block generateBlock(std::uint64_t block_counter) const;

    // 입력 전체를 키스트림과 XOR한다.
    // 같은 키, 논스, initial_block_counter로 다시 호출하면 복호화된다.
    std::vector<std::uint8_t> process(
        const std::vector<std::uint8_t>& input,
        std::uint64_t initial_block_counter = 0) const;

private:
    // Salsa20에서 사용하는 32비트 순환 왼쪽 회전 연산.
    static std::uint32_t rotateLeft(std::uint32_t value, unsigned bits);

    // 바이트 배열과 32비트 워드 사이를 리틀 엔디언으로 변환한다.
    static std::uint32_t load32(const std::uint8_t* input);
    static void store32(std::uint8_t* output, std::uint32_t value);

    // 네 워드에 Salsa20의 add-rotate-xor 연산을 순차적으로 적용한다.
    static void quarterRound(
        std::uint32_t& a,
        std::uint32_t& b,
        std::uint32_t& c,
        std::uint32_t& d);

    // 한 라운드에서 상태의 열 또는 행 방향으로 네 quarter-round를 적용한다.
    static void columnRound(std::array<std::uint32_t, 16>& state);
    static void rowRound(std::array<std::uint32_t, 16>& state);

private:
    // 상수, 키, 논스가 배치된 4 x 4 워드 상태.
    // 카운터 위치인 8번과 9번 워드는 generateBlock()에서 채운다.
    std::array<std::uint32_t, 16> initial_state_{};
};

#endif
