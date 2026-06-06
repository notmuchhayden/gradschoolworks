#include "Salsa20.h"

#include <algorithm>
#include <stdexcept>

namespace {
// 256비트 키를 사용하는 Salsa20의 고정 문자열 "expand 32-byte k".
// 네 개의 32비트 리틀 엔디언 워드로 나누어 상태의 대각선에 배치한다.
constexpr std::array<std::uint8_t, 16> Sigma{
    'e', 'x', 'p', 'a', 'n', 'd', ' ', '3',
    '2', '-', 'b', 'y', 't', 'e', ' ', 'k',
};
}

Salsa20::Salsa20(const Key& key, const Nonce& nonce) {
    // Salsa20 초기 상태의 상수는 0, 5, 10, 15번 대각선에 놓인다.
    initialState_[0] = load32(Sigma.data());
    initialState_[5] = load32(Sigma.data() + 4);
    initialState_[10] = load32(Sigma.data() + 8);
    initialState_[15] = load32(Sigma.data() + 12);

    // 256비트 키의 앞 절반은 1~4번, 뒤 절반은 11~14번에 배치한다.
    initialState_[1] = load32(key.data());
    initialState_[2] = load32(key.data() + 4);
    initialState_[3] = load32(key.data() + 8);
    initialState_[4] = load32(key.data() + 12);
    initialState_[11] = load32(key.data() + 16);
    initialState_[12] = load32(key.data() + 20);
    initialState_[13] = load32(key.data() + 24);
    initialState_[14] = load32(key.data() + 28);

    // 64비트 논스는 6번과 7번에 저장한다. 8번과 9번은 카운터용이다.
    initialState_[6] = load32(nonce.data());
    initialState_[7] = load32(nonce.data() + 4);
}

Salsa20::Block Salsa20::generateBlock(std::uint64_t blockCounter) const {
    // 생성자에서 만든 공통 상태를 복사한 뒤 현재 블록 번호만 채운다.
    // 하위 32비트가 먼저 오는 리틀 엔디언 워드 순서를 사용한다.
    auto state = initialState_;
    state[8] = static_cast<std::uint32_t>(blockCounter);
    state[9] = static_cast<std::uint32_t>(blockCounter >> 32);

    // 라운드가 끝난 상태에 원래 상태를 더하는 feed-forward 단계에 사용한다.
    const auto original = state;

    // column round와 row round가 합쳐져 2라운드가 된다.
    // Salsa20/8에서는 이 double-round를 네 번 수행한다.
    for (unsigned round = 0; round < Rounds; round += 2) {
        columnRound(state);
        rowRound(state);
    }

    // 변환된 각 워드에 초기 워드를 모듈러 2^32 덧셈한 후
    // 리틀 엔디언 바이트 순서로 직렬화하여 64바이트 키스트림을 만든다.
    Block output{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        store32(output.data() + i * 4, state[i] + original[i]);
    }
    return output;
}

std::vector<std::uint8_t> Salsa20::process(
    const std::vector<std::uint8_t>& input,
    std::uint64_t initialBlockCounter) const {
    // 마지막 불완전 블록도 처리할 수 있도록 올림 나눗셈으로 블록 수를 구한다.
    const std::uint64_t blockCount =
        (static_cast<std::uint64_t>(input.size()) + BlockSize - 1) / BlockSize;

    // 카운터가 UINT64_MAX를 넘어 다시 0으로 순환하면 같은 키스트림이
    // 재사용되므로, 필요한 마지막 블록 번호를 표현할 수 있는지 확인한다.
    if (blockCount > 0 &&
        initialBlockCounter > UINT64_MAX - (blockCount - 1)) {
        throw std::overflow_error("Salsa20 block counter overflow");
    }

    std::vector<std::uint8_t> output(input.size());
    for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
        // 블록마다 카운터를 1씩 증가시켜 서로 다른 키스트림을 생성한다.
        const Block keyStream = generateBlock(initialBlockCounter + blockIndex);
        const std::size_t offset = static_cast<std::size_t>(blockIndex) * BlockSize;

        // 마지막 블록이 64바이트보다 짧으면 실제 남은 바이트만 XOR한다.
        const std::size_t bytes =
            std::min(BlockSize, input.size() - offset);
        for (std::size_t i = 0; i < bytes; ++i) {
            output[offset + i] = input[offset + i] ^ keyStream[i];
        }
    }
    return output;
}

std::uint32_t Salsa20::rotateLeft(std::uint32_t value, unsigned bits) {
    // 밀려나간 상위 비트를 하위 위치로 되돌려 32비트 순환 회전을 만든다.
    return (value << bits) | (value >> (32U - bits));
}

std::uint32_t Salsa20::load32(const std::uint8_t* input) {
    // 플랫폼의 실제 엔디언과 무관하게 네 바이트를 리틀 엔디언 워드로 조립한다.
    return static_cast<std::uint32_t>(input[0]) |
           (static_cast<std::uint32_t>(input[1]) << 8) |
           (static_cast<std::uint32_t>(input[2]) << 16) |
           (static_cast<std::uint32_t>(input[3]) << 24);
}

void Salsa20::store32(std::uint8_t* output, std::uint32_t value) {
    // 내부 워드를 Salsa20 출력 규격인 리틀 엔디언 바이트 순서로 분해한다.
    output[0] = static_cast<std::uint8_t>(value);
    output[1] = static_cast<std::uint8_t>(value >> 8);
    output[2] = static_cast<std::uint8_t>(value >> 16);
    output[3] = static_cast<std::uint8_t>(value >> 24);
}

void Salsa20::quarterRound(
    std::uint32_t& a,
    std::uint32_t& b,
    std::uint32_t& c,
    std::uint32_t& d) {
    // 덧셈은 uint32_t 오버플로 규칙에 따라 자동으로 modulo 2^32가 된다.
    // 각 단계의 변경 결과가 다음 단계 입력으로 사용되어 네 워드로 확산된다.
    b ^= rotateLeft(a + d, 7);
    c ^= rotateLeft(b + a, 9);
    d ^= rotateLeft(c + b, 13);
    a ^= rotateLeft(d + c, 18);
}

void Salsa20::columnRound(std::array<std::uint32_t, 16>& state) {
    // 4 x 4 상태의 네 열에 quarter-round를 적용한다.
    // 두 번째 이후 열은 Salsa20 규격에 맞는 순환된 인덱스 순서를 사용한다.
    quarterRound(state[0], state[4], state[8], state[12]);
    quarterRound(state[5], state[9], state[13], state[1]);
    quarterRound(state[10], state[14], state[2], state[6]);
    quarterRound(state[15], state[3], state[7], state[11]);
}

void Salsa20::rowRound(std::array<std::uint32_t, 16>& state) {
    // 이어서 네 행에 quarter-round를 적용해 열 사이의 정보도 확산시킨다.
    quarterRound(state[0], state[1], state[2], state[3]);
    quarterRound(state[5], state[6], state[7], state[4]);
    quarterRound(state[10], state[11], state[8], state[9]);
    quarterRound(state[15], state[12], state[13], state[14]);
}
