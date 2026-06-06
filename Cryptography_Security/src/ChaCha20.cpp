#include "ChaCha20.h"

#include <algorithm>
#include <stdexcept>

namespace {
// 256비트 키 ChaCha 상태의 첫 행을 구성하는 "expand 32-byte k".
constexpr std::array<std::uint8_t, 16> sigma{
    'e', 'x', 'p', 'a', 'n', 'd', ' ', '3',
    '2', '-', 'b', 'y', 't', 'e', ' ', 'k',
};
}

ChaCha20::ChaCha20(const Key& key, const Nonce& nonce) {
    /*
    // ChaCha20의 4 x 4 상태는 다음과 같이 16개의 32비트 워드로 구성된다.
    [ 0] constant   [ 1] constant   [ 2] constant   [ 3] constant
    [ 4] key        [ 5] key        [ 6] key        [ 7] key
    [ 8] key        [ 9] key        [10] key        [11] key
    [12] counter    [13] counter    [14] nonce      [15] nonce
    
    */
    // ChaCha 초기 상태의 첫 행(0~3)에 네 개의 상수 워드를 배치한다.
    initial_state_[0] = load32(sigma.data());
    initial_state_[1] = load32(sigma.data() + 4);
    initial_state_[2] = load32(sigma.data() + 8);
    initial_state_[3] = load32(sigma.data() + 12);

    // 둘째와 셋째 행(4~11)에 256비트 키를 연속으로 배치한다.
    for (std::size_t i = 0; i < 8; ++i) {
        initial_state_[4 + i] = load32(key.data() + i * 4);
    }

    // 원 논문의 64비트 카운터/64비트 논스 형식을 사용한다.
    // 마지막 행의 12~13번은 카운터, 14~15번은 논스이다.
    initial_state_[14] = load32(nonce.data());
    initial_state_[15] = load32(nonce.data() + 4);
}

ChaCha20::Block ChaCha20::generateBlock(std::uint64_t block_counter) const {
    // 키와 논스가 들어 있는 공통 상태를 복사하고 현재 카운터를 설정한다.
    auto state = initial_state_;
    state[12] = static_cast<std::uint32_t>(block_counter);
    state[13] = static_cast<std::uint32_t>(block_counter >> 32);

    // 최종 feed-forward 덧셈을 위해 라운드 적용 전 상태를 보존한다.
    const auto original = state;

    // 각 반복은 column round 1회와 diagonal round 1회로 이루어진
    // double-round이다. 네 번 반복하면 ChaCha8의 8라운드가 된다.
    for (unsigned round = 0; round < Rounds; round += 2) {
        // 동일한 열에 있는 네 워드를 병렬적인 단위로 갱신한다.
        quarterRound(state[0], state[4], state[8], state[12]);
        quarterRound(state[1], state[5], state[9], state[13]);
        quarterRound(state[2], state[6], state[10], state[14]);
        quarterRound(state[3], state[7], state[11], state[15]);

        // 대각선 방향 조합을 갱신하여 서로 다른 열 사이로 값을 확산시킨다.
        quarterRound(state[0], state[5], state[10], state[15]);
        quarterRound(state[1], state[6], state[11], state[12]);
        quarterRound(state[2], state[7], state[8], state[13]);
        quarterRound(state[3], state[4], state[9], state[14]);
    }

    // 라운드 결과와 초기 상태를 modulo 2^32로 더한 뒤 직렬화한다.
    Block output{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        store32(output.data() + i * 4, state[i] + original[i]);
    }
    return output;
}

std::vector<std::uint8_t> ChaCha20::process(
    const std::vector<std::uint8_t>& input,
    std::uint64_t initial_block_counter) const {
    // 64바이트 단위로 처리하되, 마지막 불완전 블록도 하나로 계산한다.
    const std::uint64_t block_count =
        (static_cast<std::uint64_t>(input.size()) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 64비트 카운터가 순환하면 키스트림이 재사용될 수 있으므로 차단한다.
    if (block_count > 0 &&
        initial_block_counter > UINT64_MAX - (block_count - 1)) {
        throw std::overflow_error("ChaCha20 block counter overflow");
    }

    std::vector<std::uint8_t> output(input.size());
    for (std::uint64_t block_index = 0; block_index < block_count; ++block_index) {
        // 각 데이터 블록은 독립적인 카운터 값으로 키스트림을 생성한다.
        const Block keyStream = generateBlock(initial_block_counter + block_index);
        const std::size_t offset = static_cast<std::size_t>(block_index) * BLOCK_SIZE;

        // 입력 끝에서는 남아 있는 바이트 수만 처리한다.
        const std::size_t bytes =
            std::min(BLOCK_SIZE, input.size() - offset);
        for (std::size_t i = 0; i < bytes; ++i) {
            output[offset + i] = input[offset + i] ^ keyStream[i];
        }
    }
    return output;
}

std::uint32_t ChaCha20::rotateLeft(std::uint32_t value, unsigned bits) {
    // 고정 거리 회전은 데이터에 따른 분기나 테이블 조회가 필요 없다.
    return (value << bits) | (value >> (32U - bits));
}

std::uint32_t ChaCha20::load32(const std::uint8_t* input) {
    // 바이트 네 개를 ChaCha가 정의한 리틀 엔디언 32비트 워드로 변환한다.
    return static_cast<std::uint32_t>(input[0]) |
           (static_cast<std::uint32_t>(input[1]) << 8) |
           (static_cast<std::uint32_t>(input[2]) << 16) |
           (static_cast<std::uint32_t>(input[3]) << 24);
}

void ChaCha20::store32(std::uint8_t* output, std::uint32_t value) {
    // 32비트 워드를 하위 바이트부터 출력한다.
    output[0] = static_cast<std::uint8_t>(value);
    output[1] = static_cast<std::uint8_t>(value >> 8);
    output[2] = static_cast<std::uint8_t>(value >> 16);
    output[3] = static_cast<std::uint8_t>(value >> 24);
}

void ChaCha20::quarterRound(
    std::uint32_t& a,
    std::uint32_t& b,
    std::uint32_t& c,
    std::uint32_t& d) {
    // ChaCha quarter-round는 a, b, c, d를 각각 두 번 갱신한다.
    // 덧셈, XOR, 회전을 교차 적용하여 한 입력 워드의 변화가 나머지
    // 워드와 여러 비트 위치로 빠르게 퍼지도록 설계되어 있다.
    a += b;
    d = rotateLeft(d ^ a, 16);
    c += d;
    b = rotateLeft(b ^ c, 12);
    a += b;
    d = rotateLeft(d ^ a, 8);
    c += d;
    b = rotateLeft(b ^ c, 7);
}
