from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal


Algorithm = Literal["chacha", "salsa"]


def _rotl32(value: int, bits: int) -> int:
    value &= 0xFFFFFFFF
    return ((value << bits) & 0xFFFFFFFF) | (value >> (32 - bits))


def _u32_words(data: bytes) -> list[int]:
    return list(struct.unpack("<" + "I" * (len(data) // 4), data))


def _words_to_bytes(words: list[int]) -> bytes:
    return struct.pack("<" + "I" * len(words), *(word & 0xFFFFFFFF for word in words))


def _chacha_quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = _rotl32(state[d], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = _rotl32(state[b], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = _rotl32(state[d], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = _rotl32(state[b], 7)


def chacha_block(key: bytes, nonce: bytes, counter: int = 0, rounds: int = 20) -> bytes:
    if len(key) != 32:
        raise ValueError("ChaCha key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("ChaCha nonce must be 12 bytes")
    if rounds % 2:
        raise ValueError("ChaCha rounds must be even")

    constants = _u32_words(b"expand 32-byte k")
    state = constants + _u32_words(key) + [counter & 0xFFFFFFFF] + _u32_words(nonce)
    working = state.copy()

    for _ in range(rounds // 2):
        _chacha_quarter_round(working, 0, 4, 8, 12)
        _chacha_quarter_round(working, 1, 5, 9, 13)
        _chacha_quarter_round(working, 2, 6, 10, 14)
        _chacha_quarter_round(working, 3, 7, 11, 15)
        _chacha_quarter_round(working, 0, 5, 10, 15)
        _chacha_quarter_round(working, 1, 6, 11, 12)
        _chacha_quarter_round(working, 2, 7, 8, 13)
        _chacha_quarter_round(working, 3, 4, 9, 14)

    return _words_to_bytes([(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)])


def _salsa_quarter_round(y0: int, y1: int, y2: int, y3: int) -> tuple[int, int, int, int]:
    z1 = y1 ^ _rotl32((y0 + y3) & 0xFFFFFFFF, 7)
    z2 = y2 ^ _rotl32((z1 + y0) & 0xFFFFFFFF, 9)
    z3 = y3 ^ _rotl32((z2 + z1) & 0xFFFFFFFF, 13)
    z0 = y0 ^ _rotl32((z3 + z2) & 0xFFFFFFFF, 18)
    return z0, z1, z2, z3


def _salsa_rounds(words: list[int], rounds: int) -> list[int]:
    x = words.copy()
    for _ in range(rounds // 2):
        x[0], x[4], x[8], x[12] = _salsa_quarter_round(x[0], x[4], x[8], x[12])
        x[5], x[9], x[13], x[1] = _salsa_quarter_round(x[5], x[9], x[13], x[1])
        x[10], x[14], x[2], x[6] = _salsa_quarter_round(x[10], x[14], x[2], x[6])
        x[15], x[3], x[7], x[11] = _salsa_quarter_round(x[15], x[3], x[7], x[11])
        x[0], x[1], x[2], x[3] = _salsa_quarter_round(x[0], x[1], x[2], x[3])
        x[5], x[6], x[7], x[4] = _salsa_quarter_round(x[5], x[6], x[7], x[4])
        x[10], x[11], x[8], x[9] = _salsa_quarter_round(x[10], x[11], x[8], x[9])
        x[15], x[12], x[13], x[14] = _salsa_quarter_round(x[15], x[12], x[13], x[14])
    return x


def salsa_block(key: bytes, nonce: bytes, counter: int = 0, rounds: int = 20) -> bytes:
    if len(key) != 32:
        raise ValueError("Salsa20 key must be 32 bytes")
    if len(nonce) != 8:
        raise ValueError("Salsa20 nonce must be 8 bytes")
    if rounds % 2:
        raise ValueError("Salsa20 rounds must be even")

    c = _u32_words(b"expand 32-byte k")
    k = _u32_words(key)
    n = _u32_words(nonce)
    ctr = [counter & 0xFFFFFFFF, (counter >> 32) & 0xFFFFFFFF]
    state = [
        c[0], k[0], k[1], k[2],
        k[3], c[1], n[0], n[1],
        ctr[0], ctr[1], c[2], k[4],
        k[5], k[6], k[7], c[3],
    ]
    working = _salsa_rounds(state, rounds)
    return _words_to_bytes([(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)])


@dataclass(frozen=True)
class StreamCipher:
    algorithm: Algorithm
    rounds: int = 20

    @property
    def nonce_size(self) -> int:
        return 12 if self.algorithm == "chacha" else 8

    def block(self, key: bytes, nonce: bytes, counter: int) -> bytes:
        if self.algorithm == "chacha":
            return chacha_block(key, nonce, counter, self.rounds)
        if self.algorithm == "salsa":
            return salsa_block(key, nonce, counter, self.rounds)
        raise ValueError(f"unsupported algorithm: {self.algorithm}")

    def keystream(self, key: bytes, nonce: bytes, length: int, counter: int = 0) -> bytes:
        chunks: list[bytes] = []
        block_count = (length + 63) // 64
        for offset in range(block_count):
            chunks.append(self.block(key, nonce, counter + offset))
        return b"".join(chunks)[:length]

    def xor(self, data: bytes, key: bytes, nonce: bytes, counter: int = 0) -> bytes:
        stream = self.keystream(key, nonce, len(data), counter)
        return bytes(left ^ right for left, right in zip(data, stream))


def bit_flip(data: bytes, bit_index: int) -> bytes:
    if bit_index < 0 or bit_index >= len(data) * 8:
        raise ValueError("bit index out of range")
    mutable = bytearray(data)
    mutable[bit_index // 8] ^= 1 << (bit_index % 8)
    return bytes(mutable)


def hamming_distance(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("inputs must have the same length")
    return sum((a ^ b).bit_count() for a, b in zip(left, right))
