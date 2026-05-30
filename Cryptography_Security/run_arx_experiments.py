from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

from arx_streams import StreamCipher, bit_flip, chacha_block, hamming_distance


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_samples_small"
RESULT_DIR = ROOT / "results"


def deterministic_bytes(length: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(length))


def write_ppm(path: Path, width: int, height: int, variant: int) -> None:
    header = f"P6\n{width} {height}\n255\n".encode()
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            if variant == 0:
                pixels.extend(((x * 255) // width, (y * 255) // height, ((x ^ y) * 255) // max(width, height)))
            else:
                band = 255 if ((x // 24) + (y // 24)) % 2 else 32
                pixels.extend((band, (x * 13 + y * 3) % 256, (x * 5 + y * 11) % 256))
    path.write_bytes(header + pixels)


def write_bmp(path: Path, width: int, height: int, variant: int) -> None:
    row_size = ((24 * width + 31) // 32) * 4
    pixel_size = row_size * height
    file_size = 54 + pixel_size
    header = bytearray(b"BM")
    header.extend(file_size.to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((54).to_bytes(4, "little"))
    header.extend((40).to_bytes(4, "little"))
    header.extend(width.to_bytes(4, "little"))
    header.extend(height.to_bytes(4, "little"))
    header.extend((1).to_bytes(2, "little"))
    header.extend((24).to_bytes(2, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend(pixel_size.to_bytes(4, "little"))
    header.extend((2835).to_bytes(4, "little"))
    header.extend((2835).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))

    rows = bytearray()
    for y in reversed(range(height)):
        row = bytearray()
        for x in range(width):
            if variant == 0:
                r, g, b = (x * 255) // width, (y * 255) // height, ((x + y) * 255) // (width + height)
            else:
                r = 230 if x % 64 < 32 else 40
                g = 230 if y % 64 < 32 else 40
                b = (x * y) % 256
            row.extend((b, g, r))
        row.extend(b"\x00" * (row_size - len(row)))
        rows.extend(row)
    path.write_bytes(bytes(header + rows))


def write_video_like(path: Path, width: int, height: int, frames: int) -> None:
    data = bytearray(b"RAWVIDEO\n")
    data.extend(f"{width} {height} {frames}\n".encode())
    for frame in range(frames):
        for y in range(height):
            for x in range(width):
                data.extend(((x + frame * 3) % 256, (y + frame * 7) % 256, (x + y + frame * 11) % 256))
    path.write_bytes(data)


def ensure_samples() -> list[Path]:
    DATA_DIR.mkdir(exist_ok=True)
    samples = [
        DATA_DIR / "gradient_128x128.ppm",
        DATA_DIR / "pattern_128x128.ppm",
        DATA_DIR / "gradient_128x128.bmp",
        DATA_DIR / "pattern_128x128.bmp",
        DATA_DIR / "video_like_64x36x12.rgb",
        DATA_DIR / "random_128KiB.bin",
    ]
    if not samples[0].exists():
        write_ppm(samples[0], 128, 128, 0)
    if not samples[1].exists():
        write_ppm(samples[1], 128, 128, 1)
    if not samples[2].exists():
        write_bmp(samples[2], 128, 128, 0)
    if not samples[3].exists():
        write_bmp(samples[3], 128, 128, 1)
    if not samples[4].exists():
        write_video_like(samples[4], 64, 36, 12)
    if not samples[5].exists():
        samples[5].write_bytes(deterministic_bytes(128 * 1024, 20260526))
    return samples


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shannon_entropy(data: bytes) -> float:
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts if count)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def verify_chacha_vector() -> bool:
    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    expected = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4"
        "c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2"
        "b5129cd1de164eb9cbd083e8a2503c4e"
    )
    return chacha_block(key, nonce, counter=1, rounds=20) == expected


def run_correctness(samples: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    key = bytes(range(32))
    for algorithm in ("chacha", "salsa"):
        cipher = StreamCipher(algorithm)
        nonce = bytes(range(cipher.nonce_size))
        for sample in samples:
            plain = sample.read_bytes()
            ciphertext = cipher.xor(plain, key, nonce)
            recovered = cipher.xor(ciphertext, key, nonce)
            rows.append(
                {
                    "algorithm": algorithm,
                    "sample": sample.name,
                    "bytes": len(plain),
                    "decrypt_matches": recovered == plain,
                    "plain_sha256": sha256(plain),
                    "cipher_sha256": sha256(ciphertext),
                    "cipher_entropy": f"{shannon_entropy(ciphertext):.6f}",
                }
            )
    return rows


def run_performance(samples: list[Path], repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    key = bytes((i * 7) % 256 for i in range(32))
    for algorithm in ("chacha", "salsa"):
        for rounds in (8, 12, 20):
            cipher = StreamCipher(algorithm, rounds)
            nonce = bytes((i * 11) % 256 for i in range(cipher.nonce_size))
            for sample in samples:
                plain = sample.read_bytes()
                durations = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    encrypted = cipher.xor(plain, key, nonce)
                    decrypted = cipher.xor(encrypted, key, nonce)
                    ended = time.perf_counter()
                    if decrypted != plain:
                        raise RuntimeError(f"round-trip failed for {algorithm}{rounds} {sample}")
                    durations.append((ended - started) / 2)
                mean = statistics.mean(durations)
                stdev = statistics.stdev(durations) if len(durations) > 1 else 0.0
                mbps = (len(plain) / (1024 * 1024)) / mean
                rows.append(
                    {
                        "algorithm": f"{algorithm}{rounds}",
                        "sample": sample.name,
                        "bytes": len(plain),
                        "repeats": repeats,
                        "mean_seconds": f"{mean:.6f}",
                        "stdev_seconds": f"{stdev:.6f}",
                        "throughput_MiB_s": f"{mbps:.2f}",
                    }
                )
    return rows


def run_diffusion() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    key = bytes(range(32))
    for algorithm in ("chacha", "salsa"):
        for rounds in (8, 12, 20):
            cipher = StreamCipher(algorithm, rounds)
            nonce = bytes(range(cipher.nonce_size))
            baseline = cipher.keystream(key, nonce, 4096)
            for mutation, changed_key, changed_nonce in (
                ("key_bit_0", bit_flip(key, 0), nonce),
                ("key_bit_127", bit_flip(key, 127), nonce),
                ("nonce_bit_0", key, bit_flip(nonce, 0)),
                ("nonce_last_bit", key, bit_flip(nonce, len(nonce) * 8 - 1)),
            ):
                changed = cipher.keystream(changed_key, changed_nonce, 4096)
                for block in range(64):
                    left = baseline[block * 64:(block + 1) * 64]
                    right = changed[block * 64:(block + 1) * 64]
                    distance = hamming_distance(left, right)
                    rows.append(
                        {
                            "algorithm": f"{algorithm}{rounds}",
                            "mutation": mutation,
                            "block_index": block,
                            "hamming_bits": distance,
                            "hamming_ratio": f"{distance / 512:.6f}",
                        }
                    )
    return rows


def run_nonce_reuse(samples: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pairs = [(samples[0], samples[1]), (samples[2], samples[3])]
    key = bytes((255 - i) for i in range(32))
    for algorithm in ("chacha", "salsa"):
        cipher = StreamCipher(algorithm)
        nonce = bytes([42] * cipher.nonce_size)
        for first, second in pairs:
            left = first.read_bytes()
            right = second.read_bytes()
            size = min(len(left), len(right))
            left, right = left[:size], right[:size]
            c1 = cipher.xor(left, key, nonce)
            c2 = cipher.xor(right, key, nonce)
            xor_cipher = bytes(a ^ b for a, b in zip(c1, c2))
            xor_plain = bytes(a ^ b for a, b in zip(left, right))
            rows.append(
                {
                    "algorithm": algorithm,
                    "pair": f"{first.name}/{second.name}",
                    "bytes_compared": size,
                    "cipher_xor_equals_plain_xor": xor_cipher == xor_plain,
                    "plain_xor_entropy": f"{shannon_entropy(xor_plain):.6f}",
                    "cipher_xor_entropy": f"{shannon_entropy(xor_cipher):.6f}",
                    "xor_sha256": sha256(xor_cipher),
                }
            )
    return rows


def write_summary(performance_rows: list[dict[str, object]], diffusion_rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[float]] = {}
    for row in performance_rows:
        grouped.setdefault(str(row["algorithm"]), []).append(float(row["throughput_MiB_s"]))
    perf_lines = ["algorithm,mean_throughput_MiB_s"]
    for name, values in sorted(grouped.items()):
        perf_lines.append(f"{name},{statistics.mean(values):.2f}")

    diffusion_grouped: dict[tuple[str, str], list[float]] = {}
    for row in diffusion_rows:
        key = (str(row["algorithm"]), str(row["mutation"]))
        diffusion_grouped.setdefault(key, []).append(float(row["hamming_ratio"]))
    diff_lines = ["algorithm,mutation,mean_hamming_ratio"]
    for (algorithm, mutation), values in sorted(diffusion_grouped.items()):
        diff_lines.append(f"{algorithm},{mutation},{statistics.mean(values):.6f}")

    (RESULT_DIR / "summary_throughput.csv").write_text("\n".join(perf_lines) + "\n")
    (RESULT_DIR / "summary_diffusion.csv").write_text("\n".join(diff_lines) + "\n")


def write_environment() -> None:
    lines = [
        f"python,{platform.python_version()}",
        f"platform,{platform.platform()}",
        f"processor,{platform.processor() or 'unknown'}",
        f"cpu_count,{os.cpu_count()}",
    ]
    (RESULT_DIR / "environment.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)
    samples = ensure_samples()
    correctness = run_correctness(samples)
    performance = run_performance(samples, args.repeats)
    diffusion = run_diffusion()
    nonce_reuse = run_nonce_reuse(samples)

    write_csv(RESULT_DIR / "correctness.csv", correctness)
    write_csv(RESULT_DIR / "performance.csv", performance)
    write_csv(RESULT_DIR / "diffusion.csv", diffusion)
    write_csv(RESULT_DIR / "nonce_reuse.csv", nonce_reuse)
    write_summary(performance, diffusion)
    write_environment()
    print(f"ChaCha20 RFC 8439 block vector: {'PASS' if verify_chacha_vector() else 'FAIL'}")
    print(f"Wrote results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
