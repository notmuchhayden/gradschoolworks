#!/usr/bin/env python3
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples.csv"
OUTPUT = ROOT / "results" / "metrics_before.csv"

METHOD_PATTERN = re.compile(
    r"\b(public|protected|private)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+\w+\s*\([^;{}]*\)\s*\{"
)
BRANCH_PATTERN = re.compile(r"\b(if|for|while|case|catch)\b|&&|\|\|")


def read_text(path):
    return path.read_text(encoding="utf-8")


def count_loc(text):
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("//"))


def count_methods(text):
    return len(METHOD_PATTERN.findall(text))


def approximate_complexity(text):
    return 1 + len(BRANCH_PATTERN.findall(text))


def main():
    rows = []
    with SAMPLES.open(newline="", encoding="utf-8") as sample_file:
        for sample in csv.DictReader(sample_file):
            main_file = ROOT / sample["main_file"]
            text = read_text(main_file)
            rows.append({
                "sample_id": sample["sample_id"],
                "smell_type": sample["smell_type"],
                "loc_before": count_loc(text),
                "method_count_before": count_methods(text),
                "complexity_before": approximate_complexity(text),
                "pmd_violations_before": "",
                "checkstyle_violations_before": "",
                "duplication_before": "",
                "test_result_before": "",
            })

    with OUTPUT.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "sample_id",
            "smell_type",
            "loc_before",
            "method_count_before",
            "complexity_before",
            "pmd_violations_before",
            "checkstyle_violations_before",
            "duplication_before",
            "test_result_before",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
