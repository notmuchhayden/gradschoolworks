#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Insert a Java source file into an LLM prompt template.")
    parser.add_argument("--template", required=True, help="Prompt template path containing {{CODE}}")
    parser.add_argument("--source", required=True, help="Java source file path")
    parser.add_argument("--output", help="Optional output file. Prints to stdout when omitted.")
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    source_path = Path(args.source)
    source = source_path.read_text(encoding="utf-8")
    prompt = template.replace("{{CODE}}", f"```java\n{source}\n```")

    if args.output:
        Path(args.output).write_text(prompt, encoding="utf-8")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
