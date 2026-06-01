#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/target/classes"
SOURCE_LIST="$ROOT_DIR/target/main-sources.txt"

mkdir -p "$BUILD_DIR"
find "$ROOT_DIR/src/main/java" -name "*.java" | sort > "$SOURCE_LIST"

if [ ! -s "$SOURCE_LIST" ]; then
  echo "No Java source files found."
  exit 1
fi

javac -d "$BUILD_DIR" @"$SOURCE_LIST"
echo "Compiled main sources into $BUILD_DIR"
