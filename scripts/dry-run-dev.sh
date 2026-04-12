#!/usr/bin/env bash
# Example: bench dry-run without `astrag` on PATH (uses repo .venv or uv).
set -euo pipefail
export LIBCLANG_PATH=/usr/lib/llvm-19/lib/libclang-19.so.1 
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/astrag" ]]; then
  exec "$ROOT/.venv/bin/astrag" run \
    --config configs/sqlitebrowser.yaml \
    --dataset datasets/samples.jsonl \
    --experiment-id dev \
    --dry-run
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run astrag run \
    --config configs/sqlitebrowser.yaml \
    --dataset datasets/samples.jsonl \
    --experiment-id dev \
    --dry-run
fi

echo "No .venv/bin/astrag and no uv. Run: cd $ROOT && uv sync --group dev" >&2
exit 1
