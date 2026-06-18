#!/usr/bin/env bash
set -euo pipefail

echo "=== Running pre-push checks ==="

echo ""
echo "--- ruff format (check) ---"
ruff format --check .

echo ""
echo "--- ruff lint ---"
ruff check .

echo ""
echo "--- mypy type check ---"
if find . -path ./.venv -prune -o -name "*.py" -print -quit | grep -q .; then
  mypy .
else
  echo "No Python files found; skipping mypy."
fi

echo ""
echo "--- pytest ---"
if [ -d tests ]; then
  pytest tests/ -v
else
  echo "No tests directory found; skipping pytest."
fi

echo ""
echo "=== All checks passed! ==="
