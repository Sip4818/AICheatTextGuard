#!/usr/bin/env bash
set -euo pipefail

echo "=== Running pre-push checks ==="

echo ""
echo "--- ruff format (auto-fix) ---"
uv tool run ruff format .

echo ""
echo "--- ruff lint (auto-fix) ---"
uv tool run ruff check --fix .

echo ""
echo "--- ruff format (verify) ---"
uv tool run ruff format --check .

echo ""
echo "--- ruff lint (verify) ---"
uv tool run ruff check .

echo ""
echo "--- mypy type check ---"
if find . -path ./.venv -prune -o -name "*.py" -print -quit | grep -q .; then
  uv tool run mypy .
else
  echo "No Python files found; skipping mypy."
fi

echo ""
echo "--- pytest ---"
if [ -d tests ]; then
  uv tool run pytest tests/ -v
else
  echo "No tests directory found; skipping pytest."
fi

echo ""
echo "=== All checks passed! ==="
