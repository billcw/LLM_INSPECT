#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Set up .venv using README.md first (Python 3.11 or 3.12)."
  exit 1
fi
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
