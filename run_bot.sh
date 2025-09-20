#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[run_bot] Launching trading runtime"
exec python "${PROJECT_ROOT}/main.py"
