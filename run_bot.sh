#!/bin/bash
# Make the scripts executable: chmod +x run_bot.sh run_dashboard.sh
# Run this bot script with: ./run_bot.sh
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Launching bot..."
python "${PROJECT_ROOT}/main.py"

