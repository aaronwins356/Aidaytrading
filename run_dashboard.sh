#!/bin/bash
# Make the scripts executable: chmod +x run_bot.sh run_dashboard.sh
# Run this dashboard script with: ./run_dashboard.sh
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Launching dashboard..."
streamlit run "${PROJECT_ROOT}/dashboard/app.py" --server.address 0.0.0.0 --server.port 8501

