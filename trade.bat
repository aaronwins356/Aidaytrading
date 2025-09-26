@echo off
call .venv\Scripts\activate
python -m ai_trader.main --mode live --config configs\config.yaml
pause
