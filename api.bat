@echo off
call .venv\Scripts\activate
python -m ai_trader.main --mode api --config configs\config.yaml
pause
