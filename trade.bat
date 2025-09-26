@echo off
SETLOCAL
call ".venv\Scripts\activate.bat"
python -m ai_trader.main --mode live --config configs\config.yaml
ENDLOCAL
