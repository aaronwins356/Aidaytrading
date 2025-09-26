@echo off
SETLOCAL
call ".venv\Scripts\activate.bat"
python -m ai_trader.main --mode api --config configs\config.yaml
ENDLOCAL
