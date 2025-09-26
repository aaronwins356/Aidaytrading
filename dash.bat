@echo off
SETLOCAL
call ".venv\Scripts\activate.bat"
streamlit run ai_trader\streamlit_app.py
ENDLOCAL
