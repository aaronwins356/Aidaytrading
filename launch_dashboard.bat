@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Run Streamlit dashboard
streamlit run desk\apps\dashboard.py --server.port=8501

pause
