@echo off
REM Navigate to project folder
cd /d C:\Users\moe\Desktop\my_trading_bot

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the trading bot
python main.py

REM Keep the window open after execution
pause
