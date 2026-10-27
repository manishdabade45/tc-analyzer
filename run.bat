@echo off
cd /d "%~dp0"
echo Starting Terms ^& Conditions Analyzer...
call .venv\Scripts\activate.bat
streamlit run app.py
pause
