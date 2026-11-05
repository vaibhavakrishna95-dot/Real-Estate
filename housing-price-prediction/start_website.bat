@echo off
title Ames Housing AI Valuation Platform
echo ========================================================
echo    Ames Real Estate AI Valuation Platform
echo    Machine Learning Solution & Web Application
echo ========================================================
echo.
cd /d "%~dp0"
echo Starting web server on http://127.0.0.1:8000 ...
echo Press Ctrl+C in this window to stop the server.
echo.
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
