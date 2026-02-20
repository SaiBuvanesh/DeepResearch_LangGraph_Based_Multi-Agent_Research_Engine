@echo off
setlocal
cd /d "%~dp0"

echo =========================================
echo   DeepResearch Agent - Startup Script
echo =========================================

REM Check if virtual environment exists
if not exist "env" (
    echo [ERROR] Virtual environment 'env' not found in %CD%
    echo Please create it and install requirements:
    echo   python -m venv env
    echo   env\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call "env\Scripts\activate"

echo [INFO] Verifying Streamlit installation...
python -m streamlit --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Streamlit not found in environment.
    echo [INFO] Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
)

echo [INFO] Launching Streamlit application...
echo [INFO] Visit http://localhost:8501 once the server starts.
python -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit encountered an issue during execution.
)

echo.
pause
