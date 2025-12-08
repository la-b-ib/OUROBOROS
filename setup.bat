@echo off
echo ╔═══════════════════════════════════════╗
echo ║   OUROBOROS Setup Script              ║
echo ║   Topological Malware Analysis        ║
echo ╚═══════════════════════════════════════╝
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [5/6] Installing dependencies...
echo This may take several minutes (GUDHI compilation requires time)...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies.
    echo Common issues:
    echo   - GUDHI requires Visual Studio Build Tools
    echo   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    pause
    exit /b 1
)

REM Create test samples
echo.
echo [6/6] Creating test samples...
python create_test_samples.py

echo.
echo ✅ Setup complete!
echo.
echo To run OUROBOROS:
echo   1. Activate environment: venv\Scripts\activate.bat
echo   2. Run: streamlit run app.py
echo.
echo Test samples created in: test_samples\
echo.
pause
