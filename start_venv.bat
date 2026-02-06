@echo off
echo 🚀 Starting MNIST GAN in Virtual Environment...

REM Check if venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found in .venv!
    echo    Please create it first: python -m venv .venv
    pause
    exit /b
)

REM Install dependencies (quietly)
echo 📦 Verifying dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

REM Start App
echo 🌐 Starting Flask Server...
".venv\Scripts\python.exe" app.py

pause
