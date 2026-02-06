@echo off
REM Flask Deployment Script for Windows

echo 🚀 Starting MNIST GAN Flask Deployment...

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check if model checkpoint exists
if not exist "checkpoints\best_model.pth" (
    echo ⚠️  Warning: No trained model found at checkpoints\best_model.pth
    echo    The app will start with an untrained model.
    echo    Please train the model first using: python scripts\train.py
)

REM Create necessary directories
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs
if not exist "templates" mkdir templates
if not exist "static" mkdir static

REM Start Flask app
echo 🌐 Starting Flask development server...
python app.py

pause
