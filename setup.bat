@echo off
setlocal enabledelayedexpansion

REM GenAI Media Studio - One-Click Setup Script for Windows
REM Supports Windows 10/11 with WSL or native Windows

echo.
echo   ╔══════════════════════════════════════════════════════════════╗
echo   ║                                                              ║
echo   ║    🎬 GenAI Media Studio - One-Click Setup 🎵              ║
echo   ║                                                              ║
echo   ║    Transform your ideas into reality with AI magic! ✨      ║
echo   ║                                                              ║
echo   ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+ first.
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found! Please install Node.js 18+ first.
    echo Visit: https://nodejs.org/
    pause
    exit /b 1
)

echo [INFO] Starting GenAI Media Studio setup...

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "outputs\videos" mkdir "outputs\videos"
if not exist "outputs\audio" mkdir "outputs\audio"
if not exist "models" mkdir "models"
echo [SUCCESS] Directories created!

REM Install Python dependencies
echo [INFO] Installing Python dependencies...
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r backend\requirements.txt
echo [SUCCESS] Python dependencies installed!

REM Install Node.js dependencies
echo [INFO] Installing Node.js dependencies...
cd frontend
call npm install
cd ..
echo [SUCCESS] Node.js dependencies installed!

REM Check for FFmpeg
echo [INFO] Checking for FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg not found! Please install FFmpeg manually:
    echo Visit: https://ffmpeg.org/download.html
) else (
    echo [SUCCESS] FFmpeg found!
)

REM Check GPU support
echo [INFO] Checking GPU support...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] NVIDIA GPU detected!
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
) else (
    echo [WARNING] No NVIDIA GPU detected. The application will run on CPU.
)

REM Create startup scripts
echo [INFO] Creating startup scripts...

REM Backend startup script
echo @echo off > start_backend.bat
echo echo 🚀 Starting GenAI Media Studio Backend... >> start_backend.bat
echo cd backend >> start_backend.bat
echo call ..\venv\Scripts\activate.bat >> start_backend.bat
echo python main.py >> start_backend.bat
echo pause >> start_backend.bat

REM Frontend startup script
echo @echo off > start_frontend.bat
echo echo 🎨 Starting GenAI Media Studio Frontend... >> start_frontend.bat
echo cd frontend >> start_frontend.bat
echo npm run dev >> start_frontend.bat
echo pause >> start_frontend.bat

REM Combined startup script
echo @echo off > start_app.bat
echo echo 🎬 Starting GenAI Media Studio... >> start_app.bat
echo echo Backend will run on http://localhost:8000 >> start_app.bat
echo echo Frontend will run on http://localhost:3000 >> start_app.bat
echo echo. >> start_app.bat
echo echo Starting backend in background... >> start_app.bat
echo start /B cmd /c "cd backend && call ..\venv\Scripts\activate.bat && python main.py" >> start_app.bat
echo timeout /t 3 /nobreak ^>nul >> start_app.bat
echo echo Starting frontend... >> start_app.bat
echo start /B cmd /c "cd frontend && npm run dev" >> start_app.bat
echo echo Application started! Press any key to stop. >> start_app.bat
echo pause >> start_app.bat

echo [SUCCESS] Startup scripts created!

REM Final message
echo.
echo   ╔══════════════════════════════════════════════════════════════╗
echo   ║                                                              ║
echo   ║    🎉 Setup Complete! GenAI Media Studio is ready! 🎉     ║
echo   ║                                                              ║
echo   ╚══════════════════════════════════════════════════════════════╝
echo.
echo [SUCCESS] Installation completed successfully!
echo.
echo [INFO] To start the application:
echo   start_app.bat     - Start both backend and frontend
echo   start_backend.bat - Start backend only
echo   start_frontend.bat - Start frontend only
echo.
echo [INFO] Access the application at:
echo   Frontend: http://localhost:3000
echo   Backend API: http://localhost:8000
echo.
echo [INFO] For more information, check the README.md file.
echo.
pause
