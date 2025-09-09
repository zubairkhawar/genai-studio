@echo off
setlocal enabledelayedexpansion

REM GenAI Media Studio Setup Script for Windows
REM Cross-platform setup for Windows

title GenAI Media Studio Setup

REM Colors (using Windows color codes)
set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "RED=[91m"
set "PURPLE=[95m"
set "CYAN=[96m"
set "NC=[0m"

REM ASCII Art
echo.
echo %CYAN%╔══════════════════════════════════════════════════════════════╗%NC%
echo %CYAN%║                                                              ║%NC%
echo %CYAN%║              🎬 GenAI Media Studio Setup 🎵                  ║%NC%
echo %CYAN%║                                                              ║%NC%
echo %CYAN%║         AI-Powered Text-to-Video ^& Audio Generation          ║%NC%
echo %CYAN%║                                                              ║%NC%
echo %CYAN%╚══════════════════════════════════════════════════════════════╝%NC%
echo.

REM Function to print colored output
:print_status
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    call :print_warning "Running as administrator. This is not recommended for security reasons."
    pause
)

REM Check system requirements
call :print_status "Checking system requirements..."

REM Check Python version
python --version >nul 2>&1
if %errorLevel% neq 0 (
    call :print_error "Python not found. Please install Python 3.8+ from https://python.org"
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
call :print_success "Python %PYTHON_VERSION% found"

REM Check Node.js version
node --version >nul 2>&1
if %errorLevel% neq 0 (
    call :print_error "Node.js not found. Please install Node.js 18+ from https://nodejs.org"
    pause
    exit /b 1
)

for /f %%i in ('node --version') do set NODE_VERSION=%%i
call :print_success "Node.js %NODE_VERSION% found"

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if %errorLevel% neq 0 (
    call :print_warning "FFmpeg not found. Please install FFmpeg from https://ffmpeg.org"
    call :print_warning "Or use chocolatey: choco install ffmpeg"
    call :print_warning "Or use winget: winget install FFmpeg"
    pause
) else (
    call :print_success "FFmpeg found"
)

REM Check GPU support
call :print_status "Checking GPU support..."

nvidia-smi >nul 2>&1
if %errorLevel% == 0 (
    call :print_success "NVIDIA GPU detected"
    set GPU_TYPE=nvidia
) else (
    call :print_warning "No NVIDIA GPU detected"
    set GPU_TYPE=cpu
)

REM Create necessary directories
call :print_status "Creating directories..."
if not exist "outputs" mkdir outputs
if not exist "outputs\videos" mkdir outputs\videos
if not exist "outputs\audio" mkdir outputs\audio
if not exist "models" mkdir models
if not exist "logs" mkdir logs

REM Install Python dependencies
call :print_status "Installing Python dependencies..."
cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    call :print_status "Creating Python virtual environment..."
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
call :print_status "Installing Python packages..."
pip install -r requirements.txt

REM Install PyTorch with appropriate backend
if "%GPU_TYPE%"=="nvidia" (
    call :print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    call :print_status "Installing PyTorch with CPU support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

call venv\Scripts\deactivate.bat
cd ..

REM Install Node.js dependencies
call :print_status "Installing Node.js dependencies..."
cd frontend
call npm install
cd ..

REM Install root dependencies
call :print_status "Installing root dependencies..."
call npm install

REM Create environment file
call :print_status "Creating environment configuration..."
(
echo # GenAI Media Studio Configuration
echo NODE_ENV=development
echo BACKEND_PORT=8000
echo FRONTEND_PORT=3000
echo OUTPUT_DIR=outputs
echo MODELS_DIR=models
echo GPU_TYPE=%GPU_TYPE%
echo.
echo # Optional: Set custom model cache directory
echo # HF_HOME=./models/huggingface
echo # TRANSFORMERS_CACHE=./models/transformers
) > .env

REM Create startup script
call :print_status "Creating startup script..."
(
echo @echo off
echo setlocal enabledelayedexpansion
echo.
echo REM GenAI Media Studio Startup Script
echo.
echo echo Starting GenAI Media Studio...
echo.
echo REM Check if virtual environment exists
echo if not exist "backend\venv" ^(
echo     echo Virtual environment not found. Please run setup.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Start backend in background
echo echo Starting backend server...
echo cd backend
echo call venv\Scripts\activate.bat
echo start /b python main.py
echo cd ..
echo.
echo REM Wait a moment for backend to start
echo timeout /t 3 /nobreak ^>nul
echo.
echo REM Start frontend
echo echo Starting frontend server...
echo cd frontend
echo start /b npm run dev
echo cd ..
echo.
echo echo GenAI Media Studio is running!
echo echo Frontend: http://localhost:3000
echo echo Backend API: http://localhost:8000
echo echo.
echo echo Press any key to stop both servers
echo pause ^>nul
echo.
echo REM Stop processes
echo echo Stopping servers...
echo taskkill /f /im python.exe 2^>nul
echo taskkill /f /im node.exe 2^>nul
echo echo Servers stopped.
) > start.bat

REM Create stop script
call :print_status "Creating stop script..."
(
echo @echo off
echo echo Stopping GenAI Media Studio...
echo taskkill /f /im python.exe 2^>nul
echo taskkill /f /im node.exe 2^>nul
echo echo GenAI Media Studio stopped.
) > stop.bat

REM Create update script
call :print_status "Creating update script..."
(
echo @echo off
echo echo Updating GenAI Media Studio...
echo.
echo REM Update Python dependencies
echo cd backend
echo call venv\Scripts\activate.bat
echo pip install --upgrade -r requirements.txt
echo call venv\Scripts\deactivate.bat
echo cd ..
echo.
echo REM Update Node.js dependencies
echo cd frontend
echo call npm update
echo cd ..
echo.
echo REM Update root dependencies
echo call npm update
echo.
echo echo Update complete!
) > update.bat

REM Create desktop shortcut
call :print_status "Creating desktop shortcut..."
set "DESKTOP=%USERPROFILE%\Desktop"
set "CURRENT_DIR=%CD%"

powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\GenAI Media Studio.lnk'); $Shortcut.TargetPath = '%CURRENT_DIR%\start.bat'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.Description = 'GenAI Media Studio - AI-Powered Text-to-Video & Audio Generation'; $Shortcut.Save()}"

REM Summary
echo.
echo %GREEN%╔══════════════════════════════════════════════════════════════╗%NC%
echo %GREEN%║                    Setup Complete! 🎉                      ║%NC%
echo %GREEN%╚══════════════════════════════════════════════════════════════╝%NC%
echo.
call :print_success "GenAI Media Studio has been successfully installed!"
echo.
echo %CYAN%Quick Start:%NC%
echo   %YELLOW%start.bat%NC%     - Start the application
echo   %YELLOW%stop.bat%NC%      - Stop the application
echo   %YELLOW%update.bat%NC%    - Update dependencies
echo.
echo %CYAN%Access URLs:%NC%
echo   %BLUE%Frontend:%NC% http://localhost:3000
echo   %BLUE%Backend API:%NC% http://localhost:8000
echo   %BLUE%API Docs:%NC% http://localhost:8000/docs
echo.
echo %CYAN%GPU Support:%NC% %GPU_TYPE%
echo %CYAN%Output Directory:%NC% .\outputs\
echo %CYAN%Models Directory:%NC% .\models\
echo.
call :print_warning "Note: First-time model downloads may take several minutes."
call :print_warning "Make sure you have a stable internet connection."
echo.
echo %PURPLE%Happy creating! 🎬🎵%NC%
echo.
pause