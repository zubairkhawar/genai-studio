# Windows Setup Instructions

## Quick Start for Windows

### Option 1: Batch Files (Recommended)
1. **Double-click** `setup-amd.bat`
2. **Wait for setup to complete** (it will show progress)
3. **Double-click** `start-amd.bat`
4. **Open browser** to http://localhost:3000

### Option 2: PowerShell (Alternative)
1. **Right-click** `setup-amd.ps1` → "Run with PowerShell"
2. **Wait for setup to complete**
3. **Right-click** `start-amd.ps1` → "Run with PowerShell"
4. **Open browser** to http://localhost:3000

## What the setup does:
- ✅ Checks if Python and Node.js are installed
- ✅ Creates Python virtual environment
- ✅ Installs all required dependencies
- ✅ Sets up PyTorch for AMD GPU
- ✅ Installs frontend dependencies

## If you get errors:
1. **Make sure Python 3.8+ is installed**: https://www.python.org/downloads/
2. **Make sure Node.js 16+ is installed**: https://nodejs.org/
3. **Run as Administrator** if needed
4. **Check Windows Defender** isn't blocking the scripts

## After setup:
- The app will be running at http://localhost:3000
- Download models from the Settings page when ready
- No models are downloaded during setup (keeps it fast)

## Troubleshooting:
- If scripts close immediately, run them from Command Prompt
- If PowerShell scripts don't run, enable execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

