param(
    [switch]$Force,
    [switch]$Start
)

# Fail fast on errors
$ErrorActionPreference = 'Stop'

function Write-Section($message) {
    Write-Host "`n=== $message ===" -ForegroundColor Cyan
}

function Assert-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found. $hint"
    }
}

function Ensure-Directory($path) {
    if (-not (Test-Path $path)) { [void](New-Item -ItemType Directory -Path $path) }
}

function Try-Ensure-FFmpeg() {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { return }
    Write-Host "ffmpeg not found on PATH." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Attempting to install ffmpeg via winget..." -ForegroundColor Yellow
        try {
            winget install --id Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements | Out-Null
        } catch { }
        if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Host "winget install did not make ffmpeg available immediately; you may need to open a new shell or add it to PATH." -ForegroundColor Yellow
        }
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "Attempting to install ffmpeg via choco..." -ForegroundColor Yellow
        try {
            choco install ffmpeg -y | Out-Null
        } catch { }
    } else {
        Write-Host "Install ffmpeg from https://ffmpeg.org/download.html or via winget/choco and re-run if needed." -ForegroundColor Yellow
    }
}

function Run($cmd, $workDir) {
    if ($workDir) { Push-Location $workDir }
    try {
        Write-Host "> $cmd" -ForegroundColor DarkGray
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName  = $env:COMSPEC
        $psi.Arguments = "/c $cmd"
        $psi.RedirectStandardOutput = $false
        $psi.RedirectStandardError  = $false
        $psi.UseShellExecute = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { throw "Command failed ($($p.ExitCode)): $cmd" }
    }
    finally {
        if ($workDir) { Pop-Location }
    }
}

# Repo root guard
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot
if (-not (Test-Path "$repoRoot/backend") -or -not (Test-Path "$repoRoot/frontend")) {
    throw "Please run this script from the repository root."
}

Write-Section "Preflight checks"
Assert-Command "python" "Install Python 3.11+ from Microsoft Store or python.org and ensure it's on PATH."
Assert-Command "node"   "Install Node.js 18+ from nodejs.org and ensure it's on PATH."
Assert-Command "npm"    "Install Node.js which includes npm."
Try-Ensure-FFmpeg

# Detect Python version
$pyVersionOut = (& python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
$pyMajMin = $pyVersionOut.Split('.')[0..1] -join '.'
if ([version]$pyMajMin -lt [version]"3.11") {
    throw "Python >= 3.11 is required. Found $pyVersionOut"
}

# Ensure common directories
Write-Section "Preparing directories"
Ensure-Directory (Join-Path $repoRoot 'models')
Ensure-Directory (Join-Path $repoRoot 'outputs')
Ensure-Directory (Join-Path $repoRoot 'outputs/videos')
Ensure-Directory (Join-Path $repoRoot 'outputs/audio')
Ensure-Directory (Join-Path $repoRoot 'outputs/voice-previews')

# ------------------ Backend ------------------
Write-Section "Setting up backend virtual environment"
$backendDir = Join-Path $repoRoot 'backend'
$venvPath = Join-Path $backendDir 'venv'

if (Test-Path $venvPath -and $Force) {
    Write-Host "Removing existing venv because -Force was provided..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Run "python -m venv `"$venvPath`"" $repoRoot
}

# Use venv's pip
$venvPython = Join-Path $venvPath 'Scripts/python.exe'
$venvPip    = Join-Path $venvPath 'Scripts/pip.exe'
if (-not (Test-Path $venvPython)) { throw "Virtual environment not created correctly at $venvPath" }

Write-Section "Installing backend requirements"
Run "`"$venvPip`" install --upgrade pip wheel setuptools" $repoRoot
Run "`"$venvPip`" install -r `"$backendDir/requirements.txt`"" $repoRoot

## Intentionally not downloading models here; downloads are triggered from the frontend

# ------------------ Frontend ------------------
Write-Section "Installing frontend dependencies"
$frontendDir = Join-Path $repoRoot 'frontend'
Run "npm ci" $frontendDir

Write-Section "Setup complete"
Write-Host "Backend venv: $venvPath" -ForegroundColor Green
Write-Host "To run backend: `n  $($venvPath)\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor Green
Write-Host "To run frontend: `n  cd frontend && npm run dev" -ForegroundColor Green

if ($Start) {
    Write-Section "Starting backend and frontend"
    $uvicornExe = Join-Path $venvPath 'Scripts/uvicorn.exe'
    if (-not (Test-Path $uvicornExe)) {
        # Fallback to python -m uvicorn if direct exe not found
        $uvicornCmd = "`"$venvPython`" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
        Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location `'$repoRoot`'; $uvicornCmd"
    } else {
        Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location `'$repoRoot`'; & `'$uvicornExe`' backend.main:app --host 0.0.0.0 --port 8000 --reload"
    }

    # Start frontend in separate window
    Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location `'$frontendDir`'; npm run dev"
    Write-Host "Backend and frontend launched in separate terminals." -ForegroundColor Green
}
