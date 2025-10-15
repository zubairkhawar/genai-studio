Param(
    [switch]$NoLaunch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Test-Command($name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Ensure-Tls12 {
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch {}
}

function Download-File($Url, $OutPath) {
    Ensure-Tls12
    Write-Info "Downloading: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $OutPath -UseBasicParsing
}

function Ensure-Directory($Path) {
    if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}

function Test-PyImport($PythonExe, $ModuleName) {
    try {
        $proc = Start-Process -FilePath $PythonExe -ArgumentList @('-c', "import $ModuleName") -PassThru -Wait -NoNewWindow -WindowStyle Hidden
        return ($proc.ExitCode -eq 0)
    } catch {
        return $false
    }
}

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path $ScriptRoot
Set-Location $ProjectRoot

$ToolsDir = Join-Path $ProjectRoot 'tools'
Ensure-Directory $ToolsDir

# 1) Ensure Python (user install, quiet)
$pythonCmd = $null
if (Test-Command 'py') { $pythonCmd = 'py' }
elseif (Test-Command 'python') { $pythonCmd = 'python' }
else {
    Write-Info 'Python not found. Installing Python 3.11 (user mode)...'
    $pyVer = '3.11.9'
    $pyUrl = "https://www.python.org/ftp/python/$pyVer/python-$pyVer-amd64.exe"
    $pyInstaller = Join-Path $ToolsDir "python-$pyVer-amd64.exe"
    Download-File $pyUrl $pyInstaller
    $pyArgs = '/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 SimpleInstall=1'
    Start-Process -FilePath $pyInstaller -ArgumentList $pyArgs -Wait -NoNewWindow
    if (Test-Command 'py') { $pythonCmd = 'py' }
    elseif (Test-Command 'python') { $pythonCmd = 'python' }
    else { throw 'Python installation failed.' }
}

# Resolve python executable for venv creation
try {
    if ($pythonCmd -eq 'py') { $pythonExe = & py -3.11 -c "import sys;print(sys.executable)" }
    else { $pythonExe = & $pythonCmd -c "import sys;print(sys.executable)" }
} catch { $pythonExe = $pythonCmd }

# 2) Ensure Node.js (user install, quiet)
if (-not (Test-Command 'node')) {
    Write-Info 'Node.js not found. Installing Node.js 20 LTS (user mode)...'
    $nodeVer = '20.17.0'
    $nodeUrl = "https://nodejs.org/dist/v$nodeVer/node-v$nodeVer-x64.msi"
    $nodeMsi = Join-Path $ToolsDir "node-v$nodeVer-x64.msi"
    Download-File $nodeUrl $nodeMsi
    # Per-user install (no admin): ALLUSERS=0
    Start-Process msiexec.exe -ArgumentList "/i `"$nodeMsi`" /qn ALLUSERS=0" -Wait -NoNewWindow
    if (-not (Test-Command 'node')) { throw 'Node.js installation failed.' }
}

# 3) Ensure FFmpeg (portable)
$ffmpegBin = $null
try {
    if (Test-Command 'ffmpeg') { $ffmpegBin = (Get-Command ffmpeg).Source }
} catch {}
if (-not $ffmpegBin) {
    Write-Info 'FFmpeg not found. Downloading portable FFmpeg...'
    $ffmpegZip = Join-Path $ToolsDir 'ffmpeg-latest-win64.zip'
    # BtbN nightly static build (latest release zip)
    $ffmpegUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-lgpl.zip'
    Download-File $ffmpegUrl $ffmpegZip
    $ffmpegDir = Join-Path $ToolsDir 'ffmpeg'
    Ensure-Directory $ffmpegDir
    # Clear previous
    if (Test-Path $ffmpegDir) { Remove-Item $ffmpegDir -Recurse -Force -ErrorAction SilentlyContinue | Out-Null }
    Ensure-Directory $ffmpegDir
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir -Force
    # Find bin folder
    $binPath = Get-ChildItem -Path $ffmpegDir -Recurse -Directory | Where-Object { $_.Name -eq 'bin' } | Select-Object -First 1
    if ($null -eq $binPath) { throw 'FFmpeg bin directory not found after extraction.' }
    $env:PATH = "$($binPath.FullName);$env:PATH"
    Write-Info "FFmpeg ready at: $($binPath.FullName)"
}

# 4) Python venv + deps
$BackendDir = Join-Path $ProjectRoot 'backend'
$VenvDir = Join-Path $BackendDir '.venv'
if (-not (Test-Path $VenvDir)) {
    Write-Info 'Creating Python virtual environment...'
    & $pythonExe -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
$VenvPip = Join-Path $VenvDir 'Scripts/pip.exe'

Write-Info 'Upgrading pip...'
& $VenvPython -m pip install -U pip setuptools wheel

# Install PyTorch CPU wheels explicitly first to avoid CUDA/ROCm issues on Windows AMD
Write-Info 'Installing PyTorch (CPU wheels)...'
& $VenvPip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Prepare filtered requirements (exclude torch/torchvision/torchaudio, xformers, and flash-attn)
$ReqPath = Join-Path $BackendDir 'requirements.txt'
$TempReq = Join-Path $BackendDir 'requirements.filtered.txt'
$lines = Get-Content $ReqPath
$filtered = $lines | Where-Object { 
    $_ -notmatch '^(torch|torchvision|torchaudio)\b' -and 
    $_ -notmatch '^xformers\b' -and 
    $_ -notmatch '^flash-attn\b' -and
    $_ -notmatch '^flash_attn\b'
}
# Relax pynvml pin for wider Windows/Python compatibility
$filtered = $filtered | ForEach-Object { if ($_ -match '^pynvml') { 'pynvml' } else { $_ } }
Set-Content -Path $TempReq -Value $filtered -Encoding UTF8

Write-Info 'Installing backend Python dependencies...'
& $VenvPip install -r $TempReq

Write-Info 'Removing xformers to prevent compatibility issues...'
try {
    & $VenvPip uninstall xformers -y
    Write-Info 'xformers removed successfully'
} catch {
    Write-Info 'xformers was not installed or already removed'
}

# Also remove flash-attn which can cause similar issues
try {
    & $VenvPip uninstall flash-attn -y
    Write-Info 'flash-attn removed successfully'
} catch {
    Write-Info 'flash-attn was not installed'
}

# Ensure critical backend packages exist
Write-Info 'Verifying critical backend packages in venv...'
$criticalPackages = @(
    @{Name='fastapi'; Module='fastapi'; Package='fastapi'},
    @{Name='uvicorn'; Module='uvicorn'; Package='uvicorn[standard]'},
    @{Name='opencv-python'; Module='cv2'; Package='opencv-python>=4.8.0'},
    @{Name='diffusers'; Module='diffusers'; Package='diffusers>=0.21.0'},
    @{Name='soundfile'; Module='soundfile'; Package='soundfile>=0.12.0'},
    @{Name='librosa'; Module='librosa'; Package='librosa>=0.10.0'},
    @{Name='numpy'; Module='numpy'; Package='numpy>=1.24.0'},
    @{Name='Pillow'; Module='PIL'; Package='Pillow>=10.0.0'},
    @{Name='transformers'; Module='transformers'; Package='transformers>=4.35.0'},
    @{Name='ffmpeg-python'; Module='ffmpeg'; Package='ffmpeg-python>=0.2.0'},
    @{Name='accelerate'; Module='accelerate'; Package='accelerate>=0.24.0'},
    @{Name='scipy'; Module='scipy'; Package='scipy>=1.11.0'},
    @{Name='pydantic'; Module='pydantic'; Package='pydantic>=2.5.0'},
    @{Name='python-dotenv'; Module='dotenv'; Package='python-dotenv>=1.0.0'},
    @{Name='aiofiles'; Module='aiofiles'; Package='aiofiles>=23.2.0'},
    @{Name='python-multipart'; Module='multipart'; Package='python-multipart>=0.0.6'}
)

foreach ($pkg in $criticalPackages) {
    $isOk = Test-PyImport -PythonExe $VenvPython -ModuleName $pkg.Module
    if (-not $isOk) {
        Write-Info "Installing $($pkg.Name)..."
        & $VenvPip install -U $pkg.Package
    }
}

# Final cleanup - ensure xformers and related packages are completely removed
Write-Info 'Final cleanup of problematic packages...'
$problematicPackages = @('xformers', 'flash-attn', 'flash_attn')
foreach ($pkg in $problematicPackages) {
    try {
        & $VenvPip uninstall $pkg -y
        Write-Info "$pkg removed during cleanup"
    } catch {
        Write-Info "$pkg was not installed"
    }
}

# Verify no problematic packages remain
Write-Info 'Verifying no problematic packages remain...'
try {
    $xformersTest = & $VenvPython -c "import xformers" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warn 'xformers still present - attempting force removal...'
        & $VenvPip uninstall xformers -y --break-system-packages
    }
} catch {
    Write-Info 'xformers verification completed - package not found'
}

# 5) Frontend deps
$FrontendDir = Join-Path $ProjectRoot 'frontend'
Write-Info 'Installing frontend dependencies...'
Set-Location $FrontendDir

# Handle npm permission issues by cleaning node_modules first if it exists
if (Test-Path 'node_modules') {
    Write-Info 'Cleaning existing node_modules to avoid permission issues...'
    try {
        Remove-Item 'node_modules' -Recurse -Force -ErrorAction SilentlyContinue
        # Also remove package-lock.json to ensure clean install
        if (Test-Path 'package-lock.json') {
            Remove-Item 'package-lock.json' -Force -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Warn 'Could not remove node_modules completely. This may cause permission issues.'
    }
}

# Try npm install with various fallbacks
try {
    & cmd /c "npm install --force"
} catch {
    Write-Warn 'npm install failed, trying with --legacy-peer-deps...'
    try {
        & cmd /c "npm install --legacy-peer-deps --force"
    } catch {
        Write-Warn 'npm install with --legacy-peer-deps also failed, trying --no-optional...'
        try {
            & cmd /c "npm install --no-optional --force"
        } catch {
            Write-Warn 'npm install with --no-optional also failed, trying with --no-fund...'
            try {
                & cmd /c "npm install --no-fund --force"
            } catch {
                Write-Err 'All npm install attempts failed. You may need to run as Administrator or check antivirus settings.'
            }
        }
    }
}

# Verify and fix SWC helpers if needed
Write-Info 'Checking for SWC helpers...'
$swcHelpersPath = 'node_modules\@swc\helpers'
if (-not (Test-Path $swcHelpersPath)) {
    Write-Info 'Installing SWC helpers...'
    try {
        & cmd /c "npm install @swc/helpers --force"
    } catch {
        Write-Warn 'Failed to install SWC helpers manually'
    }
}

Set-Location $ProjectRoot

# Verify Next.js is available
Write-Info 'Verifying Next.js installation...'
$nextOk = Test-Command 'next'
if (-not $nextOk) {
    # Check if next is in node_modules/.bin
    $nextBin = Join-Path $FrontendDir 'node_modules\.bin\next.cmd'
    if (Test-Path $nextBin) {
        Write-Info 'Next.js found in node_modules/.bin'
    } else {
        Write-Warn 'Next.js not found. Frontend may not work properly.'
    }
}

# 6) Launch services
if (-not $NoLaunch) {
    Write-Info 'Launching backend and frontend in separate windows...'

    # Ensure FFmpeg portable bin is in new processes PATH (prepend bin if we extracted)
    $ffmpegBinDir = ($env:PATH -split ';' | Where-Object { $_ -like '*\\ffmpeg*\\bin' } | Select-Object -First 1)

    # Build backend launch command safely using format strings to avoid quoting issues
    # Set environment variables to disable xformers and handle compatibility issues
    $envVars = '$env:XFORMERS_MORE_DETAILS="0"; $env:PYTORCH_ENABLE_MPS_FALLBACK="1"; $env:DIFFUSERS_USE_XFORMERS="0"'
    if ($ffmpegBinDir) {
        $backendCmd = '{0}; $env:PATH="{1};" + $env:PATH; cd "{2}"; & "{3}" -m uvicorn main:app --host 0.0.0.0 --port 8000' -f $envVars, $ffmpegBinDir, $BackendDir, $VenvPython
    } else {
        $backendCmd = '{0}; cd "{1}"; & "{2}" -m uvicorn main:app --host 0.0.0.0 --port 8000' -f $envVars, $BackendDir, $VenvPython
    }
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $backendCmd -WindowStyle Normal

    # Build frontend launch command
    # Launch frontend via cmd to avoid npm.ps1 wrapper issues
    $frontendCmd = 'cd /d "{0}" && npm run dev' -f $FrontendDir
    Start-Process cmd.exe -ArgumentList '/k', $frontendCmd -WindowStyle Normal

    Write-Info 'Backend: http://localhost:8000'
    Write-Info 'Frontend: http://localhost:3000'
} else {
    Write-Info 'Setup complete. Skipping launch due to -NoLaunch.'
}

Write-Info 'All done.'


