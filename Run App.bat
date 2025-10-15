@echo off
setlocal enableextensions

REM Launch the PowerShell bootstrapper with bypassed execution policy (no admin required)
powershell -NoProfile -ExecutionPolicy Bypass -File "run-windows.ps1"

pause
endlocal

