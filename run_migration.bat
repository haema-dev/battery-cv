@echo off
cd /d "%~dp0"
echo ==========================================
echo Starting Migration using AzCopy...
echo ==========================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\migrate_with_azcopy.ps1"
echo.
echo Migration Finished.
pause
