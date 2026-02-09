@echo off
cd /d "%~dp0"
echo ==========================================
echo Verifying Azure Blob Size...
echo ==========================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\verify_upload.ps1"
pause
