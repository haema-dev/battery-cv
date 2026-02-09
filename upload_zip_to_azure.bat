@echo off
setlocal enabledelayedexpansion

:: ==========================================
:: AZURE UPLOAD HELPER (Using AzCopy)
:: ==========================================

:: 1. Locate AzCopy
set "AZCOPY_TOOL=%~dp0azcopy.exe"
if not exist "!AZCOPY_TOOL!" (
    echo [ERROR] azcopy.exe not found in this folder!
    echo Please make sure azcopy.exe is next to this script.
    pause
    exit /b
)

echo ========================================================
echo        Azure Blob Storage Upload (Zip Files)
echo ========================================================
echo.

:: 2. Input Source Directory
echo [Step 1] Drag and Drop the folder containing ZIP files here, or paste the path.
echo (The folder with: TS_Exterior...zip, VS_Exterior...zip)
set /p "SOURCE_DIR=Path:> "
:: Remove quotes if present
set "SOURCE_DIR=!SOURCE_DIR:"=!"

if not exist "!SOURCE_DIR!" (
    echo [ERROR] Directory not found: !SOURCE_DIR!
    pause
    exit /b
)

echo.
echo [Step 2] Paste your Azure Container SAS URL.
echo (Format: https://<account>.blob.core.windows.net/<container>?<token>)
set /p "SAS_URL=SAS URL:> "

:: 3. Execute Upload
echo.
echo [*] Checking files in !SOURCE_DIR!...
echo.
echo [*] DEBUG: Searching for ZIP files in: !SOURCE_DIR!
dir /s /b "!SOURCE_DIR!\*.zip"
echo.
echo [?] Do you see your 97GB and 2GB files in the list above?
echo     If NO, you might have selected the wrong folder.
echo     If YES, press any key to start uploading...
pause

:: [NETWORKING TWEAK]
set AZCOPY_CONCURRENCY_VALUE=AUTO
set AZCOPY_JOB_PLAN_LOCATION=%TEMP%

:RETRY_LOOP
echo.
echo ========================================================
echo   Starting / Resuming Upload (Auto-Retry Mode)
echo ========================================================
echo.

"!AZCOPY_TOOL!" copy "!SOURCE_DIR!" "!SAS_URL!" --recursive=true --include-pattern "*.zip" --overwrite=ifSourceNewer --put-md5 --block-size-mb 100

if %errorlevel% neq 0 (
    echo.
    echo [!] Network Error Detected.
    echo [!] Waiting 10 seconds before AUTO-RETRY...
    timeout /t 10
    goto RETRY_LOOP
) else (
    echo.
    echo [SUCCESS] Upload Complete!
)

pause
