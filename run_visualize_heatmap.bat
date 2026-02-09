@echo off
set "VENV_PYTHON=.\venv_model\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment python not found at: %VENV_PYTHON%
    echo Please make sure you are in the BatterySample folder and venv_model exists.
    pause
    exit /b
)

echo [*] Running Heatmap Visualization using venv_model...
"%VENV_PYTHON%" visualize_heatmap.py %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Error occurred.
    pause
) else (
    echo.
    echo [*] Done.
    pause
)
