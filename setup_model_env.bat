@echo off
cd /d "%~dp0"
echo ========================================================
echo [Setup] Creating Clean Environment for AI Modeling
echo ========================================================

:: 1. Create new venv to avoid old conflicts
if not exist "venv_model" (
    echo [*] Creating virtual environment 'venv_model'...
    python -m venv venv_model
)

:: 2. Upgrade pip
echo [*] Upgrading pip...
call venv_model\Scripts\python -m pip install --upgrade pip

:: 3. Install Anomalib & Torch (CPU/CUDA auto)
echo [*] Installing Anomalib (This may take a few minutes)...
:: Installing minimal dependencies for anomalies (Patchcore/Padim)
call venv_model\Scripts\pip install anomalib[full] torch torchvision

echo ========================================================
echo [Setup] Complete!
echo You can now run 'run_training.bat'
echo ========================================================
pause
