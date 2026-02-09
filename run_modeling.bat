@echo off
cd /d "%~dp0"

echo =========================================
echo [Step 1] Organizing Dataset Folder...
echo =========================================
:: Use the NEW venv to avoid conflicts
call venv_model\Scripts\python organize_dataset.py

echo.
echo =========================================
echo [Step 2] Training CV-1 Gate Model...
echo =========================================
call venv_model\Scripts\python train_cv1_gate.py

pause
