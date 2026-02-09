@echo off
cd /d "%~dp0"
echo ========================================================
echo [CV-1 Gate Model Inference]
echo Drop an image file here to check for defects!
echo ========================================================
call venv_model\Scripts\python run_inference.py %1
pause
