@echo off
cd /d "%~dp0"
echo ========================================================
echo [Dataset Builder]
echo Using CSV to extract Clean/Defect images from Archive.
echo ========================================================
call venv_model\Scripts\python build_dataset_from_csv.py
pause
