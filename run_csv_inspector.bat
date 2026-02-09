@echo off
cd /d "%~dp0"
echo ========================================================
echo [CSV Inspector]
echo Use this to check your label CSV file.
echo ========================================================
:: Use venv_model python which needs pandas. 
:: If pandas is missing, we install it first.
call venv_model\Scripts\pip install pandas
call venv_model\Scripts\python inspect_csv.py
