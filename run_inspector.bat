@echo off
cd /d "%~dp0"
echo ========================================================
echo [Archive Inspector]
echo Use this to extract a small sample from your 97GB file
echo to verify contents without extracting everything.
echo ========================================================
python inspect_archive.py
pause
