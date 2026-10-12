@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" ui.py
) else (
    py -3 ui.py
)
if errorlevel 1 pause
