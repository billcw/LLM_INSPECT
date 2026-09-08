@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Set up .venv using README.md first. Use Python 3.11 or 3.12.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
pause
