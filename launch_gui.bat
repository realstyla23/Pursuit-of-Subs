@echo off
cd /d "%~dp0"
".\venv\Scripts\python.exe" subtranslate.py --gui
pause
