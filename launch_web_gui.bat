@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python subtranslate.py --web-gui
pause
