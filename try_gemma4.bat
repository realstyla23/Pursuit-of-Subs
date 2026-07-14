@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python subtranslate.py --mode polish --input-dir . --polish-model gemma4:e4b --polish-parallel 1
pause
