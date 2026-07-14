@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python subtranslate.py --mode polish --input-dir . --polish-model qwen2.5:7b --polish-parallel 1
pause
