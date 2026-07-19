@echo off
cd /d "%~dp0"
".\venv\Scripts\python.exe" subtranslate.py --mode polish --input-dir . --polish-model qwen2.5:7b --polish-parallel 2
pause
