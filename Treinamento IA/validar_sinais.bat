@echo off
title Validador de Sinais LIBRAS (IA Biomecanica)
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" "scripts\validate_signs_live.py" %*
pause
