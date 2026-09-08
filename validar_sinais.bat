@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ===============================================================================
echo   Validador Biomecanico de Sinais LIBRAS (IA Desktop)
echo ===============================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "Treinamento IA\scripts\validate_signs_live.py" %*
) else (
    python "Treinamento IA\scripts\validate_signs_live.py" %*
)
pause
