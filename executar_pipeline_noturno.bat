@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo =====================================================================
echo   Iniciando Pipeline Noturno Automatizado de Treinamento - LIBRAS
echo =====================================================================
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

"%PYTHON_EXEC%" scripts\run_overnight_pipeline.py %*
pause
