@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo =====================================================================
echo   Iniciando Pipeline Noturno Automatizado de Treinamento - LIBRAS
echo =====================================================================
python scripts\run_overnight_pipeline.py %*
pause
