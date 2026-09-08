@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo =========================================================
echo   Iniciando Calibrador Guiado do Polegar - LIBRAS TCC
echo =========================================================
python scripts\guided_thumb_calibrator.py %*
pause
