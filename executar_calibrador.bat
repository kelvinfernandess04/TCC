@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ===================================================
echo   Iniciando Calibrador Biomecanico Guiado - LIBRAS
echo ===================================================
python scripts\guided_hand_calibrator.py %*
pause
