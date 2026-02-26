@echo off
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run setup_env.bat first.
    pause
    exit /b 1
)
python src/visualizador_comp.py
pause
