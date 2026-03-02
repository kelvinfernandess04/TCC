@echo off
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run setup_env.bat first.
    pause
    exit /b 1
)
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[datetime]::UtcNow.Ticks"`) do set START_TIME=%%i
"venv\Scripts\python.exe" src/normalizador.py
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[datetime]::UtcNow.Ticks"`) do set END_TIME=%%i
powershell -NoProfile -Command "$secs = [math]::Round(((%END_TIME% - %START_TIME%) / 10000000), 2); Write-Host \"`n[Timer] Tempo de execucao: $secs segundos\""
pause
