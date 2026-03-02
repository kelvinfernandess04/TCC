@echo off
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[datetime]::UtcNow.Ticks"`) do set START_TIME=%%i
echo ==================================================
echo       Setting up Python Environment (TCC)
echo ==================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python (tested with 3.13) and try again.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)

:: Upgrade pip and install dependencies
echo [INFO] Installing dependencies from requirements.txt...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo [SUCCESS] Environment setup complete!
echo ==================================================
echo.
echo You can now use the run_*.bat scripts to launch the tools.
for /f "usebackq" %%i in (`powershell -NoProfile -Command "[datetime]::UtcNow.Ticks"`) do set END_TIME=%%i
powershell -NoProfile -Command "$secs = [math]::Round(((%END_TIME% - %START_TIME%) / 10000000), 2); Write-Host \"`n[Timer] Tempo total de execucao de setup: $secs segundos\""
pause
