# Atalho PowerShell na raiz do projeto para o validador de sinais LIBRAS
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$ValidateScript = Join-Path $ScriptDir "Treinamento IA\scripts\validate_signs_live.py"

if (Test-Path $VenvPython) {
    & $VenvPython $ValidateScript $args
} else {
    & python $ValidateScript $args
}
