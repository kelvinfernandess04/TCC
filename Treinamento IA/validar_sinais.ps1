# Script PowerShell para iniciar a validacao de sinais LIBRAS
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir "..\.venv\Scripts\python.exe"
$ValidateScript = Join-Path $ScriptDir "scripts\validate_signs_live.py"

& $VenvPython $ValidateScript $args
