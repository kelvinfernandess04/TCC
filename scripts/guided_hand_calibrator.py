#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atalho raiz para o Calibrador Biomecânico Guiado da Mão (LIBRAS TCC)
==================================================================
Permite executar:
    python scripts/guided_hand_calibrator.py
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

# Configurar saída do console para UTF-8 no Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

# Auto-detecção: se o usuário chamou com o python global, re-executa automaticamente com o .venv
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    sys.exit(subprocess.call([venv_python] + sys.argv))

TARGET_DIR = os.path.join(BASE_DIR, "Treinamento IA", "scripts")

if TARGET_DIR not in sys.path:
    sys.path.insert(0, TARGET_DIR)

from guided_hand_calibrator import main

if __name__ == "__main__":
    main()

