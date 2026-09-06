#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atalho de Execução: Studio de Visualização 3D de Calibração LIBRAS
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPTS = os.path.join(BASE_DIR, "Treinamento IA", "scripts")
if TRAIN_SCRIPTS not in sys.path:
    sys.path.insert(0, TRAIN_SCRIPTS)

from visualizador_calibracao import Calibration3DVisualizer

if __name__ == "__main__":
    app = Calibration3DVisualizer()
    app.run()
