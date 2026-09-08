#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper raiz para o inspect_seeds.py de Treinamento IA"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "Treinamento IA", "scripts")
if TARGET_DIR not in sys.path:
    sys.path.insert(0, TARGET_DIR)

from inspect_seeds import main

if __name__ == "__main__":
    main()
