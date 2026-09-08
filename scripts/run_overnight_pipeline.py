#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper raiz para run_overnight_pipeline.py"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "Treinamento IA", "scripts")
if TARGET_DIR not in sys.path:
    sys.path.insert(0, TARGET_DIR)

from run_overnight_pipeline import run_pipeline

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
