"""
Pose Verifier Live Wrapper
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPTS = os.path.join(BASE_DIR, "Treinamento IA", "scripts")
if TRAIN_SCRIPTS not in sys.path:
    sys.path.insert(0, TRAIN_SCRIPTS)

from pose_verifier_live import PoseVerifierLive, main

if __name__ == "__main__":
    main()
