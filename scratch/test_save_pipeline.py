import sys
import os
import json
import numpy as np

sys.path.insert(0, 'Treinamento IA/scripts')
from kinematic_seed_generator import HandKinematicsDirect

print("Testing HandKinematicsDirect instantiation from calibration_settings.json...")
gen = HandKinematicsDirect.from_calibration_file("Treinamento IA/data/calibration_settings.json")
print("HandKinematicsDirect loaded successfully!")
