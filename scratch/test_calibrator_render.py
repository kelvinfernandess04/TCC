import sys
import os
import cv2
import numpy as np

sys.path.insert(0, 'Treinamento IA/scripts')
from guided_thumb_calibrator import GuidedThumbCalibrator, THUMB_CALIBRATION_STEPS

calib = GuidedThumbCalibrator()

# Create dummy frame 1280x720
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
frame[:] = (30, 30, 40)

step = THUMB_CALIBRATION_STEPS[0]

# 1. Test Capturing HUD
calib.state = "CAPTURING"
calib.current_sub_angle = "FRONTAL"
hud_cap = calib._render_capturing_hud(frame.copy(), step, True, np.zeros((21, 3)))
cv2.imwrite("scratch/test_hud_cap.png", hud_cap)

# 2. Test Waiting Lateral HUD
hud_wait = calib._render_waiting_lateral_hud(frame.copy(), step)
cv2.imwrite("scratch/test_hud_wait.png", hud_wait)

# 3. Test Review HUD
calib.captured_data[step['id']] = {
    'step_meta': step,
    'frontal': {},
    'lateral': {},
    'pts_norm': np.zeros((21, 3)),
    'snapshot': frame.copy()
}
calib.current_review_metrics, calib.current_review_status = calib._format_thumb_metrics(step, np.zeros((21, 3)))
hud_rev = calib._render_review_hud(frame.copy(), step)
cv2.imwrite("scratch/test_hud_rev.png", hud_rev)

print("All HUD renders completed successfully!")
