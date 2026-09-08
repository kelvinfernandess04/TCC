import sys
import cv2
import numpy as np

sys.path.insert(0, 'Treinamento IA/scripts')
from guided_thumb_calibrator import GuidedThumbCalibrator, THUMB_CALIBRATION_STEPS

# Full 21 points with fingers from pts_front_can and thumb from P
pts_21 = np.zeros((21, 3))
# Reconstruct fingers from test_real_camera_thumb
from test_real_camera_thumb import P, pts_front_can
pts_21[:] = pts_front_can[:]
pts_21[0:5] = P

calib = GuidedThumbCalibrator()
vp1 = calib._render_viewport_3d(pts_21, 440, 490, 18.0, -12.0, "VISTA 1: FRONTAL / ORBITAL")
vp2 = calib._render_viewport_3d(pts_21, 440, 490, 90.0, 0.0, "VISTA 2: PERFIL LATERAL 90°")

canvas = np.hstack([vp1, vp2])
cv2.imwrite("scratch/comparison_new_fusion_transversal.png", canvas)
print("Saved scratch/comparison_new_fusion_transversal.png")
