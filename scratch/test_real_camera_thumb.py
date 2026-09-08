import cv2
import mediapipe as mp
import numpy as np

# Load the user's actual capture images
img_front = cv2.imread('Treinamento IA/data/calibration_captures/thumb_thumb_transversal_frontal.png')
img_lat = cv2.imread('Treinamento IA/data/calibration_captures/thumb_thumb_transversal_lateral.png')

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
res_front = hands.process(cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB))
res_lat = hands.process(cv2.cvtColor(img_lat, cv2.COLOR_BGR2RGB))

h_f, w_f, _ = img_front.shape
h_l, w_l, _ = img_lat.shape

pts_front_raw = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in res_front.multi_hand_landmarks[0].landmark])
pts_lat_raw = np.array([[lm.x * w_l, lm.y * h_l, lm.z * w_l] for lm in res_lat.multi_hand_landmarks[0].landmark])

# Canonical frame
def to_canonical(pts):
    p0 = pts[0].copy()
    palm_len = np.linalg.norm(pts[9] - p0)
    pts_n = (pts - p0) / palm_len
    
    v_y = pts_n[9]
    y_unit = v_y / np.linalg.norm(v_y)
    v_x_raw = pts_n[17] - pts_n[5]
    v_x = v_x_raw - np.dot(v_x_raw, y_unit) * y_unit
    x_unit = v_x / np.linalg.norm(v_x)
    
    e_x = x_unit
    e_y = -y_unit
    e_z = np.cross(e_x, e_y)
    R = np.stack([e_x, e_y, e_z], axis=0)
    return pts_n @ R.T, R, palm_len

pts_front_can, R_f, palm_len_f = to_canonical(pts_front_raw)

# Lateral Z
p0_l = pts_lat_raw[0]
p9_l = pts_lat_raw[9]
v_palm_l = p9_l - p0_l
norm_palm_l = np.linalg.norm(v_palm_l)
y_lat = v_palm_l / norm_palm_l
x_sag = np.array([-y_lat[1], y_lat[0], 0.0])
mcp_proj = float(np.dot(pts_lat_raw[2] - p0_l, x_sag))
sign = 1.0 if mcp_proj >= 0 else -1.0

z_lat = {}
for j in range(5):
    z_lat[j] = float(sign * np.dot(pts_lat_raw[j] - p0_l, x_sag) / norm_palm_l)

print("100% Real Camera Readings for Transversal Thumb:")
for j in range(5):
    print(f"  Point {j}: X_front={pts_front_can[j,0]:.3f}, Y_front={pts_front_can[j,1]:.3f}, Z_lat={z_lat[j]:+.3f}")

# Reconstructing with rigid lengths
l1 = 0.415
l2 = 0.320
l3 = 0.249

# Full 100% camera points:
Q = np.zeros((5, 3))
for j in range(5):
    Q[j] = np.array([pts_front_can[j, 0], pts_front_can[j, 1], z_lat[j]])

# Preserve rigid lengths along camera-read directions
P = np.zeros((5, 3))
P[0] = np.array([0.0, 0.0, 0.0])
P[1] = Q[1].copy() # 100% Camera read Point 1!

v1 = Q[2] - Q[1]
u1 = v1 / np.linalg.norm(v1)
P[2] = P[1] + l1 * u1

v2 = Q[3] - Q[2]
u2 = v2 / np.linalg.norm(v2)
P[3] = P[2] + l2 * u2

v3 = Q[4] - Q[3]
u3 = v3 / np.linalg.norm(v3)
P[4] = P[3] + l3 * u3

print("\nPreserved-rigid Points P:")
for j in range(5):
    print(f"  P{j}: X={P[j,0]:.3f}, Y={P[j,1]:.3f}, Z={P[j,2]:+.3f}")

print(f"Point 1 Z plane: {P[1,2]:+.3f} (IS NOT in wrist plane Z=0!)")
print(f"L1={np.linalg.norm(P[2]-P[1]):.3f}, L2={np.linalg.norm(P[3]-P[2]):.3f}, L3={np.linalg.norm(P[4]-P[3]):.3f}")
