import sys
import numpy as np

# Test mathematical fusion with realistic mock data
palm_base = np.array([
    [ 0.000,  0.000,  0.000],  # Wrist (0)
    [-0.164, -0.295,  0.000],  # Thumb CMC (1)
    [-0.138, -0.980,  0.000],  # Index MCP (5)
    [ 0.000, -0.997,  0.000],  # Middle MCP (9)
    [ 0.100, -0.950,  0.000],  # Ring MCP (13)
    [ 0.213, -0.887,  0.000]   # Pinky MCP (17)
], dtype=np.float64)

rigid_thumb_lengths = (0.415, 0.320, 0.249)

# Simulating frontal capture of thumb_open (pointing down-left at ~45 deg in XY)
pts_front_can = np.zeros((21, 3))
pts_front_can[1] = palm_base[1]
# Thumb extending outwards in X and slightly downwards in Y
pts_front_can[2] = pts_front_can[1] + np.array([-0.35, -0.20, -0.30]) # Note: noisy frontal Z
pts_front_can[3] = pts_front_can[2] + np.array([-0.25, -0.15, -0.20])
pts_front_can[4] = pts_front_can[3] + np.array([-0.20, -0.10, -0.16])

# Simulating lateral capture
pts_lat = np.zeros((21, 3))
pts_lat[0] = [0.0, 0.0, 0.0]
pts_lat[9] = [0.0, -1.0, 0.0]

def fuse_test(pts_front_can, pts_lat, thumb_state):
    out = pts_front_can.copy()
    l1, l2, l3 = rigid_thumb_lengths
    
    p0 = pts_lat[0]
    p9 = pts_lat[9]
    v_palm = p9 - p0
    norm_palm = np.linalg.norm(v_palm)
    y_lat = v_palm / norm_palm if norm_palm > 1e-6 else np.array([0.0, -1.0, 0.0])
    x_sag = np.array([-y_lat[1], y_lat[0], 0.0])
    
    mcp_proj = float(np.dot(pts_lat[2] - p0, x_sag))
    sign = 1.0 if mcp_proj >= 0 else -1.0
    
    z_lat = {}
    for j in [1, 2, 3, 4]:
        z_lat[j] = float(sign * np.dot(pts_lat[j] - p0, x_sag))
        
    if thumb_state == 0:
        dz1, dz2, dz3 = 0.0, 0.0, 0.0
    elif thumb_state == 1:
        dz1 = float(np.clip(z_lat[2] - z_lat[1], 0.01, 0.12))
        dz2 = float(np.clip(z_lat[3] - z_lat[2], 0.01, 0.08))
        dz3 = float(np.clip(z_lat[4] - z_lat[3], 0.00, 0.06))
    else:
        dz1 = float(np.clip(z_lat[2] - z_lat[1], 0.15, 0.40))
        dz2 = float(np.clip(z_lat[3] - z_lat[2], 0.10, 0.35))
        dz3 = float(np.clip(z_lat[4] - z_lat[3], 0.05, 0.25))
        
    dx1 = pts_front_can[2, 0] - pts_front_can[1, 0]
    dy1 = pts_front_can[2, 1] - pts_front_can[1, 1]
    v1 = np.array([dx1, dy1, dz1], dtype=np.float64)

    dx2 = pts_front_can[3, 0] - pts_front_can[2, 0]
    dy2 = pts_front_can[3, 1] - pts_front_can[2, 1]
    v2 = np.array([dx2, dy2, dz2], dtype=np.float64)

    dx3 = pts_front_can[4, 0] - pts_front_can[3, 0]
    dy3 = pts_front_can[4, 1] - pts_front_can[3, 1]
    v3 = np.array([dx3, dy3, dz3], dtype=np.float64)

    u1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-6 else np.array([-1.0, 0.0, 0.0])
    u2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-6 else u1
    u3 = v3 / np.linalg.norm(v3) if np.linalg.norm(v3) > 1e-6 else u2

    out[1] = palm_base[1].copy()
    out[1, 2] = 0.0
    out[2] = out[1] + l1 * u1
    out[3] = out[2] + l2 * u2
    out[4] = out[3] + l3 * u3

    return out

fused_open = fuse_test(pts_front_can, pts_lat, 0)
print("Fused thumb_open Z values:")
for j in [1, 2, 3, 4]:
    print(f"Joint {j}: X={fused_open[j, 0]:.3f}, Y={fused_open[j, 1]:.3f}, Z={fused_open[j, 2]:.3f}")

l1_fused = np.linalg.norm(fused_open[2] - fused_open[1])
l2_fused = np.linalg.norm(fused_open[3] - fused_open[2])
l3_fused = np.linalg.norm(fused_open[4] - fused_open[3])
print(f"Bone lengths preserved: L1={l1_fused:.3f} (target {rigid_thumb_lengths[0]:.3f}), L2={l2_fused:.3f} (target {rigid_thumb_lengths[1]:.3f}), L3={l3_fused:.3f} (target {rigid_thumb_lengths[2]:.3f})")
