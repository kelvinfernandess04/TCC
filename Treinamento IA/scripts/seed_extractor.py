import os
import json
import math
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEEDS_DIR = os.path.join(DATA_DIR, 'seeds')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'calibration_settings.json')

def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

def fuse_dual_plane_landmarks(pose_dict):
    if not isinstance(pose_dict, dict):
        return None
    front = pose_dict.get('front')
    profile = pose_dict.get('profile')

    if front is None and profile is None:
        return None

    if front is not None and profile is not None:
        fused = np.array(front)
        fused[:, 2] = np.array(profile)[:, 0]
        return fused

    return np.array(front if front is not None else profile)

def get_fallback_hand():
    return np.array([
        [0.0, 0.0, 0.0],
        [-0.06, -0.04, -0.02], [-0.11, -0.09, -0.04], [-0.15, -0.14, -0.05], [-0.18, -0.18, -0.06],
        [-0.08, -0.25, 0.00], [-0.10, -0.35, 0.00], [-0.11, -0.42, 0.00], [-0.12, -0.48, 0.00],
        [0.00, -0.26, 0.00], [0.00, -0.37, 0.00], [0.00, -0.45, 0.00], [0.00, -0.52, 0.00],
        [0.08, -0.24, 0.00], [0.10, -0.34, 0.00], [0.11, -0.41, 0.00], [0.12, -0.47, 0.00],
        [0.16, -0.20, 0.00], [0.19, -0.28, 0.00], [0.21, -0.34, 0.00], [0.22, -0.39, 0.00]
    ])

def generate_anatomical_hand_3d(finger_states, spread_states, thumb_opp, thumb_ip, captured_poses=None):
    fallback = get_fallback_hand()

    if not captured_poses:
        captured_poses = {}

    p_0_spread = fuse_dual_plane_landmarks(captured_poses.get('stage_0_spread'))
    p_0_closed = fuse_dual_plane_landmarks(captured_poses.get('stage_0_closed'))
    p_1        = fuse_dual_plane_landmarks(captured_poses.get('stage_1'))
    p_2        = fuse_dual_plane_landmarks(captured_poses.get('stage_2'))
    p_3        = fuse_dual_plane_landmarks(captured_poses.get('stage_3'))
    p_opp      = fuse_dual_plane_landmarks(captured_poses.get('thumb_opposition'))
    p_ip       = fuse_dual_plane_landmarks(captured_poses.get('thumb_ip_flexed'))

    if p_0_spread is None: p_0_spread = fallback
    if p_0_closed is None: p_0_closed = p_0_spread
    if p_1 is None: p_1 = p_0_spread
    if p_2 is None: p_2 = p_1
    if p_3 is None: p_3 = p_2

    lms = np.zeros((21, 3))
    lms[0] = p_0_spread[0]
    for idx in [1, 5, 9, 13, 17]:
        lms[idx] = p_0_spread[idx]

    def get_finger_chain(pose, mcp_idx, tip_idxs):
        mcp = pose[mcp_idx]
        return [pose[tip_idxs[0]] - mcp, pose[tip_idxs[1]] - mcp, pose[tip_idxs[2]] - mcp]

    # Fingers 4 (Index, Middle, Ring, Pinky)
    fingers_info = [
        ('Index',  5,  [6, 7, 8],   'Middle_Index',  8.0),
        ('Middle', 9,  [10, 11, 12], 'Middle_Index', 0.0),
        ('Ring',   13, [14, 15, 16], 'Ring_Middle',  -6.0),
        ('Pinky',  17, [18, 19, 20], 'Pinky_Ring',   -14.0)
    ]

    for finger, mcp_idx, tip_idxs, sp_key, abduct_angle in fingers_info:
        st = float(finger_states.get(finger, 0.0))
        sp = float(spread_states.get(sp_key, 0.0))

        # Base chain from stage_0_closed (sp=0) to stage_0_spread (sp=1)
        c_open_closed = get_finger_chain(p_0_closed, mcp_idx, tip_idxs)
        c_open_spread = get_finger_chain(p_0_spread, mcp_idx, tip_idxs)
        c_0 = [c_open_closed[i] * (1.0 - sp) + c_open_spread[i] * sp for i in range(3)]

        c_1 = get_finger_chain(p_1, mcp_idx, tip_idxs)
        c_2 = get_finger_chain(p_2, mcp_idx, tip_idxs)
        c_3 = get_finger_chain(p_3, mcp_idx, tip_idxs)

        if st <= 1.0:
            c_final = [c_0[i] * (1.0 - st) + c_1[i] * st for i in range(3)]
        elif st <= 2.0:
            w = st - 1.0
            c_final = [c_1[i] * (1.0 - w) + c_2[i] * w for i in range(3)]
        else:
            w = st - 2.0
            c_final = [c_2[i] * (1.0 - w) + c_3[i] * w for i in range(3)]

        if sp > 0.01 and st <= 1.0:
            R_sp = rot_z(abduct_angle * sp)
            c_final = [R_sp.dot(v) for v in c_final]

        mcp_pos = lms[mcp_idx]
        lms[tip_idxs[0]] = mcp_pos + c_final[0]
        lms[tip_idxs[1]] = mcp_pos + c_final[1]
        lms[tip_idxs[2]] = mcp_pos + c_final[2]

    # Thumb Kinematics (Anatomical: CMC -> MCP -> IP -> TIP)
    t_st = float(finger_states.get('Thumb', 0.0))
    t_opp = float(thumb_opp)
    t_ip = float(thumb_ip)
    t_sp = float(spread_states.get('Index_Thumb', 0.0))

    cmc_pos = lms[1]
    
    # 1. Spread (Abertura Polegar-Indicador: 0 = aberto/leque, 1 = fechado/unido)
    t_c_open_spread = [p_0_spread[idx] - p_0_spread[1] for idx in [2, 3, 4]]
    t_c_open_closed = [p_0_closed[idx] - p_0_closed[1] for idx in [2, 3, 4]]
    t_c_base = [t_c_open_spread[i] * (1.0 - t_sp) + t_c_open_closed[i] * t_sp for i in range(3)]

    # 2. Opposition (Movimento Transversal F: cruzando na frente da palma)
    if p_opp is not None:
        t_c_opp = [p_opp[idx] - p_opp[1] for idx in [2, 3, 4]]
    else:
        idx_mcp = lms[5]
        mid_mcp = lms[9]
        vec_idx = idx_mcp - cmc_pos
        vec_mid = mid_mcp - cmc_pos
        t_c_opp = [
            vec_idx * 0.5 + np.array([0.04, 0.0, -0.04]),
            vec_idx * 0.75 + np.array([0.08, 0.0, -0.05]),
            vec_mid * 0.85 + np.array([0.12, 0.0, -0.06])
        ]

    opp_weight = max(t_opp, 1.0 if t_st >= 2.5 else 0.0)
    t_c_final = [t_c_base[i] * (1.0 - opp_weight) + t_c_opp[i] * opp_weight for i in range(3)]

    # 3. IP Flexion (Ponta do Polegar P: flexão da falange distal IP->TIP)
    if t_ip > 0.01:
        if p_ip is not None:
            vec_tip_ip_flexed = p_ip[4] - p_ip[3]
            vec_tip_ip_spread = p_0_spread[4] - p_0_spread[3]
            delta_tip = vec_tip_ip_flexed - vec_tip_ip_spread
            t_c_final[2] = t_c_final[2] + delta_tip * t_ip
        else:
            t_c_final[2][1] += t_ip * 0.04
            t_c_final[2][2] -= t_ip * 0.05

    lms[2] = cmc_pos + t_c_final[0]
    lms[3] = cmc_pos + t_c_final[1]
    lms[4] = cmc_pos + t_c_final[2]

    return lms

def main():
    logging.info("=== Extrator Anatômico Cinemático Híbrido ===")

    captured_poses = {}
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            captured_poses = data.get("captured_poses", {})
        except Exception as e:
            logging.warning(f"Erro ao ler calibração: {e}")

    seeds = {}
    regular_states = [0, 1, 2, 3]
    thumb_states = [0, 2, 3]

    total = 0
    for pinky_s in regular_states:
        for ring_s in regular_states:
            pr_options = [0, 1] if (pinky_s <= 1 and ring_s <= 1) else [0]
            for pr_spread in pr_options:

                for middle_s in regular_states:
                    rm_options = [0, 1] if (ring_s <= 1 and middle_s <= 1) else [0]
                    for rm_spread in rm_options:

                        for index_s in regular_states:
                            mi_options = [0, 1] if (middle_s <= 1 and index_s <= 1) else [0]
                            for mi_spread in mi_options:

                                for thumb_s in thumb_states:
                                    it_options = [0, 1] if (index_s <= 1 and thumb_s <= 1) else [0]
                                    for it_spread in it_options:

                                        for thumb_opp in [0, 1]:
                                            thumb_ip = 1 if thumb_s == 3 else 0

                                            label = (f"{pinky_s}{pr_spread}{ring_s}{rm_spread}"
                                                     f"{middle_s}{mi_spread}{index_s}"
                                                     f"{it_spread}{thumb_opp}{thumb_ip}")

                                            finger_states = {
                                                'Pinky': pinky_s, 'Ring': ring_s,
                                                'Middle': middle_s, 'Index': index_s,
                                                'Thumb': thumb_s
                                            }
                                            spread_states = {
                                                'Pinky_Ring': pr_spread, 'Ring_Middle': rm_spread,
                                                'Middle_Index': mi_spread, 'Index_Thumb': it_spread
                                            }

                                            lms = generate_anatomical_hand_3d(
                                                finger_states, spread_states, thumb_opp, thumb_ip, captured_poses
                                            )

                                            seeds[label] = [
                                                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                                                for p in lms
                                            ]
                                            total += 1

    os.makedirs(SEEDS_DIR, exist_ok=True)
    seeds_file = os.path.join(SEEDS_DIR, 'seeds.json')
    with open(seeds_file, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, indent=2)

    logging.info(f"Banco de 3.936 sementes cinemáticas salvas em: {seeds_file}")

if __name__ == "__main__":
    main()
