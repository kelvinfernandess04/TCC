import os
import json
import glob
import math
import numpy as np
import itertools
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURES_DIR = os.path.join(BASE_DIR, 'data', 'captured_gestures')
SEEDS_DIR = os.path.join(BASE_DIR, 'data', 'seeds')

# MediaPipe landmark indices per finger
FINGER_JOINTS = {
    'Thumb':  [0, 1, 2, 3, 4],
    'Index':  [0, 5, 6, 7, 8],
    'Middle': [0, 9, 10, 11, 12],
    'Ring':   [0, 13, 14, 15, 16],
    'Pinky':  [0, 17, 18, 19, 20]
}

# MCP indices for spread angle computation (base of each finger)
MCP_INDICES = {
    'Thumb': 1, 'Index': 5, 'Middle': 9, 'Ring': 13, 'Pinky': 17
}
PIP_INDICES = {
    'Thumb': 2, 'Index': 6, 'Middle': 10, 'Ring': 14, 'Pinky': 18
}

# ---------------------------------------------------------
# VECTOR MATH UTILITIES
# ---------------------------------------------------------

def vec_angle(v1, v2):
    """Angle between two vectors in degrees."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))

def joint_flexion(p_prev, p_joint, p_next):
    """Flexion at a joint = 180 - angle(prev->joint, joint->next). 0 = straight."""
    v1 = p_prev - p_joint
    v2 = p_next - p_joint
    return 180.0 - vec_angle(v1, v2)

# ---------------------------------------------------------
# EXTRACTION FROM CAPTURES
# ---------------------------------------------------------

def load_all_captures():
    """Load all captured_gestures_*.json and return flat list of frames (each = 21 x 3D points)."""
    pattern = os.path.join(CAPTURES_DIR, 'captured_gestures_*.json')
    files = sorted(glob.glob(pattern))
    logging.info(f"Encontrados {len(files)} arquivos de captura.")

    all_frames = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            frames = json.load(f)
        for frame in frames:
            if len(frame) == 21:
                pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in frame])
                all_frames.append(pts)
    logging.info(f"Total de frames válidos carregados: {len(all_frames)}")
    return all_frames

def extract_bone_lengths(frames):
    """Extract average bone lengths for each finger segment across all frames."""
    # Bone segments: for each finger, consecutive pairs of joints
    bone_segments = {
        'Thumb':  [(1,2), (2,3), (3,4)],
        'Index':  [(5,6), (6,7), (7,8)],
        'Middle': [(9,10), (10,11), (11,12)],
        'Ring':   [(13,14), (14,15), (15,16)],
        'Pinky':  [(17,18), (18,19), (19,20)]
    }
    # Also palm bones (wrist to MCP)
    palm_segments = {
        'Thumb':  (0,1),
        'Index':  (0,5),
        'Middle': (0,9),
        'Ring':   (0,13),
        'Pinky':  (0,17)
    }

    lengths = {f: [[] for _ in range(3)] for f in bone_segments}
    palm_lengths = {f: [] for f in palm_segments}

    for pts in frames:
        for finger, segs in bone_segments.items():
            for i, (a, b) in enumerate(segs):
                lengths[finger][i].append(np.linalg.norm(pts[b] - pts[a]))
        for finger, (a, b) in palm_segments.items():
            palm_lengths[finger].append(np.linalg.norm(pts[b] - pts[a]))

    avg_lengths = {f: [float(np.mean(seg)) for seg in segs] for f, segs in lengths.items()}
    avg_palm = {f: float(np.mean(v)) for f, v in palm_lengths.items()}

    return avg_lengths, avg_palm

def extract_palm_bases(frames):
    """Extract average knuckle positions relative to the wrist (points in simulator space)."""
    indices = {'Thumb': 1, 'Index': 5, 'Middle': 9, 'Ring': 13, 'Pinky': 17}
    bases = {f: [] for f in indices}
    for pts in frames:
        wrist = pts[0]
        for finger, idx in indices.items():
            bx = pts[idx][0] - wrist[0]
            by = -(pts[idx][1] - wrist[1])
            bz = -(pts[idx][2] - wrist[2])
            bases[finger].append([bx, by, bz])
    return {f: [float(v) for v in np.mean(bases[f], axis=0)] for f in bases}

def extract_joint_angles(frames):
    """Extract min/max flexion angles for MCP, PIP, DIP of each finger."""
    # Joint triplets: (previous, joint, next) for flexion calculation
    joint_triplets = {
        'Thumb':  {'MCP': (1, 2, 3), 'DIP': (2, 3, 4)},  # Thumb has IP instead of PIP/DIP
        'Index':  {'MCP': (5, 6, 7), 'PIP': (6, 7, 8)},   # Simplified: MCP flex = at PIP, PIP flex = at DIP
        'Middle': {'MCP': (9, 10, 11), 'PIP': (10, 11, 12)},
        'Ring':   {'MCP': (13, 14, 15), 'PIP': (14, 15, 16)},
        'Pinky':  {'MCP': (17, 18, 19), 'PIP': (18, 19, 20)}
    }
    # For MCP flexion (knuckle bend), we use the palm bone as reference
    mcp_triplets = {
        'Thumb':  (0, 1, 2),
        'Index':  (0, 5, 6),
        'Middle': (0, 9, 10),
        'Ring':   (0, 13, 14),
        'Pinky':  (0, 17, 18)
    }

    angles = {}
    for finger in joint_triplets:
        angles[finger] = {'MCP': [], 'PIP_DIP': []}

    for pts in frames:
        for finger in joint_triplets:
            # MCP flexion (at knuckle)
            a, b, c = mcp_triplets[finger]
            mcp_flex = joint_flexion(pts[a], pts[b], pts[c])
            angles[finger]['MCP'].append(mcp_flex)

            if finger == 'Thumb':
                # Para o polegar, extrair a flexão IP (triplete 2, 3, 4) isoladamente
                ip_flex = joint_flexion(pts[2], pts[3], pts[4])
                angles[finger]['PIP_DIP'].append(ip_flex)
            else:
                # PIP+DIP combined (they share the same tendon per user specification)
                pip_dip_vals = []
                for jname, (a2, b2, c2) in joint_triplets[finger].items():
                    pip_dip_vals.append(joint_flexion(pts[a2], pts[b2], pts[c2]))
                # Average of both joints since they're mechanically linked
                angles[finger]['PIP_DIP'].append(float(np.mean(pip_dip_vals)))

    ranges = {}
    for finger in angles:
        ranges[finger] = {
            'MCP': (float(np.percentile(angles[finger]['MCP'], 2)),
                    float(np.percentile(angles[finger]['MCP'], 98))),
            'PIP_DIP': (float(np.percentile(angles[finger]['PIP_DIP'], 2)),
                        float(np.percentile(angles[finger]['PIP_DIP'], 98)))
        }
    return ranges

def extract_spread_angles(frames):
    """Extract min/max spread angles between adjacent fingers."""
    pairs = [
        ('Pinky', 'Ring', 17, 18, 13, 14),
        ('Ring', 'Middle', 13, 14, 9, 10),
        ('Middle', 'Index', 9, 10, 5, 6),
        ('Index', 'Thumb', 5, 6, 1, 2)
    ]

    spreads = {f"{a}_{b}": [] for a, b, *_ in pairs}
    for pts in frames:
        for a_name, b_name, mcp_a, pip_a, mcp_b, pip_b in pairs:
            v_a = pts[pip_a] - pts[mcp_a]
            v_b = pts[pip_b] - pts[mcp_b]
            # Project onto XY plane for cleaner spread measurement
            v_a_2d = v_a[:2]
            v_b_2d = v_b[:2]
            angle = vec_angle(v_a_2d, v_b_2d)
            spreads[f"{a_name}_{b_name}"].append(angle)

    spread_ranges = {}
    for key, vals in spreads.items():
        spread_ranges[key] = (float(np.percentile(vals, 5)),
                              float(np.percentile(vals, 95)))
    return spread_ranges

def extract_thumb_opposition(frames):
    """Extract thumb opposition angle range (perpendicular movement toward palm center)."""
    opp_angles = []
    for pts in frames:
        # Palm plane normal: cross product of (wrist->middle_mcp) and (wrist->index_mcp)
        v1 = pts[9] - pts[0]   # wrist -> middle MCP
        v2 = pts[5] - pts[0]   # wrist -> index MCP
        palm_normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(palm_normal)
        if norm_len < 1e-9:
            opp_angles.append(0.0)
            continue
        palm_normal = palm_normal / norm_len

        # Thumb metacarpal direction
        thumb_dir = pts[2] - pts[1]  # CMC -> MCP
        thumb_norm = np.linalg.norm(thumb_dir)
        if thumb_norm < 1e-9:
            opp_angles.append(0.0)
            continue
        thumb_dir = thumb_dir / thumb_norm

        # Opposition = how much the thumb metacarpal projects onto the palm normal
        projection = abs(np.dot(thumb_dir, palm_normal))
        opp_angle = math.degrees(math.asin(np.clip(projection, 0, 1)))
        opp_angles.append(opp_angle)

    return (float(np.percentile(opp_angles, 5)), float(np.percentile(opp_angles, 95)))

# ---------------------------------------------------------
# ROTATION MATRICES (EULER)
# ---------------------------------------------------------

def rot_x(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

def rot_y(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

def rot_z(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

# ---------------------------------------------------------
# FORWARD KINEMATICS SEED GENERATION
# ---------------------------------------------------------

def calc_finger_chain(base_pos, base_rot_mat, lengths, pitches):
    """Forward kinematics: compute joint positions from base, rotation, bone lengths, and flexion angles."""
    points = [base_pos.copy()]
    current_mat = base_rot_mat.copy()
    current_pos = base_pos.copy()

    for length, pitch in zip(lengths, pitches):
        joint_rot = rot_x(pitch)
        current_mat = current_mat.dot(joint_rot)
        local_bone = np.array([0, length, 0])
        world_bone = current_mat.dot(local_bone)
        current_pos = current_pos + world_bone
        points.append(current_pos.copy())

    return points

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t

def compute_finger_pitches(state, mcp_range, pip_dip_range):
    """
    Compute [MCP, PIP, DIP] pitch angles based on finger state.
    State 0: Straight (min flexion)
    State 1: PIP/DIP half-bent, MCP relaxed (mechanically linked tendon)
    State 2: PIP/DIP fully closed, MCP straight (hook)
    State 3: All joints fully closed
    """
    mcp_min, mcp_max = mcp_range
    pip_min, pip_max = pip_dip_range

    if state == 0:
        return [mcp_min, pip_min, pip_min]
    elif state == 1:
        half_pip = lerp(pip_min, pip_max, 0.5)
        return [lerp(mcp_min, mcp_max, 0.15), half_pip, half_pip]
    elif state == 2:
        return [mcp_min, pip_max, pip_max]
    else:  # state == 3
        return [mcp_max, pip_max, pip_max]

def compute_thumb_pitches(state, mcp_range, pip_dip_range):
    """
    Thumb-specific pitch computation.
    Thumb only has states 0, 2, 3 (no state 1 — cannot partially bend).
    """
    mcp_min, mcp_max = mcp_range
    ip_min, ip_max = pip_dip_range

    if state == 0:
        return [mcp_min, ip_min, 0.0]
    elif state == 2:
        return [mcp_min, ip_max, 0.0]
    else:  # state == 3
        return [mcp_max, ip_max, 0.0]

def generate_hand_3d(finger_states, spread_states, thumb_opp_state,
                     avg_lengths, avg_palm, angle_ranges, spread_ranges, opp_range,
                     avg_palm_bases=None):
    """
    Generate 21 3D landmarks for a hand configuration.

    finger_states: dict {Pinky: 0-3, Ring: 0-3, Middle: 0-3, Index: 0-3, Thumb: 0-3}
    spread_states: dict {Pinky_Ring: 0/1, Ring_Middle: 0/1, Middle_Index: 0/1, Index_Thumb: 0/1}
    thumb_opp_state: 0 or 1
    """
    if avg_palm_bases is not None:
        palm_bases = {f: np.array(avg_palm_bases[f]) for f in avg_palm_bases}
    else:
        # Palm base positions (normalized around wrist at origin)
        # Derived from average palm bone lengths and natural finger spread
        palm_bases = {
            'Thumb':  np.array([-0.06, 0.04, 0.02]),
            'Index':  np.array([-0.08, 0.45, 0.0]),
            'Middle': np.array([ 0.00, 0.48, 0.0]),
            'Ring':   np.array([ 0.08, 0.45, 0.0]),
            'Pinky':  np.array([ 0.16, 0.38, 0.0])
        }
        # Scale palm bases by actual proportions
        for finger in palm_bases:
            direction = palm_bases[finger] / max(np.linalg.norm(palm_bases[finger]), 1e-9)
            palm_bases[finger] = direction * avg_palm[finger]

    # Default yaw (base direction of each finger)
    default_yaw = {'Index': 5.0, 'Middle': 0.0, 'Ring': -5.0, 'Pinky': -15.0}

    landmarks_3d = [np.array([0.0, 0.0, 0.0])]  # Wrist

    fingers_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    for finger in fingers_order:
        state = finger_states[finger]
        lengths = avg_lengths[finger]

        if finger == 'Thumb':
            # Biological orientation of CMC -> MCP (segment 1->2) grouped by (L, P)
            thumb_angles = {
                (0, 0): {'yaw': -25.0, 'pitch': 5.4},
                (0, 2): {'yaw': -27.2, 'pitch': 7.8},
                (0, 3): {'yaw': -10.5, 'pitch': 6.4},
                (1, 0): {'yaw': -24.7, 'pitch': 16.5},
                (1, 2): {'yaw': -35.5, 'pitch': 20.3},
                (1, 3): {'yaw': -17.2, 'pitch': 25.7}
            }

            # Real thumb segment lengths grouped by (L, P) from captures
            thumb_lengths = {
                (0, 0): [0.0982, 0.0758, 0.0572],
                (0, 2): [0.0859, 0.0666, 0.0424],
                (0, 3): [0.0873, 0.0761, 0.0546],
                (1, 0): [0.0780, 0.0613, 0.0523],
                (1, 2): [0.0672, 0.0672, 0.0362],
                (1, 3): [0.0609, 0.0470, 0.0383]
            }

            # Real thumb flexion angles (MCP, IP) grouped by (L, P) from captures
            thumb_flexions = {
                (0, 0): {'mcp': 4.10, 'ip': 6.54},
                (0, 2): {'mcp': 9.60, 'ip': 14.37},
                (0, 3): {'mcp': 34.74, 'ip': 18.46},
                (1, 0): {'mcp': 5.87, 'ip': 5.27},
                (1, 2): {'mcp': 10.79, 'ip': 17.82},
                (1, 3): {'mcp': 39.84, 'ip': 15.74}
            }
            
            opp_t = float(thumb_opp_state)
            opp_t_scaled = opp_t * 1.6  # Anti Z-Flattening compensation (1.6x multiplier)
            p_idx = 0 if state == 0 else (2 if state == 2 else 3)
            
            # Interpolate Yaw and Pitch with scaled opposition
            yaw_L0 = thumb_angles[(0, p_idx)]['yaw']
            yaw_L1 = thumb_angles[(1, p_idx)]['yaw']
            pitch_L0 = thumb_angles[(0, p_idx)]['pitch']
            pitch_L1 = thumb_angles[(1, p_idx)]['pitch']
            yaw = lerp(yaw_L0, yaw_L1, opp_t_scaled)
            pitch = lerp(pitch_L0, pitch_L1, opp_t_scaled)

            # Interpolate Segment Lengths
            lens_L0 = thumb_lengths[(0, p_idx)]
            lens_L1 = thumb_lengths[(1, p_idx)]
            lengths = [lerp(l0, l1, opp_t) for l0, l1 in zip(lens_L0, lens_L1)]

            # Interpolate MCP and IP Flexion Angles
            mcp_L0 = thumb_flexions[(0, p_idx)]['mcp']
            mcp_L1 = thumb_flexions[(1, p_idx)]['mcp']
            ip_L0 = thumb_flexions[(0, p_idx)]['ip']
            ip_L1 = thumb_flexions[(1, p_idx)]['ip']
            mcp_flex = lerp(mcp_L0, mcp_L1, opp_t)
            ip_flex = lerp(ip_L0, ip_L1, opp_t)
            j1_y = -yaw
            j1_p = pitch
            j2_y = 0.0
            j2_p = mcp_flex
            j3_y = 0.0
            j3_p = ip_flex

            # J1 rotates segment 0-1 (Wrist to CMC)
            R1 = rot_z(j1_y).dot(rot_x(j1_p))
            p1 = R1.dot(palm_bases['Thumb'])

            # J2 rotates segment 1-2 (CMC to MCP)
            R2 = R1.dot(rot_z(j2_y).dot(rot_x(j2_p)))
            p2 = p1 + R2.dot(np.array([0.0, lengths[0], 0.0]))

            # J3 rotates segment 2-3 (MCP to IP)
            R3 = R2.dot(rot_z(j3_y).dot(rot_x(j3_p)))
            p3 = p2 + R3.dot(np.array([0.0, lengths[1], 0.0]))

            # Segment 3-4 (IP to Tip) follows segment 2-3 (relative rotation 0.0)
            p4 = p3 + R3.dot(np.array([0.0, lengths[2], 0.0]))

            chain = [p1, p2, p3, p4]
        else:
            pitches = compute_finger_pitches(state, angle_ranges[finger]['MCP'],
                                             angle_ranges[finger]['PIP_DIP'])

            # Spread angle
            yaw = default_yaw.get(finger, 0.0)

            # Apply spread adjustments between adjacent fingers
            if finger == 'Index':
                if spread_states.get('Middle_Index', 0) == 1:
                    yaw += 10.0
                if spread_states.get('Index_Thumb', 0) == 1:
                    yaw += 5.0
            elif finger == 'Middle':
                if spread_states.get('Middle_Index', 0) == 1:
                    yaw -= 10.0
                if spread_states.get('Ring_Middle', 0) == 1:
                    yaw += 5.0
            elif finger == 'Ring':
                if spread_states.get('Ring_Middle', 0) == 1:
                    yaw -= 5.0
                if spread_states.get('Pinky_Ring', 0) == 1:
                    yaw += 5.0
            elif finger == 'Pinky':
                if spread_states.get('Pinky_Ring', 0) == 1:
                    yaw -= 10.0

            j1_y = yaw
            j1_p = pitches[0]
            j2_y = 0.0
            j2_p = pitches[1]
            j3_y = 0.0
            j3_p = pitches[2]

            # J1 rotates segment 0-base (Wrist to MCP)
            R1 = rot_z(j1_y).dot(rot_x(j1_p))
            p1 = R1.dot(palm_bases[finger])

            # J2 rotates segment base-mid1
            R2 = R1.dot(rot_z(j2_y).dot(rot_x(j2_p)))
            p2 = p1 + R2.dot(np.array([0.0, lengths[0], 0.0]))

            # J3 rotates segment mid1-mid2
            R3 = R2.dot(rot_z(j3_y).dot(rot_x(j3_p)))
            p3 = p2 + R3.dot(np.array([0.0, lengths[1], 0.0]))

            # Tip follows mid2 segment (relative rotation 0.0)
            p4 = p3 + R3.dot(np.array([0.0, lengths[2], 0.0]))

            chain = [p1, p2, p3, p4]
            
        landmarks_3d.extend(chain)

    # Filtro clínico de segurança biomecânica (anti-recuo)
    if landmarks_3d[1][2] < landmarks_3d[0][2] - 0.01:
        return None

    return landmarks_3d

# ---------------------------------------------------------
# COMBINATION GENERATION WITH XAXAXAXAAX CONVENTION
# ---------------------------------------------------------

def generate_all_combinations(avg_lengths, avg_palm, avg_palm_bases, angle_ranges, spread_ranges, opp_range):
    """
    Generate all valid XAXAXAXAAX combinations and their 3D seeds.
    
    Rules:
    - Regular fingers (Pinky, Ring, Middle, Index): states 0, 1, 2, 3
    - Thumb: states 0, 2, 3 only (cannot partially bend, no state 1)
    - Spread (A): only has 2 options (0/1) when BOTH adjacent fingers are ≤ 1.
      Otherwise forced to 0 (only 1 option, no duplicate labels).
    - Thumb opposition: always 2 options (0/1)
    """
    seeds = {}
    regular_states = [0, 1, 2, 3]
    thumb_states = [0, 2, 3]  # Polegar não consegue se inclinar parcialmente

    total = 0

    for pinky_s in regular_states:
        for ring_s in regular_states:
            # Spread Pinky-Ring: 2 options if both ≤1, else forced 0
            pr_options = [0, 1] if (pinky_s <= 1 and ring_s <= 1) else [0]
            for pr_spread in pr_options:

                for middle_s in regular_states:
                    # Spread Ring-Middle
                    rm_options = [0, 1] if (ring_s <= 1 and middle_s <= 1) else [0]
                    for rm_spread in rm_options:

                        for index_s in regular_states:
                            # Spread Middle-Index
                            mi_options = [0, 1] if (middle_s <= 1 and index_s <= 1) else [0]
                            for mi_spread in mi_options:

                                for thumb_s in thumb_states:
                                    # Spread Index-Thumb (thumb ≤1 means thumb == 0)
                                    it_options = [0, 1] if (index_s <= 1 and thumb_s <= 1) else [0]
                                    for it_spread in it_options:

                                        for thumb_opp in [0, 1]:
                                            # Build label: XAXAXAXAAX
                                            label = (f"{pinky_s}{pr_spread}{ring_s}{rm_spread}"
                                                     f"{middle_s}{mi_spread}{index_s}"
                                                     f"{it_spread}{thumb_opp}{thumb_s}")

                                            finger_states = {
                                                'Pinky': pinky_s,
                                                'Ring': ring_s,
                                                'Middle': middle_s,
                                                'Index': index_s,
                                                'Thumb': thumb_s
                                            }
                                            spread_states = {
                                                'Pinky_Ring': pr_spread,
                                                'Ring_Middle': rm_spread,
                                                'Middle_Index': mi_spread,
                                                'Index_Thumb': it_spread
                                            }

                                            lms = generate_hand_3d(
                                                finger_states, spread_states, thumb_opp,
                                                avg_lengths, avg_palm, angle_ranges,
                                                spread_ranges, opp_range,
                                                avg_palm_bases=avg_palm_bases
                                            )

                                            if lms is None:
                                                continue

                                            seeds[label] = [
                                                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                                                for p in lms
                                            ]
                                            total += 1

    logging.info(f"Combinações válidas geradas: {total}")
    return seeds

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    logging.info("=== Extrator Biomecânico de Sementes (XAXAXAXAAX) ===")

    # 1. Load all captures
    frames = load_all_captures()
    if not frames:
        logging.error("Nenhum frame de captura encontrado. Abortando.")
        return

    # 2. Extract biomechanical parameters
    logging.info("Extraindo comprimentos dos ossos...")
    avg_lengths, avg_palm = extract_bone_lengths(frames)
    for finger, lens in avg_lengths.items():
        logging.info(f"  {finger}: {[f'{l:.4f}' for l in lens]}")

    logging.info("Extraindo ângulos articulares (ROM)...")
    angle_ranges = extract_joint_angles(frames)
    for finger, ranges in angle_ranges.items():
        logging.info(f"  {finger} MCP: {ranges['MCP'][0]:.1f}°-{ranges['MCP'][1]:.1f}° | "
                     f"PIP/DIP: {ranges['PIP_DIP'][0]:.1f}°-{ranges['PIP_DIP'][1]:.1f}°")

    logging.info("Extraindo ângulos de abertura entre dedos...")
    spread_ranges = extract_spread_angles(frames)
    for pair, (lo, hi) in spread_ranges.items():
        logging.info(f"  {pair}: {lo:.1f}°-{hi:.1f}°")

    logging.info("Extraindo oposição do polegar...")
    opp_range = extract_thumb_opposition(frames)
    logging.info(f"  Oposição: {opp_range[0]:.1f}°-{opp_range[1]:.1f}°")

    logging.info("Extraindo bases da palma (anatomia real)...")
    avg_palm_bases = extract_palm_bases(frames)
    for finger, base in avg_palm_bases.items():
        logging.info(f"  {finger} Base: [{base[0]:.4f}, {base[1]:.4f}, {base[2]:.4f}]")

    # 3. Generate all valid seeds
    logging.info("Gerando sementes para todas as combinações válidas...")
    seeds = generate_all_combinations(avg_lengths, avg_palm, avg_palm_bases, angle_ranges, spread_ranges, opp_range)

    # 4. Save with metadata
    seeds["__metadata__"] = {
        "avg_lengths": avg_lengths,
        "avg_palm": avg_palm,
        "avg_palm_bases": avg_palm_bases,
        "angle_ranges": angle_ranges,
        "spread_ranges": spread_ranges,
        "opp_range": opp_range
    }

    os.makedirs(SEEDS_DIR, exist_ok=True)
    seeds_file = os.path.join(SEEDS_DIR, 'seeds.json')
    with open(seeds_file, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, indent=2)

    logging.info(f"Banco de sementes salvo com {len(seeds) - 1} classes (+ __metadata__) em: {seeds_file}")

if __name__ == "__main__":
    main()
