import os
import json
import math
import time
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'datasets', 'synthetic_dataset')
SEEDS_FILE = os.path.join(BASE_DIR, 'data', 'seeds', 'seeds.json')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'data', 'calibration_settings.json')

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
# BIOMECHANICAL FORWARD KINEMATICS & HAND GENERATOR
# ---------------------------------------------------------

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t

def calc_finger_chain_yaw_pitch(base_pos, base_rot_mat, lengths, yaws, pitches):
    points = [base_pos.copy()]
    current_mat = base_rot_mat.copy()
    current_pos = base_pos.copy()

    for length, yaw, pitch in zip(lengths, yaws, pitches):
        joint_rot = rot_z(yaw).dot(rot_x(pitch))
        current_mat = current_mat.dot(joint_rot)
        local_bone = np.array([0, length, 0])
        world_bone = current_mat.dot(local_bone)
        current_pos = current_pos + world_bone
        points.append(current_pos.copy())

    return points

def generate_hand_3d(finger_states, spread_states, thumb_opp,
                     avg_lengths, avg_palm, ranges, stages,
                     rule_spread_constraint, rule_tendon_pinky_ring):
    """
    Generate 21 3D landmarks for a hand configuration using exact calibrated limits.
    """
    # Palm base positions (normalized static reference)
    palm_bases = {
        'Thumb':  np.array([-0.06, 0.04, 0.02]),
        'Index':  np.array([-0.08, 0.45, 0.0]),
        'Middle': np.array([ 0.00, 0.48, 0.0]),
        'Ring':   np.array([ 0.08, 0.45, 0.0]),
        'Pinky':  np.array([ 0.16, 0.38, 0.0])
    }
    for finger in palm_bases:
        direction = palm_bases[finger] / max(np.linalg.norm(palm_bases[finger]), 1e-9)
        palm_bases[finger] = direction * avg_palm[finger]

    landmarks_3d = [np.array([0.0, 0.0, 0.0])]  # Wrist
    fingers_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    # Apply Rule B (Tendon linkage ring/pinky states)
    f_states = finger_states.copy()
    if rule_tendon_pinky_ring:
        f_states['Ring'] = f_states['Pinky']

    for finger in fingers_order:
        state = str(f_states[finger])
        lengths = avg_lengths[finger]

        if finger == 'Thumb':
            thumb_lengths = {
                (0, 0): [0.0982, 0.0758, 0.0572],
                (0, 2): [0.0859, 0.0666, 0.0424],
                (0, 3): [0.0873, 0.0761, 0.0546],
                (1, 0): [0.0780, 0.0613, 0.0523],
                (1, 2): [0.0672, 0.0672, 0.0362],
                (1, 3): [0.0609, 0.0470, 0.0383]
            }
            opp_factor = float(thumb_opp)
            p_idx = 0 if state == '0' else (2 if state == '2' else 3)
            if state == '1': p_idx = 2

            lens_L0 = thumb_lengths[(0, p_idx)]
            lens_L1 = thumb_lengths[(1, p_idx)]
            lengths = [lerp(l0, l1, opp_factor) for l0, l1 in zip(lens_L0, lens_L1)]

            j1_y = stages['Thumb'][state]['J1_Yaw']
            j1_p = stages['Thumb'][state]['J1_Pitch']
            j2_y = stages['Thumb'][state]['J2_Yaw']
            j2_p = stages['Thumb'][state]['J2_Pitch']
            j3_y = stages['Thumb'][state]['J3_Yaw']
            j3_p = stages['Thumb'][state]['J3_Pitch']

            if rule_spread_constraint:
                sp_factor = max(0.0, 1.0 - (float(state) / 3.0))
                j1_y = j1_y * sp_factor

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
            j1_y = stages[finger][state]['J1_Yaw']
            j1_p = stages[finger][state]['J1_Pitch']
            j2_y = stages[finger][state]['J2_Yaw']
            j2_p = stages[finger][state]['J2_Pitch']
            j3_y = stages[finger][state]['J3_Yaw']
            j3_p = stages[finger][state]['J3_Pitch']

            j2_y = 0.0
            j3_y = 0.0
            j3_p = j2_p

            # Rule A: Spread constraint
            mi_sp = spread_states['Middle_Index']
            rm_sp = spread_states['Ring_Middle']
            pr_sp = spread_states['Pinky_Ring']
            it_sp = spread_states['Index_Thumb']

            if rule_spread_constraint:
                if f_states['Middle'] > 1 or f_states['Index'] > 1: mi_sp = 0
                if f_states['Ring'] > 1 or f_states['Middle'] > 1: rm_sp = 0
                if f_states['Pinky'] > 1 or f_states['Ring'] > 1: pr_sp = 0
                if f_states['Index'] > 1 or f_states['Thumb'] > 1: it_sp = 0

            idx_th_ang = lerp(ranges['Spread']['Index_Thumb'][0], ranges['Spread']['Index_Thumb'][1], it_sp)
            mi_ind_ang = lerp(ranges['Spread']['Middle_Index'][0], ranges['Spread']['Middle_Index'][1], mi_sp)
            rg_mi_ang = lerp(ranges['Spread']['Ring_Middle'][0], ranges['Spread']['Ring_Middle'][1], rm_sp)
            pk_rg_ang = lerp(ranges['Spread']['Pinky_Ring'][0], ranges['Spread']['Pinky_Ring'][1], pr_sp)

            if finger == 'Index':
                j1_y += mi_ind_ang * 0.5
                j1_y += idx_th_ang * 0.1
            elif finger == 'Middle':
                j1_y -= mi_ind_ang * 0.5
                j1_y += rg_mi_ang * 0.3
            elif finger == 'Ring':
                j1_y -= rg_mi_ang * 0.3
                j1_y += pk_rg_ang * 0.3
            elif finger == 'Pinky':
                j1_y -= pk_rg_ang * 0.5

            if rule_spread_constraint:
                sp_factor = max(0.0, 1.0 - (float(state) / 3.0))
                j1_y = j1_y * sp_factor

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

    # Biomechanical clinical safety filter (disabled so that user-defined limits are strictly respected)
    # if landmarks_3d[1][2] < landmarks_3d[0][2] - 0.01:
    #     return None

    return landmarks_3d

def decode_label_to_states(label):
    pinky_s = int(label[0])
    pr_spread = int(label[1])
    ring_s = int(label[2])
    rm_spread = int(label[3])
    middle_s = int(label[4])
    mi_spread = int(label[5])
    index_s = int(label[6])
    it_spread = int(label[7])
    thumb_opp = int(label[8])
    thumb_s = int(label[9])

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
    return finger_states, spread_states, thumb_opp

# ---------------------------------------------------------
# GLOBAL SPATIAL TRANSFORMS & PROJECTIONS
# ---------------------------------------------------------

def apply_global_transform(landmarks_3d, pitch_deg, yaw_deg, roll_deg):
    Rx = rot_x(pitch_deg)
    Ry = rot_y(yaw_deg)
    Rz = rot_z(roll_deg)
    R = Rz.dot(Ry).dot(Rx)

    pts_2d = []
    for pt in landmarks_3d:
        pt_rot = R.dot(pt)
        z_offset = 4.0
        z_factor = z_offset / (z_offset - pt_rot[2])
        x2d = pt_rot[0] * z_factor
        y2d = pt_rot[1] * z_factor
        pts_2d.append([x2d, y2d])

    return pts_2d

def bounce_wave(progress, cycles):
    cycle_pos = (progress * cycles) % 1.0
    if cycle_pos < 0.5:
        return (cycle_pos * 4.0) - 1.0
    else:
        return 1.0 - ((cycle_pos - 0.5) * 4.0)

def normalize_and_add_noise(pts_2d):
    xs = [p[0] for p in pts_2d]
    ys = [p[1] for p in pts_2d]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    size = max(width, height)

    normalized = []
    for x, y in pts_2d:
        nx = (x - min_x) / size
        ny = (y - min_y) / size
        # Subtle sensor noise overlay for ML generalization
        nx += random.gauss(0, 0.005)
        ny += random.gauss(0, 0.005)
        normalized.append([nx, ny])

    return normalized

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m:02d}m"

# ---------------------------------------------------------
# MAIN GENERATOR LOOP
# ---------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("=" * 60)
    logging.info("  GERADOR SINTÉTICO DE LIBRAS (Varredura Contínua 3D)")
    logging.info("=" * 60)

    # 1. Carregar calibração manual do usuário
    stages = None
    rule_spread_constraint = True
    rule_tendon_pinky_ring = True

    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                calib = json.load(f)
            stages = calib.get("stages", None)
            rule_spread_constraint = calib.get("rule_spread_constraint", True)
            rule_tendon_pinky_ring = calib.get("rule_tendon_pinky_ring", True)
            
            # Retroactive compatibility conversion for older saves in generator
            if stages is not None:
                for f in stages:
                    for s in stages[f]:
                        item = stages[f][s]
                        if 'MCP' in item and 'J1_Pitch' not in item:
                            if f == 'Thumb':
                                cy = item.get('CMC_Yaw', -25.0)
                                cp = item.get('CMC_Pitch', 5.4)
                                mcp = item.get('MCP', 10.0)
                                pip = item.get('PIP', 5.0)
                                item['J1_Yaw'] = cy
                                item['J1_Pitch'] = cp
                                item['J2_Yaw'] = 0.0
                                item['J2_Pitch'] = mcp
                                item['J3_Yaw'] = 0.0
                                item['J3_Pitch'] = pip
                            else:
                                mcp = item.get('MCP', 5.0)
                                pip = item.get('PIP', 5.0)
                                def_y = {'Index': 5.0, 'Middle': 0.0, 'Ring': -5.0, 'Pinky': -15.0}
                                item['J1_Yaw'] = def_y.get(f, 0.0)
                                item['J1_Pitch'] = mcp
                                item['J2_Yaw'] = 0.0
                                item['J2_Pitch'] = pip
                                item['J3_Yaw'] = 0.0
                                item['J3_Pitch'] = pip
            logging.info(f"Configurações de calibração carregadas de: {CALIBRATION_FILE}")
        except Exception as e:
            logging.warning(f"Erro ao carregar {CALIBRATION_FILE}: {e}. Usando padrões.")
            stages = None

    # Defaults do Spread (Spread Ranges)
    ranges = {
        'Spread': {
            'Pinky_Ring': [0.0, 20.0],
            'Ring_Middle': [0.0, 18.0],
            'Middle_Index': [0.0, 20.0],
            'Index_Thumb': [2.0, 60.0]
        }
    }

    if stages is None:
        default_ranges = {
            'Thumb':  {'MCP': [10.0, 50.0], 'PIP_DIP': [5.0, 60.0]},
            'Index':  {'MCP': [5.0, 60.0],  'PIP_DIP': [5.0, 90.0]},
            'Middle': {'MCP': [5.0, 75.0],  'PIP_DIP': [5.0, 110.0]},
            'Ring':   {'MCP': [5.0, 80.0],  'PIP_DIP': [5.0, 105.0]},
            'Pinky':  {'MCP': [5.0, 85.0],  'PIP_DIP': [5.0, 100.0]}
        }
        stages = {}
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            stages[f] = {}
            mcp_min, mcp_max = default_ranges[f]['MCP']
            pip_min, pip_max = default_ranges[f]['PIP_DIP']
            
            def_yaw = {'Thumb': -25.0, 'Index': 5.0, 'Middle': 0.0, 'Ring': -5.0, 'Pinky': -15.0}
            
            for s in range(4):
                state = str(s)
                mcp_val = mcp_min
                pip_val = pip_min
                if state == '1':
                    mcp_val = lerp(mcp_min, mcp_max, 0.15)
                    pip_val = lerp(pip_min, pip_max, 0.5)
                elif state == '2':
                    mcp_val = mcp_min
                    pip_val = pip_max
                elif state == '3':
                    mcp_val = mcp_max
                    pip_val = pip_max

                if f != 'Thumb':
                    stages[f][state] = {
                        'J1_Yaw': def_yaw[f],
                        'J1_Pitch': mcp_val,
                        'J2_Yaw': 0.0,
                        'J2_Pitch': pip_val,
                        'J3_Yaw': 0.0,
                        'J3_Pitch': pip_val
                    }
                else:
                    cy_stages = {'0': -25.0, '1': -31.6, '2': -36.1, '3': -21.2}
                    cp_stages = {'0': 5.4, '1': 14.5, '2': 21.1, '3': 37.3}
                    stages[f][state] = {
                        'J1_Yaw': cy_stages[state],
                        'J1_Pitch': cp_stages[state],
                        'J2_Yaw': 0.0,
                        'J2_Pitch': mcp_val,
                        'J3_Yaw': 0.0,
                        'J3_Pitch': pip_val
                    }
        logging.info("Usando limites anatômicos padrões (sem arquivo de calibração).")

    # Proporções anatômicas padrão
    avg_lengths = {
        'Thumb':  [0.0914, 0.0771, 0.0621],
        'Index':  [0.0998, 0.0640, 0.0532],
        'Middle': [0.1102, 0.0769, 0.0578],
        'Ring':   [0.1001, 0.0700, 0.0553],
        'Pinky':  [0.0768, 0.0517, 0.0454]
    }
    avg_palm = {
        'Thumb': 0.070, 'Index': 0.240, 'Middle': 0.245, 'Ring': 0.235, 'Pinky': 0.210
    }

    # Se seeds.json existir, herda as proporções
    if os.path.exists(SEEDS_FILE):
        try:
            with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            metadata = seeds.get("__metadata__", {})
            if "avg_lengths" in metadata:
                avg_lengths = metadata["avg_lengths"]
            if "avg_palm" in metadata:
                avg_palm = metadata["avg_palm"]
            logging.info("Metadados anatômicos herdados do seeds.json.")
        except Exception as e:
            pass

    # 2. Gerar a lista de labels dinâmicas seguindo estritamente as regras ativas
    logging.info("Gerando combinações de poses válidas biomecanicamente...")
    
    def get_valid_labels(rule_spread_constraint, rule_tendon_pinky_ring):
        valid_keys = []
        pinky_states = [0, 1, 2, 3]
        pr_spreads = [0, 1]
        ring_states = [0, 1, 2, 3]
        rm_spreads = [0, 1]
        middle_states = [0, 1, 2, 3]
        mi_spreads = [0, 1]
        index_states = [0, 1, 2, 3]
        it_spreads = [0, 1]
        thumb_opps = [0, 1]
        thumb_states = [0, 2, 3]
        
        for pinky_s in pinky_states:
            for pr_sp in pr_spreads:
                for ring_s in ring_states:
                    # Regra B: compartilhamento de tendão
                    if rule_tendon_pinky_ring and ring_s != pinky_s:
                        continue
                    # Regra A: spreads inválidos se flexionados
                    if rule_spread_constraint and pr_sp == 1 and (pinky_s > 1 or ring_s > 1):
                        continue
                    for rm_sp in rm_spreads:
                        for middle_s in middle_states:
                            if rule_spread_constraint and rm_sp == 1 and (ring_s > 1 or middle_s > 1):
                                continue
                            for mi_sp in mi_spreads:
                                for index_s in index_states:
                                    if rule_spread_constraint and mi_sp == 1 and (middle_s > 1 or index_s > 1):
                                        continue
                                    for it_sp in it_spreads:
                                        for thumb_opp in thumb_opps:
                                            for thumb_s in thumb_states:
                                                if rule_spread_constraint and it_sp == 1 and (index_s > 1 or thumb_s > 1):
                                                    continue
                                                key = f"{pinky_s}{pr_sp}{ring_s}{rm_sp}{middle_s}{mi_sp}{index_s}{it_sp}{thumb_opp}{thumb_s}"
                                                valid_keys.append(key)
        return valid_keys

    seeds_labels = get_valid_labels(rule_spread_constraint, rule_tendon_pinky_ring)
    total_seeds = len(seeds_labels)
    
    # 600 frames por classe é a densidade ideal para a varredura contínua cobrir toda a esfera
    SAMPLES_PER_STATE = 600
    total_samples_expected = total_seeds * SAMPLES_PER_STATE

    logging.info(f"Combinações biomecânicas válidas: {total_seeds}")
    logging.info(f"Amostras por classe: {SAMPLES_PER_STATE}")
    logging.info(f"Total esperado: {total_samples_expected:,} amostras")
    logging.info("-" * 60)

    total_generated = 0
    start_time = time.time()
    last_log_time = start_time
    LOG_INTERVAL_CLASSES = max(1, total_seeds // 50)

    for idx, label in enumerate(seeds_labels):
        finger_states, spread_states, thumb_opp_state = decode_label_to_states(label)
        
        lms_3d = generate_hand_3d(
            finger_states, spread_states, thumb_opp_state,
            avg_lengths, avg_palm, ranges, stages,
            rule_spread_constraint, rule_tendon_pinky_ring
        )

        if lms_3d is None:
            continue

        # Mover o pulso para a origem (0,0,0)
        wrist = lms_3d[0].copy()
        lms_3d = [p - wrist for p in lms_3d]

        label_dataset = {
            "metadata": {"label": label, "samples": SAMPLES_PER_STATE},
            "frames": []
        }

        valid_samples = 0
        while valid_samples < SAMPLES_PER_STATE:
            progress = valid_samples / float(SAMPLES_PER_STATE)

            # Varredura Contínua 3D
            target_pitch = bounce_wave(progress, 1) * 65.0
            target_yaw = bounce_wave(progress, 2) * 65.0
            target_roll = progress * 360.0 * 2

            lms_2d = apply_global_transform(lms_3d, target_pitch, target_yaw, target_roll)
            lms_final = normalize_and_add_noise(lms_2d)

            label_dataset["frames"].append({
                "label": label,
                "landmarks": lms_final
            })
            valid_samples += 1
            total_generated += 1

        # Salvar o dataset da classe
        label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        label_file = os.path.join(label_dir, "data.json")
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(label_dataset, f, separators=(',', ':'))

        # Log progress
        now = time.time()
        classes_done = idx + 1
        should_log = (classes_done % LOG_INTERVAL_CLASSES == 0 or
                      classes_done == total_seeds or
                      (now - last_log_time) >= 5.0)

        if should_log:
            elapsed = now - start_time
            pct = (classes_done / total_seeds) * 100.0
            rate = total_generated / max(elapsed, 0.001)

            if classes_done < total_seeds:
                eta_seconds = (elapsed / classes_done) * (total_seeds - classes_done)
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "0s"

            print(f"\r[GERAÇÃO] {classes_done}/{total_seeds} classes "
                  f"({pct:.1f}%) | "
                  f"{total_generated:,} amostras | "
                  f"{rate:,.0f} amostras/s | "
                  f"Tempo: {format_time(elapsed)} | "
                  f"ETA: {eta_str} | "
                  f"Último: {label}",
                  end="", flush=True)
            last_log_time = now

    total_elapsed = time.time() - start_time
    print()
    logging.info("=" * 60)
    logging.info("  GERAÇÃO CONCLUÍDA!")
    logging.info("=" * 60)
    logging.info(f"Classes processadas: {total_seeds}")
    logging.info(f"Total de amostras geradas: {total_generated:,}")
    logging.info(f"Tempo total: {format_time(total_elapsed)}")
    logging.info(f"Velocidade média: {total_generated / max(total_elapsed, 0.001):,.0f} amostras/s")
    logging.info(f"Datasets salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
