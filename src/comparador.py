import json
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt

# --- Configurações ---
POSE_LANDMARKS_COUNT = 33
HAND_LANDMARKS_COUNT = 21

PASS_THRESHOLD = 61.0 # (Libras v9.5) - Produção: 100% Especificidade & Recuperação de Teste 2
OCCLUSION_WINDOW = 15 # (Phase 1 Final) - Tolerância de oclusão calibrada

# Caminhos dinâmicos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def numpy_sma_points(points_list, expected_size, window=5):
    """Aplica uma Média Móvel Simples ignorando NaNs sobre uma série de vetores 3D"""
    if not points_list: return []
    arr = []
    for pts in points_list:
        if not pts: arr.append(np.full((expected_size, 3), np.nan))
        else:
            tmp = np.full((expected_size, 3), np.nan)
            for p in pts:
                idx = p.get('id')
                if idx is not None and 0 <= idx < expected_size:
                    tmp[idx] = [p['x'], p['y'], p['z']]
            arr.append(tmp)
    arr = np.array(arr)
    smoothed = np.copy(arr)
    for i in range(len(arr)):
        start, end = max(0, i - window // 2), min(len(arr), i + window // 2 + 1)
        with np.errstate(all='ignore'):
            block_mean = np.nanmean(arr[start:end], axis=0)
        smoothed[i] = np.where(np.isnan(block_mean), arr[i], block_mean)
    out_list = []
    for i, pts in enumerate(points_list):
        if not pts: out_list.append(pts); continue
        new_pts = []
        for p in pts:
            idx = p.get('id')
            if idx is not None and 0 <= idx < expected_size and not np.isnan(smoothed[i][idx]).any():
                new_p = p.copy()
                new_p['x'], new_p['y'], new_p['z'] = map(float, smoothed[i][idx])
                new_pts.append(new_p)
            else: new_pts.append(p)
        out_list.append(new_pts)
    return out_list

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_landmark_array(landmarks_list, count):
    """Converte lista de dicts landmarks para array numpy (N, 3). Nan se vazio."""
    arr = np.full((count, 3), np.nan)
    if not landmarks_list:
        return arr
    for lm in landmarks_list:
        idx = lm.get('id')
        if idx is not None and 0 <= idx < count:
            arr[idx] = [lm['x'], lm['y'], lm['z']]
    return arr

def calculate_angle_3d(a, b, c):
    """Calcula o ângulo em 3D no ponto b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def normalize_spatial(frame_data):
    """
    Retorna (pose_arr, left_hand_arr, right_hand_arr) normalizados.
    Âncora: Ponto médio entre ombros (2D).
    Escala: Largura dos ombros (2D).
    """
    pose_obj = frame_data.get("pose", [])
    if not pose_obj:
        return None, None, None
    pose_arr = get_landmark_array(pose_obj[0].get("landmarks", []), POSE_LANDMARKS_COUNT)
    
    p_left, p_right = pose_arr[11], pose_arr[12]
    if np.isnan(p_left).any() or np.isnan(p_right).any():
        return None, None, None
        
    anchor = (p_left + p_right) / 2.0
    # Mantemos o Z da âncora para centralizar o corpo em profundidade
    
    dist_shoulders = np.linalg.norm(p_left - p_right) # 3D distance
    scale = dist_shoulders if dist_shoulders > 1e-6 else 1.0
        
    norm_pose = (pose_arr - anchor) / scale
    norm_left = np.full((HAND_LANDMARKS_COUNT, 3), np.nan)
    norm_right = np.full((HAND_LANDMARKS_COUNT, 3), np.nan)
    
    for hand in frame_data.get("hands", []):
        lbl = hand.get("handedness")
        arr = get_landmark_array(hand.get("landmarks", []), HAND_LANDMARKS_COUNT)
        norm_arr = (arr - anchor) / scale
        if lbl == "Left": 
            norm_left = norm_arr
        elif lbl == "Right": 
            norm_right = norm_arr
            
    return norm_pose, norm_left, norm_right

def get_hand_matrix(hand_arr):
    """
    Calcula uma Matriz 21x21 de Distâncias Topológicas.
    Essa matriz contém a distância Euclidiana de CADA ponto da mão para TODOS os outros pontos.
    Garante imunidade total à rotação e espelhamento da câmera.
    """
    if np.isnan(hand_arr).all(): 
        return np.zeros((HAND_LANDMARKS_COUNT, HAND_LANDMARKS_COUNT))
        
    p1 = hand_arr[:, np.newaxis, :]
    p2 = hand_arr[np.newaxis, :]
    
    # Matriz 21x21 de distâncias
    dist_matrix = np.linalg.norm(p1 - p2, axis=2)
    return dist_matrix

# Definição dos dedos: (MCP, PIP, DIP, Tip)
FINGER_JOINTS = [
    (1, 2, 3, 4),    # Thumb
    (5, 6, 7, 8),    # Index
    (9, 10, 11, 12), # Middle
    (13, 14, 15, 16),# Ring
    (17, 18, 19, 20)  # Pinky
]

def get_finger_angles(hand_arr):
    """
    Extrai 20 ângulos articulares 3D da mão:
    - 15 ângulos inter-falangeanos (3 por dedo: MCP, PIP, DIP)
    - 5 ângulos de inclinação do dedo em relação à palma
    Retorna array de 20 ângulos em graus. Imune a distorções de perspectiva no eixo Z.
    """
    if np.isnan(hand_arr).all():
        return np.zeros(20)
    
    angles = []
    wrist = hand_arr[0]
    
    for finger_idx, (mcp, pip, dip, tip) in enumerate(FINGER_JOINTS):
        pts = [hand_arr[mcp], hand_arr[pip], hand_arr[dip], hand_arr[tip]]
        
        # Se algum ponto é NaN, ângulos = 0
        if any(np.isnan(p).any() for p in pts):
            angles.extend([0.0, 0.0, 0.0, 0.0])
            continue
        
        # Ângulo no MCP (entre wrist/mcp e mcp/pip)
        a_mcp = calculate_angle_3d(wrist if not np.isnan(wrist).any() else pts[0], pts[0], pts[1])
        # Ângulo no PIP (entre mcp/pip e pip/dip)
        a_pip = calculate_angle_3d(pts[0], pts[1], pts[2])
        # Ângulo no DIP (entre pip/dip e dip/tip)
        a_dip = calculate_angle_3d(pts[1], pts[2], pts[3])
        
        # Inclinação do dedo em relação à palma (vetor MCP->Tip vs vetor Wrist->MCP)
        v_finger = pts[3] - pts[0]  # MCP -> Tip
        v_palm = pts[0] - wrist     # Wrist -> MCP
        n_f = np.linalg.norm(v_finger)
        n_p = np.linalg.norm(v_palm)
        if n_f > 1e-6 and n_p > 1e-6:
            cos_inc = np.clip(np.dot(v_finger, v_palm) / (n_f * n_p), -1.0, 1.0)
            a_inc = np.degrees(np.arccos(cos_inc))
        else:
            a_inc = 0.0
        
        angles.extend([a_mcp, a_pip, a_dip, a_inc])
    
    return np.array(angles)

def get_palm_normal(hand_arr):
    """
    Calcula o vetor normal da palma da mão usando os pontos:
    0 (Pulso), 5 (Base do Indicador) e 17 (Base do Mindinho).
    Retorna um vetor unitário (3D).
    """
    if np.isnan(hand_arr).all():
        return np.zeros(3)
    
    # Pontos de referência para o plano da palma
    wrist = hand_arr[0]
    index_mcp = hand_arr[5]
    pinky_mcp = hand_arr[17]
    
    if np.isnan(wrist).any() or np.isnan(index_mcp).any() or np.isnan(pinky_mcp).any():
        return np.zeros(3)
    
    # Vetores que definem o plano
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    
    # Produto vetorial para achar a normal
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    
    if norm < 1e-6:
        return np.zeros(3)
        
    return normal / norm # Vetor unitário

class KeyFrameData:
    """ Representa o Esqueleto Ativo do Corpo em um Frame """
    def __init__(self, frame_data, idx, time_ms, fs_l=0, fs_r=0):
        self.idx = idx
        self.time_ms = time_ms
        self.frames_since_l = fs_l
        self.frames_since_r = fs_r
        self.prev_kf = None # Atribuído na extração
        self.pose, self.left, self.right = normalize_spatial(frame_data)
        self.is_valid = self.pose is not None
        
        if not self.is_valid:
            self.pa_l = self.pa_r = np.zeros(3)
            self.in_rest = True
            return
            
        # 1. Forma da Mão - Ângulos articulares (20 valores por mão)
        self.angles_l = get_finger_angles(self.left)
        self.angles_r = get_finger_angles(self.right)
        
        # Matriz Topológica (Distâncias Internas)
        self.topo_l = get_hand_matrix(self.left)
        self.topo_r = get_hand_matrix(self.right)
        
        # 2. Posição (PA) - Pulsos relativos ao ombro (âncora)
        # 2. Posição (PA) - Pulsos relativos ao ombro (âncora) em 3D
        p15 = self.pose[15]
        p16 = self.pose[16]
        self.pa_l = p15 if not np.isnan(p15).all() else np.zeros(3)
        self.pa_r = p16 if not np.isnan(p16).all() else np.zeros(3)
        self.rel_hands = self.pa_l - self.pa_r
        
        # 3. Orientação da Palma (Normal 3D) e Direção dos Dedos (Pointer)
        self.palm_l = get_palm_normal(self.left)
        self.palm_r = get_palm_normal(self.right)
        self.ptr_l = self.left[9] - self.left[0] if not np.isnan(self.left).all() else np.zeros(3)
        self.ptr_r = self.right[9] - self.right[0] if not np.isnan(self.right).all() else np.zeros(3)
        
        for p in [self.ptr_l, self.ptr_r]:
            norm = np.linalg.norm(p)
            if norm > 1e-6: p /= norm
        
        # 4. Âncoras Faciais (3D)
        face_pts_indices = [0, 2, 5, 9, 10]
        face_pts = []
        for fi in face_pts_indices:
            fp = self.pose[fi] if not np.isnan(self.pose[fi]).any() else np.zeros(3)
            face_pts.append(fp)
        face_pts = np.array(face_pts)
        self.face_dist_l = np.array([np.linalg.norm(self.pa_l - fp) for fp in face_pts])
        self.face_dist_r = np.array([np.linalg.norm(self.pa_r - fp) for fp in face_pts])
        
        # 5. Estado de Repouso
        hip_y = (self.pose[23][1] + self.pose[24][1]) / 2.0
        self.in_rest = (self.pa_l[1] > hip_y - 0.1) and (self.pa_r[1] > hip_y - 0.1)

def extract_all_keyframes(frames_data):
    feats = []
    last_idx_l = -9999
    last_idx_r = -9999
    
    for i, f in enumerate(frames_data):
        # Pré-checagem rápida de presença para a memória
        pose_norm, left, right = normalize_spatial(f)
        if left is not None and not np.isnan(left).all(): last_idx_l = i
        if right is not None and not np.isnan(right).all(): last_idx_r = i
        
        kf = KeyFrameData(f, i, f.get("timestamp_ms", i*33), fs_l=i-last_idx_l, fs_r=i-last_idx_r)
        if feats: kf.prev_kf = feats[-1]
        feats.append(kf)
    return feats

def extract_base_sequence(base_feats):
    valid_idxs = [i for i, f in enumerate(base_feats) if f.is_valid and not f.in_rest]
    if not valid_idxs: return []
    sequence = [base_feats[idx] for i, idx in enumerate(valid_idxs) if i % 3 == 0]
    if valid_idxs[-1] != sequence[-1].idx:
        sequence.append(base_feats[valid_idxs[-1]])
    return sequence

def calc_keyframe_score(b_kf, t_kf):
    # 0. Sincronia de Mãos (Hand Presence Memory)
    b_has_l = not np.isnan(b_kf.left).all()
    b_has_r = not np.isnan(b_kf.right).all()
    t_has_l = not np.isnan(t_kf.left).all()
    t_has_r = not np.isnan(t_kf.right).all()
    
    # Memória Temporal: Perdoa oclusão se a base espera a mão
    t_has_l_mem = t_has_l or (t_kf.frames_since_l <= OCCLUSION_WINDOW)
    t_has_r_mem = t_has_r or (t_kf.frames_since_r <= OCCLUSION_WINDOW)
    
    pres_penalty = 0.0
    # Se a base espera a mão e o alvo não a tem (nem na memória) -> Penalidade
    if b_has_l and not t_has_l_mem: pres_penalty += 10.0
    if b_has_r and not t_has_r_mem: pres_penalty += 10.0
    
    # Se a base NÃO espera a mão e o alvo a tem (presença real) -> Penalidade (Mão extra)
    if not b_has_l and t_has_l: pres_penalty += 10.0
    if not b_has_r and t_has_r: pres_penalty += 10.0
    
    # 1. FORMA DA MÃO (Joint Angles) (40 pontos)
    ANGLE_TOLERANCE = 35.0 
    angle_weights = np.tile([0.8, 1.0, 1.5, 1.2], 5)
    angle_weights /= np.mean(angle_weights)
    
    def get_hand_err(b_ang, t_ang, has_b, has_t):
        if not has_b: return None
        if not has_t: return 1.0
        return np.mean(np.minimum(np.abs(t_ang - b_ang) * angle_weights / ANGLE_TOLERANCE, 1.0))

    err_l = get_hand_err(b_kf.angles_l, t_kf.angles_l, b_has_l, t_has_l)
    err_r = get_hand_err(b_kf.angles_r, t_kf.angles_r, b_has_r, t_has_r)
    
    valid_errs = [e for e in [err_l, err_r] if e is not None]
    shape_score = 40.0 * (1.0 - np.mean(valid_errs)) if valid_errs else 0.0
    
    # 2. TOPOLOGIA (Distance Matrix) (20 pontos)
    def get_topo_err(b_tp, t_tp, has_b, has_t):
        if not has_b: return None
        if not has_t: return 1.0
        return np.mean(np.abs(t_tp - b_tp)) / 0.15

    err_tp_l = get_topo_err(b_kf.topo_l, t_kf.topo_l, b_has_l, t_has_l)
    err_tp_r = get_topo_err(b_kf.topo_r, t_kf.topo_r, b_has_r, t_has_r)
    
    valid_t_errs = [np.clip(e, 0, 1) for e in [err_tp_l, err_tp_r] if e is not None]
    topo_score = 20.0 * (1.0 - np.mean(valid_t_errs)) if valid_t_errs else 0.0
    
    # 3. ORIENTAÇÃO (20 pontos)
    def get_orient_sim(b_p, b_d, t_p, t_d, has_b, has_t):
        if not has_b: return None
        if not has_t: return 0.0
        return (max(0, np.dot(b_p, t_p)) + max(0, np.dot(b_d, t_d))) / 2.0

    sim_l = get_orient_sim(b_kf.palm_l, b_kf.ptr_l, t_kf.palm_l, t_kf.ptr_l, b_has_l, t_has_l)
    sim_r = get_orient_sim(b_kf.palm_r, b_kf.ptr_r, t_kf.palm_r, t_kf.ptr_r, b_has_r, t_has_r)
    
    valid_sims = [s for s in [sim_l, sim_r] if s is not None]
    orient_score = 20.0 * np.mean(valid_sims) if valid_sims else 0.0
    
    # 4. POSIÇÃO (3D Context) (15 pontos)
    def get_pos_err(b_p, t_p, has_b, has_t):
        if not has_b: return None
        if not has_t: return 2.5
        # v9.6 Phase 3: Decomposição XY vs Z (Peso 1.0 vs 0.2)
        diff = t_p - b_p
        err_2d = np.sum(diff[:2]**2)
        err_z = diff[2]**2
        return np.sqrt(err_2d + 0.2 * err_z)

    pe_l = get_pos_err(b_kf.pa_l, t_kf.pa_l, b_has_l, t_has_l)
    pe_r = get_pos_err(b_kf.pa_r, t_kf.pa_r, b_has_r, t_has_r)
    
    valid_p_errs = [e for e in [pe_l, pe_r] if e is not None]
    w_e = np.mean(valid_p_errs) if valid_p_errs else 0.0
    
    r_diff = (t_kf.pa_l - t_kf.pa_r) - (b_kf.pa_l - b_kf.pa_r)
    r_e = np.sqrt(np.sum(r_diff[:2]**2) + 0.2 * r_diff[2]**2) if b_has_l and b_has_r and t_has_l and t_has_r else 0.0
    
    f_e_l = np.mean(np.abs(t_kf.face_dist_l - b_kf.face_dist_l)) if b_has_l and t_has_l else (0.5 if b_has_l else None)
    f_e_r = np.mean(np.abs(t_kf.face_dist_r - b_kf.face_dist_r)) if b_has_r and t_has_r else (0.5 if b_has_r else None)
    valid_f_errs = [e for e in [f_e_l, f_e_r] if e is not None]
    f_e = np.mean(valid_f_errs) if valid_f_errs else 0.0
    
    pos_score = max(0.0, 15.0 * (1.0 - (0.6*w_e + 0.2*r_e + 0.2*f_e) / 3.5)) # Divisor 3.5
    
    # Detalhar componentes para debug
    p_err_avg = np.mean(valid_t_errs) if valid_t_errs else 0.0
    pos_details = {"wrist": float(w_e), "relay": float(r_e), "face": float(f_e), "topo": float(p_err_avg)}
    
    final_net = shape_score + orient_score + pos_score + topo_score - pres_penalty
    return final_net, shape_score, pos_score, pos_details

def sequence_alignment_dtw(base_sequence, target_feats):
    if not base_sequence or not target_feats: return 0.0, [], []
    N_b, N_t = len(base_sequence), len(target_feats)
    
    score_matrix = np.zeros((N_b, N_t))
    details_matrix = [[(0,0,{}) for _ in range(N_t)] for _ in range(N_b)]
    for i in range(N_b):
        for j in range(N_t):
            if target_feats[j].is_valid:
                s, shp, pos, p_det = calc_keyframe_score(base_sequence[i], target_feats[j])
                score_matrix[i, j], details_matrix[i][j] = s, (shp, pos, p_det)
                
    dp = np.full((N_b, N_t), -1.0)
    traceback = np.full((N_b, N_t), -1, dtype=int)
    mov_tracker = np.zeros((N_b, N_t)) # Para salvar no report
    
    for j in range(N_t): dp[0, j] = score_matrix[0, j]
        
    for i in range(1, N_b):
        expected_jump = base_sequence[i].idx - base_sequence[i-1].idx
        max_wait = int(expected_jump * 4.0 + 30)
        
        for j in range(1, N_t):
            best_val, best_k, best_mov = -1.0, -1, 0.0
            for k in range(max(0, j - max_wait), j):
                if dp[i-1, k] < 0: continue
                
                # Movimento v9.1.2: Deltas entre keyframes subsequentes
                db_l, db_r = base_sequence[i].pa_l - base_sequence[i-1].pa_l, base_sequence[i].pa_r - base_sequence[i-1].pa_r
                dt_l, dt_r = target_feats[j].pa_l - target_feats[k].pa_l, target_feats[j].pa_r - target_feats[k].pa_r
                
                def get_sim(v1, v2):
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 < 0.02 and n2 < 0.02: return 1.0
                    if n1 < 0.02 or n2 < 0.02: return 0.0
                    return max(0.0, np.dot(v1, v2) / (n1 * n2))
                
                # Ponderação pela atividade na base
                vb = np.linalg.norm(db_l) + np.linalg.norm(db_r)
                wl, wr = (np.linalg.norm(db_l)/vb, np.linalg.norm(db_r)/vb) if vb > 1e-6 else (0.5, 0.5)
                m_score = 5.0 * (get_sim(db_l, dt_l)*wl + get_sim(db_r, dt_r)*wr)
                
                fj = j - k
                fgap = max(0, fj - (expected_jump + 1))
                # Penalidade suave logarítmica (v9.6 Phase 2)
                gap_pen = 5.0 * np.log1p(fgap)
                
                cand = dp[i-1, k] + score_matrix[i, j] + m_score - gap_pen
                if cand > best_val:
                    best_val, best_k, best_mov = cand, k, m_score
            
            if best_k != -1:
                dp[i, j], traceback[i, j], mov_tracker[i, j] = best_val, best_k, best_mov
                
    best_last_j = np.argmax(dp[-1])
    if dp[-1, best_last_j] < 0: return 0.0, ["Falha no alinhamento."], []
    
    path = []
    curr_j = best_last_j
    for i in range(N_b - 1, -1, -1):
        path.append((i, curr_j))
        curr_j = traceback[i, curr_j]
    path.reverse()
    
    total_raw, matched_frames, details = 0.0, [], []
    for i, (b_i, t_j) in enumerate(path):
        score = score_matrix[b_i, t_j]
        shp, pos, p_det = details_matrix[b_i][t_j]
        mov = mov_tracker[b_i, t_j]
        gap_pen = 0.0
        if i > 0:
            jump = t_j - path[i-1][1]
            exp_j = base_sequence[b_i].idx - base_sequence[b_i-1].idx
            fgap = max(0, jump - (exp_j + 1))
            # Penalidade suave logarítmica (v9.6 Phase 2)
            gap_pen = 5.0 * np.log1p(fgap)
            
        net = score + mov - gap_pen
        total_raw += max(0, net)
        matched_frames.append({
            "base_frame": int(base_sequence[b_i].idx), "target_frame": int(target_feats[t_j].idx),
            "raw_score": float(score + mov), "gap_penalty": float(gap_pen), "net_score": float(net),
            "shape_score": float(shp), "position_score": float(pos), "movement_score": float(mov),
            "pos_details": p_det
        })
        details.append(f"KF {base_sequence[b_i].idx:>3}->{target_feats[t_j].idx:>3} | Net: {net:>4.1f} | Shp: {shp:>4.1f} | Mov: {mov:>4.1f} | Gap: -{gap_pen:>4.1f}")
        
    return total_raw / N_b, details, matched_frames

def get_sign_name(fn):
    fn = fn.lower()
    if 'abacate' in fn and 'abacaxi' not in fn and 'abobrinha' not in fn: return 'abacate'
    if 'abacaxi' in fn: return 'abacaxi'
    if 'abobrinha' in fn: return 'abobrinha'
    if 'amigo' in fn: return 'amigo'
    if 'aprender' in fn: return 'aprender'
    if 'trabalhar' in fn: return 'trabalhar'
    parts = fn.split('_')
    for p in parts:
        if p not in ['holistic', 'landmarker', 'lm', 'base', 'teste1', 'teste2', 'ruim', 'boa', 'upscaler-1080p', 'norm', 'json', 'mp4']:
            return p
    return 'unknown'

def main():
    print("\n" + "="*80)
    print("  COMPARADOR BATCH 6.0: SEQUENCE ALIGNMENT (DTW) - TOPOLOGICAL MATRIX")
    print("="*80 + "\n")
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    json_dir = os.path.join(RESULTS_DIR, "json")
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
        
    all_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    bases = [f for f in all_files if f.endswith("_norm.json") and "_base" in os.path.basename(f)]
    targets = all_files
    
    if not bases:
        print("[!] Nenhuma BASE encontrada (*_norm.json) em data/")
        return
    if not targets:
        print("[!] Nenhum TARGET encontrado (*.json) em data/")
        return
        
    print(f"[*] Bases encontradas:   {len(bases)}")
    print(f"[*] Targets encontrados: {len(targets)}\n")
    
    report_lines = []
    report_lines.append("="*90)
    report_lines.append(" RELATÓRIO DO MOTOR DTW (DYNAMIC TIME WARPING) E MATRIZ TOPOLÓGICA ".center(90))
    report_lines.append("="*90)
    report_lines.append(f"Pass Threshold: {PASS_THRESHOLD}/100\n")
    
    tp, fn_count, tn, fp = 0, 0, 0, 0
    total_comps = 0
    start_time = time.time()
    
    for base_path in bases:
        base_name = os.path.basename(base_path)
        base_data = load_json(base_path)
        b_frames = base_data.get("frames", [])
        if not b_frames: 
            continue
        
        base_feats = extract_all_keyframes(b_frames)
        base_seq = extract_base_sequence(base_feats)
        
        if not base_seq:
            print(f"[-] Base {base_name} ignorada (Sem Keyframes ativos).")
            continue
            
        print(f"\n[BASE] {base_name} ({len(base_seq)} Keyframes extraídos)")
        report_lines.append("\n" + "-"*90)
        report_lines.append(f"BASE: {base_name}")
        report_lines.append("-" * 90)
        report_lines.append(f"{'STATUS':<10} | {'SCORE':<8} | {'TARGET':<60}")
        report_lines.append("-" * 90)
        
        b_sign = get_sign_name(base_name)
        
        for target_path in targets:
            target_name = os.path.basename(target_path)
            
            target_data = load_json(target_path)
            t_frames_raw = target_data.get("frames", [])
            
            # --- Suavização SMA do Target (Ritmo v9.1) ---
            # Aplicamos smoothing on-the-fly APENAS se o target for RAW (não _norm.json)
            if not target_name.endswith("_norm.json"):
                raw_pose = [f.get("pose")[0].get("landmarks", []) if f.get("pose") else [] for f in t_frames_raw]
                raw_left = [next((h.get("landmarks", []) for h in f.get("hands", []) if h.get("handedness") == "Left"), []) for f in t_frames_raw]
                raw_right = [next((h.get("landmarks", []) for h in f.get("hands", []) if h.get("handedness") == "Right"), []) for f in t_frames_raw]

                smooth_pose = numpy_sma_points(raw_pose, POSE_LANDMARKS_COUNT)
                smooth_left = numpy_sma_points(raw_left, HAND_LANDMARKS_COUNT)
                smooth_right = numpy_sma_points(raw_right, HAND_LANDMARKS_COUNT)

                t_frames = []
                for i in range(len(t_frames_raw)):
                    t_frames.append({
                        "timestamp_ms": t_frames_raw[i].get("timestamp_ms", 0),
                        "pose": [{"landmarks": smooth_pose[i]}],
                        "hands": [
                            {"handedness": "Left", "landmarks": smooth_left[i]},
                            {"handedness": "Right", "landmarks": smooth_right[i]}
                        ]
                    })
            else:
                t_frames = t_frames_raw

            target_feats = extract_all_keyframes(t_frames)
            
            score, details, matched_frames = sequence_alignment_dtw(base_seq, target_feats)
            
            status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
            
            t_sign = get_sign_name(target_name)
            is_same = (b_sign == t_sign)
            if is_same:
                if status == "PASS": tp += 1
                else: fn_count += 1
            else:
                if status != "PASS": tn += 1
                else: fp += 1
                
            total_comps += 1
            
            base_seq_indices = [kf.idx for kf in base_seq]
            
            output_json = os.path.join(json_dir, f"{base_name}_vs_{target_name}.json")
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump({
                    "base": base_path,
                    "target": target_path,
                    "score": score,
                    "status": status,
                    "base_keyframes": base_seq_indices,
                    "matches": matched_frames
                }, f, indent=4)
            
            t_str = (target_name[:57] + '..') if len(target_name) > 60 else target_name
            report_lines.append(f"{status:<10} | {score:<8.1f} | {t_str:<60}")

    exec_time = time.time() - start_time
    
    stats_text = [
        "",
        "="*90,
        " RESUMO ESTATÍSTICO DE CONFIABILIDADE ".center(90),
        "="*90,
        f"Tempo de Execução:        {exec_time:.2f}s",
        f"Total de Comparações:     {total_comps}",
        "-"*90,
        f"Verdadeiros Posit. (Sinal Certo, Aprovado) : {tp}",
        f"Falsos Negativos   (Sinal Certo, Reprovado): {fn_count}",
        f"Verdadeiros Negat. (Sinal Errado, Bloqueado) : {tn}",
        f"Falsos Positivos   (Sinal Errado, Vazou)   : {fp}",
        "-"*90,
        f"Precisão (100% = Zera Falsos Positivos): {tp/(tp+fp) if (tp+fp) else 0.0:.2%}",
        f"Recall   (100% = Aceita toda variação) : {tp/(tp+fn_count) if (tp+fn_count) else 0.0:.2%}",
        f"Especificidade                           : {tn/(tn+fp) if (tn+fp) else 0.0:.2%}",
        "="*90
    ]
    
    print("\n".join(stats_text))
    report_lines.extend(stats_text)

    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[+] Relatório detalhado salvo em: {report_path}")

if __name__ == "__main__":
    os.system('color')
    main()
