import json
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt

# --- Configurações ---
POSE_LANDMARKS_COUNT = 33
HAND_LANDMARKS_COUNT = 21

RESULTS_DIR = "results"
PASS_THRESHOLD = 71.0

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
    anchor[2] = 0 # Zerar Z da âncora
    
    dist_shoulders = np.linalg.norm(p_left[:2] - p_right[:2])
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

def get_semantic_zone(hand_wrist, pose_arr):
    """
    Identifica a Caixa Delimitadora Semântica (Bounding Box) onde a mão está.
    Classifica em: FACE, CHEST, BELLY, SHOULDER_L, SHOULDER_R, HIP_L, HIP_R, OUT_OF_BOUNDS
    """
    if np.isnan(hand_wrist).all() or np.isnan(pose_arr).all():
        return "UNKNOWN"
    
    hx, hy = hand_wrist[0], hand_wrist[1]
    
    # Referências do corpo normalize_spatial:
    # 0: Nariz, 11/12: Ombros, 23/24: Quadris
    nose_y = pose_arr[0][1]
    shoulder_y = (pose_arr[11][1] + pose_arr[12][1]) / 2.0
    hip_y = (pose_arr[23][1] + pose_arr[24][1]) / 2.0
    chest_y = (shoulder_y + hip_y) / 2.0
    
    shoulder_lx, shoulder_rx = pose_arr[11][0], pose_arr[12][0]
    
    if hy < nose_y + 0.15: 
        return "FACE"
    
    if hy >= nose_y + 0.15 and hy < chest_y:
        # Altura do peito/ombro, vamos checar a lateralidade
        if hx < shoulder_rx - 0.2: return "SHOULDER_R_OUT"
        if hx > shoulder_lx + 0.2: return "SHOULDER_L_OUT"
        return "CHEST"
        
    if hy >= chest_y and hy < hip_y:
        return "BELLY"
        
    if hy >= hip_y:
        if hx < shoulder_rx - 0.2: return "HIP_R_OUT"
        if hx > shoulder_lx + 0.2: return "HIP_L_OUT"
        return "HIP_REST"
        
    return "OUT_OF_BOUNDS"

def get_finger_vectors(hand_arr):
    """
    Extrai vetores direcionais (Cosseno) dos 5 dedos em vez de uma matriz rígida.
    Nós capturamos a INTENÇÃO do dedo (dobrado vs esticado) ignorando a rotação global do pulso.
    Os vetores de interesse são:
    - Thumb:  Ponta (4) -> Base (2)
    - Index:  Ponta (8) -> Base (5)
    - Middle: Ponta (12) -> Base (9)
    - Ring:   Ponta (16) -> Base (13)
    - Pinky:  Ponta (20) -> Base (17)
    """
    if np.isnan(hand_arr).all(): 
        return np.zeros((5, 3))
        
    vecs = np.zeros((5, 3))
    
    # Pares: Ponta, Base
    pairs = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
    
    for i, (tip, base) in enumerate(pairs):
        v = hand_arr[tip] - hand_arr[base]
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            vecs[i] = v / norm # Vetor unitário para focar puramente em direção/dobra
            
    return vecs

class KeyFrameData:
    """ Representa o Esqueleto Ativo do Corpo em um Frame """
    def __init__(self, frame_data, idx, time_ms, prev_pa_l=None, prev_pa_r=None):
        self.idx = idx
        self.time_ms = time_ms
        self.pose, self.left, self.right = normalize_spatial(frame_data)
        self.is_valid = self.pose is not None
        
        if not self.is_valid:
            self.vec_l = self.vec_r = np.zeros((5, 3))
            self.pa_l = self.pa_r = np.zeros(2)
            self.delta_l = self.delta_r = np.zeros(2)
            self.in_rest = True
            return
            
        # 1. Forma da Mão (Vetores Morfológicos)
        self.vec_l = get_finger_vectors(self.left)
        self.vec_r = get_finger_vectors(self.right)
        
        # 2. Posição (PA) - Pulsos relativos ao ombro (âncora)
        p15 = self.pose[15][:2]
        p16 = self.pose[16][:2]
        self.pa_l = p15 if not np.isnan(p15).all() else np.zeros(2)
        self.pa_r = p16 if not np.isnan(p16).all() else np.zeros(2)
        self.rel_hands = self.pa_l - self.pa_r
        
        # 3. Ponto de Articulação (Bounding Box Semântica)
        self.zone_l = get_semantic_zone(self.pa_l, self.pose)
        self.zone_r = get_semantic_zone(self.pa_r, self.pose)
        
        # 3. Movimento (Delta Direcional)
        self.delta_l = self.pa_l - prev_pa_l if prev_pa_l is not None else np.zeros(2)
        self.delta_r = self.pa_r - prev_pa_r if prev_pa_r is not None else np.zeros(2)
        
        # 4. Estado de Repouso
        hip_y = (self.pose[23][1] + self.pose[24][1]) / 2.0
        self.in_rest = (self.pa_l[1] > hip_y - 0.1) and (self.pa_r[1] > hip_y - 0.1)

def numpy_sma_points(points_list, expected_size, window=5):
    """Aplica uma Média Móvel Simples ignorando NaNs sobre uma série de vetores 3D"""
    if not points_list:
        return []
    
    # Substituir dicts de pontos por arrays fáceis de interpolar
    arr = []
    for pts in points_list:
        if not pts:
            arr.append(np.full((expected_size, 3), np.nan))
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
        start = max(0, i - window // 2)
        end = min(len(arr), i + window // 2 + 1)
        # Evitar np.nanmean Warning se o bloco todo for NaN
        with np.errstate(all='ignore'):
            block_mean = np.nanmean(arr[start:end], axis=0)
        # Onde a média retornou NaN (pq só tinha NaN), volta o array original (NaN)
        smoothed[i] = np.where(np.isnan(block_mean), arr[i], block_mean)
        
    # Reconstruir dicts para compatibilidade
    out_list = []
    for i, pts in enumerate(points_list):
        if not pts:
            out_list.append(pts)
            continue
            
        new_pts = []
        for p in pts:
            idx = p.get('id')
            if idx is not None and 0 <= idx < expected_size and not np.isnan(smoothed[i][idx]).any():
                new_p = p.copy()
                new_p['x'] = float(smoothed[i][idx][0])
                new_p['y'] = float(smoothed[i][idx][1])
                new_p['z'] = float(smoothed[i][idx][2])
                new_pts.append(new_p)
            else:
                new_pts.append(p)
        out_list.append(new_pts)
        
    return out_list

def extract_all_keyframes(frames_data):
    feats = []
    prev_pa_l, prev_pa_r = None, None
    for i, f in enumerate(frames_data):
        kf = KeyFrameData(f, i, f.get("timestamp_ms", i*33), prev_pa_l, prev_pa_r)
        if kf.is_valid:
            prev_pa_l, prev_pa_r = kf.pa_l, kf.pa_r
        feats.append(kf)
    return feats

def extract_base_sequence(base_feats, max_events=5):
    """
    ARQUITETURA ORIENTADA A EVENTOS:
    Extrai apenas os frames cruciais (State Machine) baseados nas curvas de movimento,
    destruindo o DTW contínuo em favor de validação discreta de pontos chave.
    - Captura o INÍCIO e o FIM absolutos do movimento.
    - Captura picos de mínima velocidade (pausas no ar / ápices da articulação).
    - Captura picos de máxima velocidade (transições fortes).
    """
    valid_idxs = [i for i, f in enumerate(base_feats) if f.is_valid and not f.in_rest]
    if not valid_idxs: 
        return []
        
    if len(valid_idxs) <= max_events:
        return [base_feats[i] for i in valid_idxs]
        
    # Calcular a velocidade global (somatória dos deltas) de cada frame ativo
    velocities = []
    for i in valid_idxs:
        kf = base_feats[i]
        v = np.linalg.norm(kf.delta_l) + np.linalg.norm(kf.delta_r)
        velocities.append(v)
        
    selected_idxs = [valid_idxs[0], valid_idxs[-1]]
    
    # Procurar por Mínimos Locais de velocidade (Pausas Táticas do sinal)
    minima = []
    for j in range(1, len(velocities)-1):
        if velocities[j] < velocities[j-1] and velocities[j] < velocities[j+1]:
            minima.append((valid_idxs[j], velocities[j]))
            
    # Dar preferência para as pausas mais "congeladas"
    minima.sort(key=lambda x: x[1])
    
    for p_idx, _ in minima:
        if len(selected_idxs) >= max_events: break
        # Não pegar frames grudados (debounce de 5 frames)
        if not any(abs(p_idx - existing) < 5 for existing in selected_idxs):
            selected_idxs.append(p_idx)
            
    # Se ainda sobrar espaço na Máquina de Estados, pegar os picos de Máxima Velocidade (Transições Cruciais)
    if len(selected_idxs) < max_events:
        maxima = []
        for j in range(1, len(velocities)-1):
             if velocities[j] > velocities[j-1] and velocities[j] > velocities[j+1]:
                 maxima.append((valid_idxs[j], velocities[j]))
                 
        maxima.sort(key=lambda x: x[1], reverse=True) # Preferir os movimentos mais explosivos
        for p_idx, _ in maxima:
             if len(selected_idxs) >= max_events: break
             if not any(abs(p_idx - existing) < 5 for existing in selected_idxs):
                 selected_idxs.append(p_idx)
                 
    selected_idxs.sort()
    return [base_feats[i] for i in selected_idxs]

def calc_keyframe_score(b_kf, t_kf):
    """
    Sistema Hierárquico de Validação de Keyframes:
    1. Forma da Mão (60% do peso)
    2. Posição no Corpo (30% do peso)
    3. Movimento (10% do peso)
    """
    # 1. FORMA DA MÃO (CONFIGURAÇÃO DE DEDOS - 0 a 60 pontos)
    # Cosine Similarity entre os vetores falange-base de cada um dos 5 dedos.
    
    def score_vectors(v_tgt, v_base):
        if np.all(v_base == 0) and np.all(v_tgt == 0): return 60.0 # Ambas nulas (OK)
        if np.all(v_base == 0) or np.all(v_tgt == 0): return 0.0   # Mão sumiu
        
        # Similaridade de cosseno de cada dedo
        dot_products = np.sum(v_tgt * v_base, axis=1) # (5,)
        norm_t = np.linalg.norm(v_tgt, axis=1)
        norm_b = np.linalg.norm(v_base, axis=1)
        
        valid = (norm_t > 0) & (norm_b > 0)
        if not np.any(valid): return 0.0
        
        cos_sims = dot_products[valid] / (norm_t[valid] * norm_b[valid])
        
        # Mapeando Cos Sim [-1.0 a 1.0] para Erro [0.0 a 1.0]
        # Se 1.0 (Mesma direção) -> Erro 0.0
        # Se 0.0 (Ortogonal) -> Erro 0.5
        # Se -1.0 (Dedo invertido) -> Erro 1.0
        errors = (1.0 - cos_sims) / 2.0
        
        avg_err = np.mean(errors)
        
        # Tolerância de 0.2 de erro angular médio (aprox 36 graus de liberdade na dobra)
        return max(0.0, 60.0 * (1.0 - (avg_err / 0.2)))
        
    shape_score_l = score_vectors(t_kf.vec_l, b_kf.vec_l)
    shape_score_r = score_vectors(t_kf.vec_r, b_kf.vec_r)
    
    # Se a mão base existe (não é 0), a respectiva mão target deve pontuar
    active_hands = 0
    total_shape = 0
    if not np.all(b_kf.vec_l == 0):
        active_hands += 1
        total_shape += shape_score_l
    if not np.all(b_kf.vec_r == 0):
        active_hands += 1
        total_shape += shape_score_r
        
    shape_score = total_shape / active_hands if active_hands > 0 else 60.0
    
    # 2. POSIÇÃO (0 a 30 pontos)
    pa_err_l = np.linalg.norm(t_kf.pa_l - b_kf.pa_l)
    pa_err_r = np.linalg.norm(t_kf.pa_r - b_kf.pa_r)
    rel_err = np.linalg.norm(t_kf.rel_hands - b_kf.rel_hands)
    
    # A posição absoluta no esqueleto varia com a postura da pessoa (mais curvada = ombro distorcido).
    # O erro relativo das mãos entre si é mais garantido para sinais de duas mãos.
    pa_err = min((pa_err_l + pa_err_r) / 2.0, rel_err)
    
    # Mapear 0.0 dist -> 30 pontos | 1.5 dist (extensão do braço) -> 0 pontos
    pos_score = max(0.0, 30.0 * (1.0 - (pa_err / 1.5)))
    
    # --- VETO SEMÂNTICO DE PONTO DE ARTICULAÇÃO ---
    # Se a zona da mão divergir da Base, aplicamos VETO na nota de posição
    # pois o sinal está sendo executado no local errado do corpo.
    if b_kf.zone_l != "UNKNOWN" and t_kf.zone_l != "UNKNOWN" and b_kf.zone_l != t_kf.zone_l:
        # Penaliza cortando 15 pontos se errou a zona da mão ativa
        pos_score = max(0.0, pos_score - 15.0)
        
    if b_kf.zone_r != "UNKNOWN" and t_kf.zone_r != "UNKNOWN" and b_kf.zone_r != t_kf.zone_r:
        pos_score = max(0.0, pos_score - 15.0)
    
    # 3. MOVIMENTO (0 a 10 pontos)
    v_b = b_kf.delta_l + b_kf.delta_r
    v_t = t_kf.delta_l + t_kf.delta_r
    norm_b = np.linalg.norm(v_b)
    norm_t = np.linalg.norm(v_t)
    
    mov_score = 10.0 # Padrão se ambos estiverem parados (perfect match de estabilidade)
    if norm_b > 0.02 and norm_t > 0.02:
        cos_sim = np.dot(v_b, v_t) / (norm_b * norm_t)
        # Cosine Similarity vai de -1 (oposto) a 1 (mesma direção)
        mov_score = 10.0 * max(0.0, (cos_sim + 1.0) / 2.0)
    elif norm_b > 0.02 and norm_t <= 0.02:
        mov_score = 0.0 # Base moveu, Target ficou parado
    elif norm_b <= 0.02 and norm_t > 0.02:
        mov_score = 0.0 # Base parada, Target moveu
        
    total_score = shape_score + pos_score + mov_score
    return total_score, shape_score, pos_score, mov_score

def sequence_alignment_dtw(base_sequence, target_feats):
    """
    Sequence Alignment usando DTW com Penalidade Não-Linear de Lacuna (Gap Penalty).
    Avalia a sequência de keyframes buscando:
    - Maior número de keyframes encaixados
    - Menor tempo de distância entre eventos
    - Dedução pesada para Gaps Consecutivos (Sinal ausente / Errado metade do ciclo)
    - Dedução leve para Gaps Alternados (Frames de baixa qualidade / falha de tracking)
    """
    if not base_sequence or not target_feats:
        return 0.0, [], []
        
    N_b = len(base_sequence)
    N_t = len(target_feats)
    
    raw_score_matrix = np.zeros((N_b, N_t))
    score_matrix = np.zeros((N_b, N_t))
    details_matrix = [[None for _ in range(N_t)] for _ in range(N_b)]
    
    for i in range(N_b):
        for j in range(N_t):
            if not target_feats[j].is_valid:
                continue
            s, shp, pos, mov = calc_keyframe_score(base_sequence[i], target_feats[j])
            raw_score_matrix[i, j] = s
            details_matrix[i][j] = (shp, pos, mov)
            
    # Iteração 27: State Persistence Layer (N-Frame Buffer)
    # Para matar o flicker do MediaPipe, um frame target só pode ter nota alta
    # se ele sustentar essa nota nos frames adjacentes (buffer temporal).
    for i in range(N_b):
        for j in range(N_t):
            # Média ao redor de j (tamanho 3)
            start_j = max(0, j - 1)
            end_j = min(N_t, j + 2)
            block = raw_score_matrix[i, start_j:end_j]
            score_matrix[i, j] = np.mean(block) if len(block) > 0 else 0.0
            
    # dp[i, j] armazena o melhor score acumulado alinhando base(0..i) até target(j)
    dp = np.full((N_b, N_t), -1.0)
    traceback = np.full((N_b, N_t), -1, dtype=int)
    
    # gap_sizes[i, j] rastreia quantos frames puros (gaps consecutivos) no target pulamos desde o último keyframe da base
    gap_sizes = np.zeros((N_b, N_t), dtype=int)
    
    for j in range(N_t):
        dp[0, j] = score_matrix[0, j]
        
    for i in range(1, N_b):
        time_diff_base = base_sequence[i].idx - base_sequence[i-1].idx
        max_chronological_wait = int(time_diff_base * 4.0 + 30) # Tolerância longa, mas com penalidade de Gap
        
        for j in range(1, N_t):
            start_k = max(0, j - max_chronological_wait)
            best_val = -float('inf')
            best_k = -1
            best_gap = 0
            
            for k in range(start_k, j):
                if dp[i-1, k] < 0:
                    continue
                    
                # O "Gap" é a quantidade de frames que o Target avançou sem encaixar nenhum keyframe da Base
                frame_gap = (j - k) - 1
                
                # Penalidade Não-Linear:
                # 0 a 3 frames de gap (Lentidão natural ou frame piscante alternado): Penalidade baixa (-1 pts por frame)
                # > 3 frames de gap (Sequência toda perdida / não executada): Penalidade Exponencial (Sinal Inválido)
                if frame_gap <= 3:
                    gap_penalty = frame_gap * 1.5 
                else:
                    gap_penalty = frame_gap * 5.0 # Peso punitivo gigante
                    
                candidate_score = dp[i-1, k] + score_matrix[i, j] - gap_penalty
                
                if candidate_score > best_val:
                    best_val = candidate_score
                    best_k = k
                    best_gap = frame_gap
                    
            if best_k != -1:
                dp[i, j] = best_val
                traceback[i, j] = best_k
                gap_sizes[i, j] = best_gap
                
    best_last_j = np.argmax(dp[-1])
    max_score = dp[-1, best_last_j]
    
    if max_score < 0:
        return 0.0, ["Falha catastrófica no alinhamento: Gaps massivos ou formato reprovado."], []
        
    path = []
    curr_j = best_last_j
    for i in range(N_b - 1, -1, -1):
        path.append((i, curr_j))
        curr_j = traceback[i, curr_j]
    path.reverse()
    
    total_raw_score = 0.0
    details = []
    matched_frames = []
    
    for i, (b_i, t_j) in enumerate(path):
        score = score_matrix[b_i, t_j]
        shp, pos, mov = details_matrix[b_i][t_j]
        gap_deduction = 0.0
        
        if i > 0:
            prev_t_j = path[i-1][1]
            fgap = (t_j - prev_t_j) - 1
            if fgap <= 3: gap_deduction = fgap * 1.5
            else: gap_deduction = fgap * 5.0
            
        final_node_score = score - gap_deduction
        total_raw_score += max(0.0, final_node_score) # Evitar que penalidade zere o histórico de acertos inteiramente
        
        b_idx = base_sequence[b_i].idx
        t_idx_real = target_feats[t_j].idx
        matched_frames.append({
            "base_frame": int(b_idx),
            "target_frame": int(t_idx_real),
            "raw_score": float(score),
            "gap_penalty": float(gap_deduction),
            "net_score": float(final_node_score),
            "shape_score": float(shp),
            "position_score": float(pos),
            "movement_score": float(mov)
        })
        details.append(f"KF Base {b_idx:>3} -> Encaixado no Target {t_j:>3} | Score: {score:>4.1f} | Gap Pen: -{gap_deduction:>4.1f} | Net: {final_node_score:>4.1f}/100 [Shape: {shp:>4.1f}]")
        
    final_score = total_raw_score / N_b
    return final_score, details, matched_frames

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
        
    all_files = glob.glob("data/*.json")
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
        
        tp_scores = []
        fp_scores = []
        
        
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
            t_frames = target_data.get("frames", [])
            target_feats = extract_all_keyframes(t_frames)
            
            score, details, matched_frames = sequence_alignment_dtw(base_seq, target_feats)
            
            status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
            
            t_sign = get_sign_name(target_name)
            is_same = (b_sign == t_sign)
            if is_same:
                if status == "PASS": 
                    tp += 1
                    tp_scores.append(score)
                else: 
                    fn_count += 1
                    tp_scores.append(score)
            else:
                if status != "PASS": 
                    tn += 1
                    fp_scores.append(score)
                else: 
                    fp += 1
                    fp_scores.append(score)
                
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
    
    min_tp = min(tp_scores) if tp_scores else 0.0
    max_fp = max(fp_scores) if fp_scores else 0.0
    avg_tp = sum(tp_scores) / len(tp_scores) if tp_scores else 0.0
    avg_fp = sum(fp_scores) / len(fp_scores) if fp_scores else 0.0
    margin = avg_tp - avg_fp
    critical_margin = min_tp - max_fp
    
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
        f"Menor Nota (Sinal Certo) : {min_tp:.1f}",
        f"Maior Nota (Sinal Errado): {max_fp:.1f}",
        f"MÉDIA Sinais Certos (TP) : {avg_tp:.1f}",
        f"MÉDIA Sinais Errados(FP) : {avg_fp:.1f}",
        f"MARGEM GERAL (Avg_TP - Avg_FP): {margin:.1f}",
        f"MARGEM CRÍTICA (Min_TP - Max_FP): {critical_margin:.1f}",
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
