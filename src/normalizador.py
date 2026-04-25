
import json
import numpy as np
import os
import glob
import sys

# --- Configurações ---
POSE_LANDMARKS_COUNT = 33
HAND_LANDMARKS_COUNT = 21
L_SHOULDER = 11
R_SHOULDER = 12

# Caminhos dinâmicos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

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

def normalize_spatial(frame_data):
    """
    Retorna (pose_arr, left_hand_arr, right_hand_arr) normalizados.
    Âncora: Ponto médio entre ombros.
    Escala: Largura dos ombros.
    """
    pose_obj = frame_data.get("pose", [])
    if not pose_obj:
        return None, None, None
    pose_lms = pose_obj[0].get("landmarks", [])
    pose_arr = get_landmark_array(pose_lms, POSE_LANDMARKS_COUNT)
    
    p_left = pose_arr[L_SHOULDER]
    p_right = pose_arr[R_SHOULDER]
    
    if np.isnan(p_left).any() or np.isnan(p_right).any():
        return None, None, None
        
    anchor = (p_left + p_right) / 2.0
    scale = np.linalg.norm(p_left - p_right)
    if scale < 1e-6: scale = 1.0
        
    norm_pose = (pose_arr - anchor) / scale
    
    norm_left = np.full((HAND_LANDMARKS_COUNT, 3), np.nan)
    norm_right = np.full((HAND_LANDMARKS_COUNT, 3), np.nan)
    
    for hand in frame_data.get("hands", []):
        lbl = hand.get("handedness")
        lms = hand.get("landmarks", [])
        arr = get_landmark_array(lms, HAND_LANDMARKS_COUNT)
        norm_arr = (arr - anchor) / scale
        
        if lbl == "Left":
            norm_left = norm_arr
        elif lbl == "Right":
            norm_right = norm_arr
            
    return norm_pose, norm_left, norm_right

def process_file(file_path):
    # Skip output files
    if file_path.endswith("_norm.json"):
        return
        
    base_name = os.path.basename(file_path).replace(".json", "")
    out_path = os.path.join(DATA_DIR, f"{base_name}_norm.json")
    
    print(f"Normalizando: {base_name} -> {out_path}")
    
    try:
        data = load_json(file_path)
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return

    raw_frames = data.get("frames", [])
    if not raw_frames:
        print(f"  -> Sem frames.")
        return
        
    # --- PASSO 1: SUAVIZAÇÃO SMA (Kills Jitter) ---
    raw_pose = [f.get("pose", [{}])[0].get("landmarks", []) for f in raw_frames]
    raw_left = [next((h.get("landmarks", []) for h in f.get("hands", []) if h.get("handedness") == "Left"), []) for f in raw_frames]
    raw_right = [next((h.get("landmarks", []) for h in f.get("hands", []) if h.get("handedness") == "Right"), []) for f in raw_frames]

    smooth_pose = numpy_sma_points(raw_pose, POSE_LANDMARKS_COUNT)
    smooth_left = numpy_sma_points(raw_left, HAND_LANDMARKS_COUNT)
    smooth_right = numpy_sma_points(raw_right, HAND_LANDMARKS_COUNT)

    # Reconstruir frames estruturados para normalização
    processed_frames = []
    for i in range(len(raw_frames)):
        processed_frames.append({
            "timestamp_ms": raw_frames[i].get("timestamp_ms", 0),
            "pose": [{"landmarks": smooth_pose[i]}],
            "hands": [
                {"handedness": "Left", "landmarks": smooth_left[i]},
                {"handedness": "Right", "landmarks": smooth_right[i]}
            ]
        })

    # --- PASSO 2: EXTRAÇÃO DE EVENTOS (VELOCITY PEAKS) ---
    all_normalized = []
    velocities = []
    prev_pa_l, prev_pa_r = None, None

    for i, frame in enumerate(processed_frames):
        p, l, r = normalize_spatial(frame)
        if p is None: continue

        pa_l = p[15][:2] if not np.isnan(p[15]).any() else np.zeros(2)
        pa_r = p[16][:2] if not np.isnan(p[16]).any() else np.zeros(2)
        delta = 0
        if prev_pa_l is not None:
            delta = np.linalg.norm(pa_l - prev_pa_l) + np.linalg.norm(pa_r - prev_pa_r)
        
        velocities.append(delta)
        prev_pa_l, prev_pa_r = pa_l, pa_r

        hip_y = (p[23][1] + p[24][1]) / 2.0
        in_rest = (pa_l[1] > hip_y - 0.1) and (pa_r[1] > hip_y - 0.1)

        all_normalized.append({
            "data": frame,
            "vel": delta,
            "in_rest": in_rest
        })

    # Selecionar Keyframes por Picos de Velocidade
    selected_keyframes = []
    if len(all_normalized) > 2:
        # Primeiro frame ativo
        for first_idx, item in enumerate(all_normalized):
            if not item["in_rest"]:
                selected_keyframes.append(item["data"])
                break
        
        # Picos de velocidade (Momentos de maior explosão/movimento)
        for j in range(1, len(velocities) - 1):
            if velocities[j] > velocities[j-1] and velocities[j] > velocities[j+1]:
                if velocities[j] > 0.005 and not all_normalized[j]["in_rest"]:
                    selected_keyframes.append(all_normalized[j]["data"])
        
        # Último frame ativo
        for last_idx in range(len(all_normalized)-1, -1, -1):
            if not all_normalized[last_idx]["in_rest"]:
                selected_keyframes.append(all_normalized[last_idx]["data"])
                break

    # Se falhou em detectar picos, ou sinal muito curto, pega amostragem simples
    if len(selected_keyframes) < 3:
        step = max(1, len(all_normalized) // 10)
        selected_keyframes = [all_normalized[i]["data"] for i in range(0, len(all_normalized), step)]

    # Salvar
    out_data = {
        "video_info": data.get("video_info", {}),
        "frames": selected_keyframes
    }
    
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(out_data, f, indent=None)
        
    print(f"  -> Gerado com {len(selected_keyframes)} keyframes.")

def main():
    print("=== Normalizador BATCH (Keyframes) ===")
    
    if not os.path.exists(DATA_DIR):
        print(f"Diretório '{DATA_DIR}' não encontrado.")
        return
        
    all_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    # Filtrar apenas os RAW (não normalizados) que são BASE
    raw_files = [f for f in all_files if not f.endswith("_norm.json") and "_base" in os.path.basename(f).lower()]
    
    if not raw_files:
        print("Nenhum arquivo RAW (*.json) encontrado em 'data/'.")
        return
        
    print(f"Arquivos RAW encontrados: {len(raw_files)}")
    
    for f in raw_files:
        process_file(f)

if __name__ == "__main__":
    main()
