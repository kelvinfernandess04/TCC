
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
    out_path = os.path.join("data", f"{base_name}_norm.json")
    
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
        
    # --- Passo 1: Chunking e Averaging (Keyframe Generation) ---
    # Simplificação: Vamos pular a verificação frame-a-frame de normalização aqui
    # e fazer a normalização DENTRO do chunking, ou apenas salvar os keyframes normalizados?
    # O user pediu "normalizador... normalize/extraiam todos".
    # Vamos manter a lógica original: Chunk -> Average -> Normalize.
    # Mas a função normalize_spatial acima retorna arrays numpy.
    
    valid_frames = [f for f in raw_frames if f.get("pose")] # Filtro básico
    
    keyframes = []
    
    for frame in valid_frames:
        timestamp = frame.get("timestamp_ms", 0)
        p, l, r = normalize_spatial(frame)
        if p is None:
            continue
            
        pose_lms = []
        for idx in range(POSE_LANDMARKS_COUNT):
             pose_lms.append({
                 "id": idx,
                 "x": p[idx][0],
                 "y": p[idx][1],
                 "z": p[idx][2]
             })
             
        hands_data = []
        if not np.isnan(l).all():
            l_lms = []
            for idx in range(HAND_LANDMARKS_COUNT):
                l_lms.append({"id": idx, "x": l[idx][0], "y": l[idx][1], "z": l[idx][2]})
            hands_data.append({"handedness": "Left", "landmarks": l_lms})
            
        if not np.isnan(r).all():
            r_lms = []
            for idx in range(HAND_LANDMARKS_COUNT):
                r_lms.append({"id": idx, "x": r[idx][0], "y": r[idx][1], "z": r[idx][2]})
            hands_data.append({"handedness": "Right", "landmarks": r_lms})
            
        keyframes.append({
            "timestamp_ms": timestamp,
            "pose": [{"landmarks": pose_lms}],
            "hands": hands_data
        })
            
    # Salvar
    out_data = {
        "video_info": data.get("video_info", {}),
        "frames": keyframes
    }
    
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(out_data, f, indent=None)
        
    print(f"  -> Gerado com {len(keyframes)} keyframes.")

def main():
    print("=== Normalizador BATCH (Keyframes) ===")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Diretório 'data' não encontrado.")
        return
        
    all_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    # Filtrar apenas os RAW (não normalizados)
    # E vamos supor que queremos normalizar TODOS os raw, 
    # ou seja, aqueles que NÃO terminam com _norm.json
    raw_files = [f for f in all_files if not f.endswith("_norm.json")]
    
    if not raw_files:
        print("Nenhum arquivo RAW (*.json) encontrado em 'data/'.")
        return
        
    print(f"Arquivos RAW encontrados: {len(raw_files)}")
    
    for f in raw_files:
        process_file(f)

if __name__ == "__main__":
    main()
