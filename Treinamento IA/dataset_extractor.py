import os
import cv2
import json
import glob
import sys
import time

# MOCK TENSORFLOW to avoid protobuf clash on MediaPipe import
from unittest.mock import MagicMock
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.tools'] = MagicMock()
sys.modules['tensorflow.tools.docs'] = MagicMock()

import mediapipe as mp

# Paths
DATASETS_DIR = r"C:\DevTools\Repositories\Faculdade\TCC\Treinamento IA\datasets"
UNIFIED_JSON_DIR = r"C:\DevTools\Repositories\Faculdade\TCC\Treinamento IA\unified_cache"
CACHE_FILE = r"C:\DevTools\Repositories\Faculdade\TCC\Treinamento IA\extraction_cache.json"

mp_holistic = mp.solutions.holistic

ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'I', 'L', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y']

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {}

def save_cache(cache_db):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_db, f)

def get_class_from_path(file_path):
    # O nome do diretório pai direto determina a classe/letra
    return os.path.basename(os.path.dirname(file_path)).strip().upper()

def run_extraction():
    os.makedirs(UNIFIED_JSON_DIR, exist_ok=True)
    cache = load_cache()
    
    # 1. Image Discovery (Crawler) - Imagens Fisicas
    print("[EXTRACTOR] Varrendo base de imagens e filtrando por Whitelist...")
    all_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        paths = glob.glob(os.path.join(DATASETS_DIR, '**', ext), recursive=True)
        for p in paths:
            if 'MNIST' in p or 'mnist' in p.lower(): continue
            # Trava: Só adiciona se o nome do diretório pai estiver na Whitelist
            label = get_class_from_path(p)
            if label in ALLOWED_LABELS:
                all_images.append(p)
            
    print(f"[EXTRACTOR] {len(all_images)} amostras primárias (Físicas) válidas pela Whitelist.")
    
    # 2. Descoberta de NPY Virtual
    import numpy as np
    X_npy_path = os.path.join(DATASETS_DIR, "27 Class Sign Language Dataset", "X.npy")
    Y_npy_path = os.path.join(DATASETS_DIR, "27 Class Sign Language Dataset", "Y.npy")
    
    X_npy_data = None
    Y_npy_data = None
    npy_total = 0
    
    if os.path.exists(X_npy_path) and os.path.exists(Y_npy_path):
        try:
            X_npy_data = np.load(X_npy_path, mmap_mode='r')
            Y_npy_data = np.load(Y_npy_path, allow_pickle=True)
            npy_total = len(X_npy_data)
            print(f"[EXTRACTOR] Base Mapeada (Virtual NPY) localizada com {npy_total} amostras internas.")
        except Exception as e:
            print(f"[EXTRACTOR] Falha ao alocar NPY virtual em memória: {e}")
            
    # Check against cache
    pending_jobs = []
    
    # Job Type A: Físicos
    for fp in all_images:
        stat = os.stat(fp)
        key = f"PHY_{fp}_{stat.st_mtime}_{stat.st_size}"
        if key not in cache:
            pending_jobs.append({'type': 'photo', 'path': fp, 'key': key})
            
    # Job Type B: Virtual NPY
    for idx in range(npy_total):
        key = f"NPY_27Class_{idx}"
        if key not in cache:
            pending_jobs.append({'type': 'npy', 'idx': idx, 'key': key})
            
    print(f"[EXTRACTOR] Mídia inédita a ser processada rigorosamente via IA: {len(pending_jobs)} amostras restam.")
    
    if not pending_jobs:
        print("[EXTRACTOR] Pulo rápido. Base em cache está 100% atualizada.")
        return 

    # 3. Extract Using MediaPipe
    unified_results = {}
    start_time = time.time()
    with mp_holistic.Holistic(
        static_image_mode=True, model_complexity=1, smooth_landmarks=False,
        enable_segmentation=False, refine_face_landmarks=False
    ) as holistic:
        
        for idx, job in enumerate(pending_jobs):
            if idx % 50 == 0 or idx == len(pending_jobs) - 1:
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                print(f"\r[{mins:02d}:{secs:02d}] [EXTRACTOR] Passando pente fino... ({idx}/{len(pending_jobs)})", end="", flush=True)
                
            if idx > 0 and idx % 500 == 0:
                save_cache(cache) # Checkpoint de Segurança
                
            cache_key = job['key']
            
            if job['type'] == 'photo':
                label = get_class_from_path(job['path'])
                if not label: continue
                image = cv2.imread(job['path'])
                
            elif job['type'] == 'npy':
                raw_label = str(Y_npy_data[job['idx']]).strip().upper()
                if not raw_label or raw_label == 'NULL' or raw_label not in ALLOWED_LABELS: 
                    cache[cache_key] = {"status": "ignored"}
                    continue
                label = raw_label
                # Array NPY é 128x128x3 Float Float Bounds 0.0 - 1.0 (ou Similar)
                raw_img = X_npy_data[job['idx']]
                if raw_img.dtype != np.uint8:
                    if raw_img.max() <= 1.0:
                        image = (raw_img * 255.0).astype(np.uint8)
                    else:
                        image = raw_img.astype(np.uint8)
                else:
                    image = raw_img
                    
            if image is None: 
                cache[cache_key] = {"status": "error"}
                continue
                
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
            
            if hand_landmarks:
                pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                
                if label not in unified_results:
                    unified_results[label] = []
                    
                unified_results[label].append(pts)
                cache[cache_key] = {"status": "success", "label": label, "pts": pts}
            else:
                cache[cache_key] = {"status": "no_hand"}

    save_cache(cache)
    print("\n[EXTRACTOR] Extração biológica concluída e cacheada!")

if __name__ == "__main__":
    run_extraction()
