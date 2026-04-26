import os
import cv2
import json
import glob
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    return os.path.basename(os.path.dirname(file_path)).strip().upper()

def process_chunk(chunk_jobs, allowed_labels, datasets_dir):
    import numpy as np
    import cv2
    import mediapipe as mp
    import os
    
    mp_holistic = mp.solutions.holistic
    local_cache = {}
    
    # Lazy load NPY para evitar gargalo de memoria entre processos
    x_npy_local = None
    y_npy_local = None
    needs_npy = any(job['type'] == 'npy' for job in chunk_jobs)
    if needs_npy:
        x_npy_path = os.path.join(datasets_dir, "27 Class Sign Language Dataset", "X.npy")
        y_npy_path = os.path.join(datasets_dir, "27 Class Sign Language Dataset", "Y.npy")
        x_npy_local = np.load(x_npy_path, mmap_mode='r')
        y_npy_local = np.load(y_npy_path, allow_pickle=True)
        
    with mp_holistic.Holistic(
        static_image_mode=True, model_complexity=1, smooth_landmarks=False,
        enable_segmentation=False, refine_face_landmarks=False
    ) as holistic:
        
        for job in chunk_jobs:
            cache_key = job['key']
            image = None
            label = None
            
            if job['type'] == 'photo':
                label = os.path.basename(os.path.dirname(job['path'])).strip().upper()
                if not label or label not in allowed_labels:
                    continue
                image = cv2.imread(job['path'])
                
            elif job['type'] == 'npy':
                raw_label = str(y_npy_local[job['idx']]).strip().upper()
                if not raw_label or raw_label == 'NULL' or raw_label not in allowed_labels:
                    local_cache[cache_key] = {"status": "ignored"}
                    continue
                label = raw_label
                raw_img = x_npy_local[job['idx']]
                if raw_img.dtype != np.uint8:
                    if raw_img.max() <= 1.0:
                        image = (raw_img * 255.0).astype(np.uint8)
                    else:
                        image = raw_img.astype(np.uint8)
                else:
                    image = raw_img
                    
            if image is None:
                local_cache[cache_key] = {"status": "error"}
                continue
                
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
            
            if hand_landmarks:
                pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                local_cache[cache_key] = {"status": "success", "label": label, "pts": pts}
            else:
                local_cache[cache_key] = {"status": "no_hand"}
                
    return local_cache

def run_extraction():
    os.makedirs(UNIFIED_JSON_DIR, exist_ok=True)
    cache = load_cache()
    
    print("[EXTRACTOR] Varrendo base de imagens e filtrando por Whitelist...")
    all_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        paths = glob.glob(os.path.join(DATASETS_DIR, '**', ext), recursive=True)
        for p in paths:
            if 'MNIST' in p or 'mnist' in p.lower(): continue
            label = get_class_from_path(p)
            if label in ALLOWED_LABELS:
                all_images.append(p)
            
    print(f"[EXTRACTOR] {len(all_images)} amostras primárias (Físicas) válidas pela Whitelist.")
    
    import numpy as np
    X_npy_path = os.path.join(DATASETS_DIR, "27 Class Sign Language Dataset", "X.npy")
    Y_npy_path = os.path.join(DATASETS_DIR, "27 Class Sign Language Dataset", "Y.npy")
    
    npy_total = 0
    if os.path.exists(X_npy_path) and os.path.exists(Y_npy_path):
        try:
            X_npy_data = np.load(X_npy_path, mmap_mode='r')
            npy_total = len(X_npy_data)
            print(f"[EXTRACTOR] Base Mapeada (Virtual NPY) localizada com {npy_total} amostras internas.")
        except Exception as e:
            print(f"[EXTRACTOR] Falha ao alocar NPY virtual em memória: {e}")
            
    pending_jobs = []
    
    for fp in all_images:
        stat = os.stat(fp)
        key = f"PHY_{fp}_{stat.st_mtime}_{stat.st_size}"
        if key not in cache:
            pending_jobs.append({'type': 'photo', 'path': fp, 'key': key})
            
    for idx in range(npy_total):
        key = f"NPY_27Class_{idx}"
        if key not in cache:
            pending_jobs.append({'type': 'npy', 'idx': idx, 'key': key})
            
    print(f"[EXTRACTOR] Mídia inédita a ser processada rigorosamente via IA: {len(pending_jobs)} amostras restam.")
    
    if not pending_jobs:
        print("[EXTRACTOR] Pulo rápido. Base em cache está 100% atualizada.")
        return 

    # --- MULTIPROCESSING ---
    cores = max(1, multiprocessing.cpu_count() - 1) # Deixa 1 núcleo livre
    chunk_size = 500
    chunks = [pending_jobs[i:i + chunk_size] for i in range(0, len(pending_jobs), chunk_size)]
    
    print(f"[EXTRACTOR] Ligando Motores Paralelos (Multiprocessing) com {cores} Núcleos e {len(chunks)} Lotes...")
    
    start_time = time.time()
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(process_chunk, chunk, ALLOWED_LABELS, DATASETS_DIR) for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                chunk_result = future.result()
                for k, v in chunk_result.items():
                    cache[k] = v
                total_processed += len(chunk_result)
            except Exception as e:
                print(f"\n[EXTRACTOR] Lote falhou catastróficamente: {e}")
                
            display_total = min(total_processed, len(pending_jobs))
            
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"\r[{mins:02d}:{secs:02d}] [EXTRACTOR] Processando Lotes em Paralelo... ({display_total}/{len(pending_jobs)})", end="", flush=True)
            
            save_cache(cache)
            
    print("\n[EXTRACTOR] Extração biológica concluída em múltiplos núcleos e cacheada!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_extraction()
