import os
import sys

# Suprime logs C++ excessivos do TensorFlow e MediaPipe (Evita spam no console)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import json
import glob
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Treinamento IA root
DATASETS_DIR = os.path.join(BASE_DIR, "data", "datasets")
UNIFIED_JSON_DIR = os.path.join(BASE_DIR, "data", "unified_cache")
CACHE_FILE = os.path.join(BASE_DIR, "data", "extraction_cache.json")

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
    import os
    import sys
    
    # [HACK] Redireciona o stderr (nível do SO) para silenciar completamente os warnings C++ (absl) do MediaPipe
    try:
        sys.stderr.flush()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
    except: pass

    import numpy as np
    import cv2
    import mediapipe as mp
    
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
        return cache, pending_jobs, None

    # --- MULTIPROCESSING ---
    cores = max(1, multiprocessing.cpu_count() - 1) # Deixa 1 núcleo livre
    chunk_size = 500
    chunks = [pending_jobs[i:i + chunk_size] for i in range(0, len(pending_jobs), chunk_size)]
    
    print(f"[EXTRACTOR] Ligando Motores Paralelos (Multiprocessing) com {cores} Núcleos e {len(chunks)} Lotes...")
    
    start_time = time.time()
    total_processed = 0
    
    executor = ProcessPoolExecutor(max_workers=cores)
    futures = [executor.submit(process_chunk, chunk, ALLOWED_LABELS, DATASETS_DIR) for chunk in chunks]
    
    try:
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
            
    except KeyboardInterrupt:
        print("\n[EXTRACTOR] Interrompido pelo usuário (Ctrl+C). Cancelando e fechando workers...")
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        # Força saída imediata para não pendurar o terminal no Windows
        os._exit(1)
            
    print("\n[EXTRACTOR] Extração biológica concluída em múltiplos núcleos e cacheada!")
    executor.shutdown(wait=True)
    return cache, pending_jobs, start_time

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cache, pending_jobs, start_time = run_extraction()
    # ---------------------------
    # Relatório final de extração
    # ---------------------------
    import json, os, time
    report_path = os.path.join(BASE_DIR, "reports", "extraction_report.json")
    # Agrega estatísticas por dataset
    dataset_stats = {}
    total_success = total_nohand = total_ignored = total_error = 0
    for key, val in cache.items():
        status = val.get("status")
        if status == "success":
            total_success += 1
            # extrair caminho relativo ao dataset (primeiro diretório dentro de DATASETS_DIR)
            path_parts = key.split("_")
            if len(path_parts) >= 2:
                full_path = "_".join(path_parts[1:-2])
                rel = os.path.relpath(full_path, DATASETS_DIR)
                dataset_name = rel.split(os.sep)[0]
                dataset_stats.setdefault(dataset_name, {"found":0,"used":0})
                dataset_stats[dataset_name]["found"] += 1
                dataset_stats[dataset_name]["used"] += 1
        elif status == "no_hand":
            total_nohand += 1
        elif status == "ignored":
            total_ignored += 1
        elif status == "error":
            total_error += 1
    total_jobs = len(cache)
    elapsed = time.time() - start_time if start_time else 0
    mins, secs = divmod(int(elapsed), 60)
    report = {
        "total_images_discovered": len(pending_jobs),
        "total_jobs": total_jobs,
        "total_success": total_success,
        "total_no_hand": total_nohand,
        "total_ignored": total_ignored,
        "total_error": total_error,
        "elapsed_time": f"{mins}m{secs}s",
        "cpu_cores_used": max(1, multiprocessing.cpu_count() - 1),
        "chunk_size": 500,
        "dataset_breakdown": dataset_stats
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n[EXTRACTOR] Relatório de extração salvo em:", report_path)
