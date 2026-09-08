import os
import base64
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Treinamento IA root
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # TCC root

TFLITE_PATH = os.path.join(BASE_DIR, "models", "modelo_gestos.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")
SEEDS_PATH = os.path.join(BASE_DIR, "data", "seeds", "seeds.json")
JS_MODEL_OUT = os.path.join(PROJECT_ROOT, "POC", "modelBase64.js")
JS_LABELS_OUT = os.path.join(PROJECT_ROOT, "POC", "labels.js")
JS_SEEDS_OUT = os.path.join(PROJECT_ROOT, "POC", "referenceSeeds.js")

def update_poc_files():
    # 1. Update Base64 Model
    if os.path.exists(TFLITE_PATH):
        print("Convertendo TFLite para Base64...")
        with open(TFLITE_PATH, "rb") as f:
            tflite_bytes = f.read()
        b64_string = base64.b64encode(tflite_bytes).decode('utf-8')

        with open(JS_MODEL_OUT, "w", encoding='utf-8') as f:
            f.write(f"export const modelBase64 = '{b64_string}';\n")

        print(f"Salvo -> {JS_MODEL_OUT}")
    else:
        print(f"Aviso: {TFLITE_PATH} não encontrado para converter para POC.")

    # 2. Update Labels
    if os.path.exists(LABELS_PATH):
        print("Carregando novas classes...")
        with open(LABELS_PATH, "r", encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]

        with open(JS_LABELS_OUT, "w", encoding='utf-8') as f:
            f.write(f"export const labels = {labels};\n")

        print(f"Salvo -> {JS_LABELS_OUT} (Total: {len(labels)} classes)")
    else:
        print(f"Aviso: {LABELS_PATH} não encontrado para exportar para POC.")

    # 3. Update Reference Seeds
    if os.path.exists(SEEDS_PATH):
        print("Exportando seeds 2D de referência para a POC...")
        with open(SEEDS_PATH, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        ref_seeds = {}
        for code, pts in seeds.items():
            x2d = [p['x'] for p in pts]
            y2d = [p['y'] for p in pts]
            min_x, max_x = min(x2d), max(x2d)
            min_y, max_y = min(y2d), max(y2d)
            span = max(max_x - min_x, max_y - min_y, 1e-6)
            norm_pts = [[round((p['x'] - min_x) / span, 4), round((p['y'] - min_y) / span, 4)] for p in pts]
            ref_seeds[code] = norm_pts
        with open(JS_SEEDS_OUT, "w", encoding="utf-8") as f:
            f.write(f"export const referenceSeeds = {json.dumps(ref_seeds)};\n")
        print(f"Salvo -> {JS_SEEDS_OUT} (Total: {len(ref_seeds)} seeds)")
    else:
        print(f"Aviso: {SEEDS_PATH} não encontrado para exportar para POC.")

if __name__ == "__main__":
    update_poc_files()
