import os
import base64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Treinamento IA root
PROJECT_ROOT = os.path.dirname(BASE_DIR) # TCC root

TFLITE_PATH = os.path.join(BASE_DIR, "models", "modelo_gestos.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")
JS_MODEL_OUT = os.path.join(PROJECT_ROOT, "POC", "modelBase64.js")
JS_LABELS_OUT = os.path.join(PROJECT_ROOT, "POC", "labels.js")

def update_poc_files():
    # 1. Update Base64 Model
    print("Convertendo TFLite para Base64...")
    with open(TFLITE_PATH, "rb") as f:
        tflite_bytes = f.read()
    b64_string = base64.b64encode(tflite_bytes).decode('utf-8')
    
    with open(JS_MODEL_OUT, "w", encoding='utf-8') as f:
        f.write(f"export const modelBase64 = '{b64_string}';\n")
        
    print(f"Salvo -> {JS_MODEL_OUT}")
    
    # 2. Update Labels
    print("Carregando novas classes...")
    with open(LABELS_PATH, "r", encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    with open(JS_LABELS_OUT, "w", encoding='utf-8') as f:
        f.write(f"export const labels = {labels};\n")
        
    print(f"Salvo -> {JS_LABELS_OUT} (Total: {len(labels)} classes)")

if __name__ == "__main__":
    update_poc_files()
