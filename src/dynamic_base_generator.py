import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from glob import glob

# Evitar conflitos do protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Configurações de Caminho
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP4_DIR = os.path.join(BASE_DIR, "MP4")
TRAIN_DIR = os.path.join(BASE_DIR, "Treinamento IA")
H5_PATH = os.path.join(TRAIN_DIR, "modelo_gestos.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "labels.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "src", "dynamic_signatures.json")

def load_ai_model():
    labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    
    model = None
    if os.path.exists(H5_PATH):
        print("[INFO] Carregando modelo H5 neural...")
        model = tf.keras.models.load_model(H5_PATH)
    
    return model, labels

def normalize_hand_for_ai(hand_landmarks):
    """Normaliza pontos min-max para o modelo neural prever."""
    pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
    pts_arr = np.array(pts)
    
    min_x, max_x = np.min(pts_arr[:, 0]), np.max(pts_arr[:, 0])
    min_y, max_y = np.min(pts_arr[:, 1]), np.max(pts_arr[:, 1])
    
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    size = max(width, height)
    
    normalized = []
    for x, y in pts:
        nx = (x - min_x) / size
        ny = (y - min_y) / size
        normalized.append(nx)
        normalized.append(ny)
        
    return normalized, pts

def extract_signatures():
    model, labels = load_ai_model()
    if not model or not labels:
        print("[ERRO] Modelo ou labels indisponível. Treinamento IA não foi finalizado corretamente.")
        return

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Buscar vídeos base
    videos = glob(os.path.join(MP4_DIR, "*_base_boa.mp4"))
    if not videos:
        print("[AVISO] Nenhum vídeo base de boa qualidade encontrado na pasta MP4.")
        return

    signatures = {}

    for video_path in videos:
        nome_sinal = os.path.basename(video_path).split('_base_boa')[0].upper()
        print(f"\n[EXTRAÇÃO] Processando sinal base: {nome_sinal}")
        
        cap = cv2.VideoCapture(video_path)
        frame_sequence = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Processar com MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            frame_data = {
                "frame": frame_idx,
                "shape_prediction": "NENHUM",
                "confidence": 0.0,
                "relative_trajectory_x": None,
                "relative_trajectory_y": None
            }
            
            # 1. Ponto Médio do Corpo (Meio entre os ombros 11 e 12)
            body_mid_x = None
            body_mid_y = None
            if results.pose_landmarks:
                l_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                body_mid_x = (l_shoulder.x + r_shoulder.x) / 2.0
                body_mid_y = (l_shoulder.y + r_shoulder.y) / 2.0
            
            # 2. Informação da Mão
            hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
            if hand_landmarks:
                # Prever Forma da Mão
                norm_coords, raw_pts = normalize_hand_for_ai(hand_landmarks)
                inp = np.array([norm_coords], dtype=np.float32)
                pred = model.predict(inp, verbose=0)[0]
                idx = np.argmax(pred)
                
                if idx < len(labels):
                    frame_data["shape_prediction"] = labels[idx]
                    frame_data["confidence"] = float(pred[idx])
                
                # Calcular Relação Pulso -> Corpo
                # Pulso é o landmark 0
                wrist_x = raw_pts[0][0]
                wrist_y = raw_pts[0][1]
                
                if body_mid_x is not None and body_mid_y is not None:
                    # Trajetória em relação ao corpo. 
                    # Negativo = acima/à esquerda do centro, Positivo = abaixo/à direita
                    frame_data["relative_trajectory_x"] = wrist_x - body_mid_x
                    frame_data["relative_trajectory_y"] = wrist_y - body_mid_y
            
            frame_sequence.append(frame_data)
            frame_idx += 1
            
        cap.release()
        
        # Guardar na estrutura global assinada
        signatures[nome_sinal] = {
            "total_frames": frame_idx,
            "sequence": frame_sequence
        }
        print(f"-> {frame_idx} frames capturados com trajetórias e formativas da mão.")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(signatures, f, indent=2)
        
    print(f"\n[SUCESSO] Base de validação dinâmica salva em: {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_signatures()
