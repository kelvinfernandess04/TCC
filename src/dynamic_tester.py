import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sys
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Treinamento IA")
H5_PATH = os.path.join(TRAIN_DIR, "modelo_gestos.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "labels.txt")
BASE_JSON = os.path.join(BASE_DIR, "src", "dynamic_signatures.json")

def load_ai_model():
    labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    model = tf.keras.models.load_model(H5_PATH) if os.path.exists(H5_PATH) else None
    return model, labels

def normalize_hand_for_ai(hand_landmarks):
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

def extract_from_video(video_path, model, labels, holistic):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        frame_data = {"frame": frame_idx, "shape_prediction": "NENHUM", "confidence": 0.0, "relative_trajectory_x": 0.0, "relative_trajectory_y": 0.0, "valid": False}
        
        body_mid_x, body_mid_y = None, None
        if results.pose_landmarks:
            l_shoulder = results.pose_landmarks.landmark[11]
            r_shoulder = results.pose_landmarks.landmark[12]
            body_mid_x = (l_shoulder.x + r_shoulder.x) / 2.0
            body_mid_y = (l_shoulder.y + r_shoulder.y) / 2.0
        
        hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
        if hand_landmarks:
            norm_coords, raw_pts = normalize_hand_for_ai(hand_landmarks)
            inp = np.array([norm_coords], dtype=np.float32)
            pred = model.predict(inp, verbose=0)[0]
            idx = np.argmax(pred)
            
            if idx < len(labels):
                frame_data["shape_prediction"] = labels[idx]
                frame_data["confidence"] = float(pred[idx])
            
            if body_mid_x is not None and body_mid_y is not None:
                frame_data["relative_trajectory_x"] = raw_pts[0][0] - body_mid_x
                frame_data["relative_trajectory_y"] = raw_pts[0][1] - body_mid_y
                frame_data["valid"] = True
                
        frame_sequence.append(frame_data)
        frame_idx += 1
    cap.release()
    return frame_sequence

def compare_sequences(base_seq, test_seq):
    # 1. Comparação de Formas (Shapes)
    base_shapes = [f["shape_prediction"] for f in base_seq if f.get("relative_trajectory_x") is not None and f["shape_prediction"] != "NENHUM"]
    test_shapes = [f["shape_prediction"] for f in test_seq if f.get("relative_trajectory_x") is not None and f["shape_prediction"] != "NENHUM"]
    
    # Contabiliza qual formato de mão dominou
    def get_top_shapes(shapes):
        counts = {}
        for s in shapes:
            counts[s] = counts.get(s, 0) + 1
        # Top 2 formas (já que um sinal costuma mudar de uma para outra)
        sorted_shapes = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [s[0] for s in sorted_shapes[:2]]
        
    base_top = get_top_shapes(base_shapes)
    test_top = get_top_shapes(test_shapes)
    
    shape_score = 0
    if len(base_top) > 0 and len(test_top) > 0:
        matches = set(base_top).intersection(set(test_top))
        shape_score = len(matches) / len(base_top) * 100.0
    
    # 2. Comparação de Trajetória (DTW)
    base_traj = np.array([[f["relative_trajectory_x"], f["relative_trajectory_y"]] for f in base_seq if f.get("relative_trajectory_x") is not None])
    test_traj = np.array([[f["relative_trajectory_x"], f["relative_trajectory_y"]] for f in test_seq if f.get("relative_trajectory_x") is not None])
    
    traj_score = 0.0
    if len(base_traj) > 0 and len(test_traj) > 0:
        distance, path = fastdtw(base_traj, test_traj, dist=euclidean)
        # Normaliza a distância pelo tamanho dos caminhos. Max dist ~ 1.0 (imagem normalizada), logo tolerância até 0.3
        norm_distance = distance / len(path)
        # Transforma num score de 0 a 100 (clamp max dist 0.4 -> 0 score)
        traj_score = max(0.0, 100.0 * (1.0 - (norm_distance / 0.4)))
        
    # Pontuação final (50% Shapes, 50% Trajetória)
    final_score = (shape_score * 0.5) + (traj_score * 0.5)
    
    return {
        "final_score": final_score,
        "shape_score": shape_score,
        "traj_score": traj_score,
        "base_top_shapes": base_top,
        "test_top_shapes": test_top
    }

def main(video_file, target_sign):
    if not os.path.exists(BASE_JSON):
        print("[ERRO] Arquivo dynamic_signatures.json não encontrado.")
        return

    with open(BASE_JSON, 'r') as f:
        signatures = json.load(f)
        
    if target_sign not in signatures:
        print(f"[ERRO] Assinatura base para '{target_sign}' não encontrada.")
        return
        
    base_seq = signatures[target_sign]["sequence"]
    
    model, labels = load_ai_model()
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)
    
    print(f"[TESTE] Extraindo dados do vídeo de teste: {video_file}...")
    test_seq = extract_from_video(video_file, model, labels, holistic)
    
    print(f"[TESTE] Processando comparações...")
    result = compare_sequences(base_seq, test_seq)
    
    print("="*40)
    print(" RELATÓRIO DE AVALIAÇÃO DO SINAL ")
    print("="*40)
    print(f"-> Sinal Alvo: {target_sign}")
    print(f"-> Formas de Mão Identificadas na Base: {result['base_top_shapes']}")
    print(f"-> Formas de Mão Identificadas no Teste: {result['test_top_shapes']}")
    print(f"-> Score de Reconhecimento de Forma (A/B/C..): {result['shape_score']:.1f}%")
    print(f"-> Score de Trajetória Corporal (DTW): {result['traj_score']:.1f}%")
    print("-" * 40)
    print(f" NOTA FINAL: {result['final_score']:.1f}%")
    
    if result['final_score'] >= 70.0:
        print(" VEREDITO: APROVADO! O sinal foi muito bem executado. ")
    else:
        print(" VEREDITO: REPROVADO. Melhore a execução e tente denovo. ")
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python dynamic_tester.py <caminho_video_mp4> <SINAL_ALVO>")
    else:
        main(sys.argv[1], sys.argv[2].upper())
