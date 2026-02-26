
import cv2
import mediapipe as mp
import json
import os
import glob
from datetime import datetime
import sys

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(landmarks):
    """Converte landmarks do MediaPipe para lista de dicionários."""
    if not landmarks:
        return []
    
    data = []
    for i, lm in enumerate(landmarks.landmark):
        data.append({
            "id": i,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility if hasattr(lm, "visibility") else 0.0,
            "presence": lm.presence if hasattr(lm, "presence") else 0.0
        })
    return data

def process_video(video_path):
    print(f"Processando: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_data = []
    frame_idx = 0
    
    # Configuração do Holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Processamento
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            
            # Montar estrutura JSON compatível com o visualizador
            frame_entry = {
                "frame": frame_idx,
                "timestamp_ms": cap.get(cv2.CAP_PROP_POS_MSEC),
                "hands": [],
                "pose": []
            }
            
            # POSE
            if results.pose_landmarks:
                frame_entry["pose"].append({
                    "landmarks": extract_landmarks(results.pose_landmarks)
                })
            
            # MÃO ESQUERDA
            if results.left_hand_landmarks:
                frame_entry["hands"].append({
                    "handedness": "Left",
                    "confidence": 1.0, 
                    "landmarks": extract_landmarks(results.left_hand_landmarks)
                })
                
            # MÃO DIREITA
            if results.right_hand_landmarks:
                frame_entry["hands"].append({
                    "handedness": "Right",
                    "confidence": 1.0,
                    "landmarks": extract_landmarks(results.right_hand_landmarks)
                })
            
            frames_data.append(frame_entry)
            
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames_est}...")
                
            frame_idx += 1
            
    cap.release()
    
    # Salvar JSON
    out_name = f"data/lm_{os.path.basename(video_path)}.json"
    
    output = {
        "video_info": {
            "source": video_path,
            "fps": fps,
            "total_frames": total_frames_est
        },
        "frames": frames_data
    }
    
    with open(out_name, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=None) # Sem indent para economizar espaço
        
    print(f"Salvo: {out_name}")

def main():
    print("=== Extrator de Keypoints BATCH (Holistic) ===")
    
    # Scan MP4/ directory
    mp4_dir = "MP4"
    if not os.path.exists(mp4_dir):
        print(f"Diretório '{mp4_dir}' não encontrado.")
        return
        
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(mp4_dir, ext)))
        
    if not video_files:
        print(f"Nenhum vídeo encontrado em '{mp4_dir}'.")
        return
        
    print(f"Vídeos encontrados: {len(video_files)}")
    
    for video_path in video_files:
        process_video(video_path)

if __name__ == "__main__":
    main()
