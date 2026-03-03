import cv2 
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import filedialog

# Lista para armazenar os dados
video_data = []

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# Definir conexões manualmente
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
    (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20),  # Mindinho
    (5, 9), (9, 13), (13, 17)  # Palma
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Rosto
    (0, 4), (4, 5), (5, 6), (6, 8),  # Rosto
    (9, 10),  # Boca
    (11, 12),  # Ombros
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Braço esquerdo
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Braço direito
    (11, 23), (12, 24), (23, 24),  # Tronco
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Perna esquerda
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)   # Perna direita
]

def draw_connections(image, landmarks, connections, color=(0, 255, 0), thickness=2):
    """Desenha as conexões entre landmarks"""
    height, width, _ = image.shape
    
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = (
                int(landmarks[start_idx].x * width),
                int(landmarks[start_idx].y * height)
            )
            end_point = (
                int(landmarks[end_idx].x * width),
                int(landmarks[end_idx].y * height)
            )
            cv2.line(image, start_point, end_point, color, thickness)

def draw_landmarks(image, landmarks, color=(255, 0, 0), radius=5):
    """Desenha os landmarks como círculos"""
    height, width, _ = image.shape
    
    for landmark in landmarks:
        point = (
            int(landmark.x * width),
            int(landmark.y * height)
        )
        cv2.circle(image, point, radius, color, -1)
        cv2.circle(image, point, radius + 2, (255, 255, 255), 2)

def draw_landmarks_on_image(rgb_image, hand_result, pose_result):
    """Desenha landmarks de mãos e pose na mesma imagem"""
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape
    
    # Desenhar POSE
    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            # Desenhar conexões da pose (verde)
            draw_connections(annotated_image, pose_landmarks, POSE_CONNECTIONS, 
                           color=(0, 255, 0), thickness=2)
            # Desenhar pontos da pose (azul)
            draw_landmarks(annotated_image, pose_landmarks, 
                         color=(255, 0, 0), radius=4)
    
    # Desenhar MÃOS
    if hand_result.hand_landmarks:
        for idx in range(len(hand_result.hand_landmarks)):
            hand_landmarks = hand_result.hand_landmarks[idx]
            handedness = hand_result.handedness[idx]
            
            # Cor baseada na mão
            hand_color = (0, 0, 255) if handedness[0].category_name == "Left" else (255, 255, 0)
            
            # Desenhar conexões da mão
            draw_connections(annotated_image, hand_landmarks, HAND_CONNECTIONS,
                           color=hand_color, thickness=2)
            # Desenhar pontos da mão
            draw_landmarks(annotated_image, hand_landmarks,
                         color=hand_color, radius=3)
            
            # Adicionar texto de handedness
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN
            
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    return annotated_image

def extract_combined_data(hand_result, pose_result, frame_number, timestamp_ms):
    """Extrai dados de mãos e pose para JSON"""
    frame_data = {
        "frame": frame_number,
        "timestamp_ms": timestamp_ms,
        "hands": [],
        "pose": []
    }
    
    # Extrair dados das MÃOS
    if hand_result.hand_landmarks:
        for idx in range(len(hand_result.hand_landmarks)):
            hand_landmarks = hand_result.hand_landmarks[idx]
            handedness = hand_result.handedness[idx]
            
            hand_data = {
                "handedness": handedness[0].category_name,
                "confidence": handedness[0].score,
                "landmarks": []
            }
            
            for landmark_idx, landmark in enumerate(hand_landmarks):
                landmark_data = {
                    "id": landmark_idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility if hasattr(landmark, 'visibility') else None,
                    "presence": landmark.presence if hasattr(landmark, 'presence') else None
                }
                hand_data["landmarks"].append(landmark_data)
            
            frame_data["hands"].append(hand_data)
    
    # Extrair dados da POSE
    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            pose_data = {
                "landmarks": []
            }
            
            for landmark_idx, landmark in enumerate(pose_landmarks):
                landmark_data = {
                    "id": landmark_idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility if hasattr(landmark, 'visibility') else None,
                    "presence": landmark.presence if hasattr(landmark, 'presence') else None
                }
                pose_data["landmarks"].append(landmark_data)
            
            frame_data["pose"].append(pose_data)
    
    return frame_data

# ABRIR SELETOR DE ARQUIVO
print("Selecione um arquivo de vídeo...")
root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Selecione um arquivo de vídeo",
    filetypes=[
        ("Arquivos de vídeo", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
        ("MP4", "*.mp4"),
        ("AVI", "*.avi"),
        ("Todos os arquivos", "*.*")
    ]
)

if not video_path:
    print("Nenhum arquivo selecionado. Encerrando...")
    exit()

print(f"Arquivo selecionado: {video_path}")

# CRIAR DETECTORES - Hand Landmarker
hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# CRIAR DETECTORES - Pose Landmarker
pose_base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Obter FPS real do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
frame_duration_ms = int(1000 / fps)

frame_timestamp_ms = 0
frame_number = 0

print(f"Processando vídeo... FPS: {fps}")
print("Pressione 'q' para parar.")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Fim do vídeo ou erro ao ler frame")
        break
    
    # Converter BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # DETECTAR MÃOS
    hand_result = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # DETECTAR POSE
    pose_result = pose_detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # Extrair dados para JSON
    frame_data = extract_combined_data(hand_result, pose_result, frame_number, frame_timestamp_ms)
    video_data.append(frame_data)
    
    # Desenhar as anotações combinadas
    annotated_frame = draw_landmarks_on_image(rgb_frame, hand_result, pose_result)
    
    # Converter de volta para BGR
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    # Exibir o frame
    cv2.imshow('Hand + Pose Tracking', annotated_frame_bgr)
    
    # Incrementar contadores
    frame_timestamp_ms += frame_duration_ms
    frame_number += 1
    
    # Mostrar progresso
    if frame_number % 30 == 0:
        print(f"Processados {frame_number} frames...")
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Salvar dados em JSON
output_filename = f"hand_pose_landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

output_data = {
    "video_info": {
        "source": video_path,
        "total_frames": frame_number,
        "fps": fps
    },
    "frames": video_data
}

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nDados exportados para: {output_filename}")
print(f"Total de frames processados: {frame_number}")
print(f"Detecções de mãos: {sum(1 for frame in video_data if frame['hands'])}")
print(f"Detecções de pose: {sum(1 for frame in video_data if frame['pose'])}")