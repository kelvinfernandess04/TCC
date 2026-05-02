import os
# --- FIX: Resolver conflitos Protobuf entre MediaPipe e TensorFlow ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import sys
from datetime import datetime

# Importar TensorFlow APÓS configurar o ambiente
import tensorflow as tf

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Treinamento IA")
H5_PATH = os.path.join(TRAIN_DIR, "models", "modelo_gestos.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "models", "labels.txt")
CUSTOM_DATASET_ROOT = os.path.join(TRAIN_DIR, "data", "datasets", "dataset_custom")

class AILiveCaptureTool:
    def __init__(self):
        print("\n" + "="*50)
        print(" LIBRAS AI - FERRAMENTA DE TREINAMENTO REAL-TIME ")
        print("="*50)

        # 1. Carregar Labels do último Treino
        self.labels = []
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding='utf-8') as f:
                self.labels = [line.strip() for line in f if line.strip()]
            print(f"[IA] {len(self.labels)} classes carregadas do projeto.")
        else:
            print("[AVISO] labels.txt não encontrado. Predição ficará inativa.")
            
        # 2. Carregar Modelo (.h5) gerado pelo neural_engine
        self.model = None
        if os.path.exists(H5_PATH):
            try:
                self.model = tf.keras.models.load_model(H5_PATH)
                print("[IA] Modelo Neural (.h5) carregado com sucesso!")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo: {e}")
        else:
            print("[AVISO] Modelo H5 não encontrado em 'Treinamento IA/'.")
            
        # 3. Inicializar MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 4. Parâmetros de Gravação
        os.makedirs(CUSTOM_DATASET_ROOT, exist_ok=True)
        self.recording_mode = False
        self.current_session_frames = []
        self.recording_max_frames = 60 # ~2-3 segundos de amostra
        
        # 5. Estado do Predictive Engine
        self.current_match = "IDLE"
        self.current_confidence = 0.0

    def normalize_for_ai(self, hand_landmarks):
        """Replica a normalização do neural_engine (Bounding Box Min-Max)."""
        pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
        pts_arr = np.array(pts)
        
        min_x, max_x = np.min(pts_arr[:, 0]), np.max(pts_arr[:, 0])
        min_y, max_y = np.min(pts_arr[:, 1]), np.max(pts_arr[:, 1])
        
        # Manter proporção mas normalizar entre 0 e 1 no bounding box
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

    def save_session_to_catalog(self):
        if not self.current_session_frames: return
        
        print("\n" + "-"*30)
        print(f" GRAVAÇÃO: {len(self.current_session_frames)} frames capturados.")
        label = input("Digite a LETRA/CLASSE deste sinal (ex: A): ").strip().upper()
        
        if not label:
            print("[CANCELADO] Nenhum nome fornecido. Dados descartados.")
            self.current_session_frames = []
            return

        # Criar subpasta para a classe (Catalogação)
        class_dir = os.path.join(CUSTOM_DATASET_ROOT, label)
        os.makedirs(class_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captura_{timestamp}.json"
        save_path = os.path.join(class_dir, filename)
        
        export_data = {
            "metadata": {
                "label": label,
                "timestamp": timestamp,
                "frame_count": len(self.current_session_frames)
            },
            "frames": []
        }
        
        for idx, landmarks in enumerate(self.current_session_frames):
            export_data["frames"].append({
                "id": idx,
                "landmarks": landmarks # Formato [[x,y],...]
            })
            
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"[CATÁLOGO] Salvo com sucesso em: {label}/{filename}")
        print("[!] Rode o 'treinamento.py' para integrar estes novos dados.")
        
        self.current_session_frames = []

    def draw_hud(self, frame):
        h, w, _ = frame.shape
        # Painel Lateral Escuro
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título
        cv2.putText(frame, "LIBRAS LIVE TRAINER", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Predição da IA Ativa
        status_color = (0, 255, 0) if self.current_confidence > 0.7 else (0, 165, 255)
        if self.current_match == "IDLE": status_color = (150, 150, 150)
        
        cv2.putText(frame, "PREDIÇÃO IA (Real-time):", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"{self.current_match}", (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2)
        cv2.putText(frame, f"Confiança: {self.current_confidence*100:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        # Comandos
        y_cmd = h - 60
        cv2.putText(frame, "CONTROLES:", (10, y_cmd - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "[R] Gravar Nova Amostra", (10, y_cmd), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 1)
        cv2.putText(frame, "[ESC] Sair do Programa", (10, y_cmd + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 250), 1)
        
        # Barra de Gravação Progressiva
        if self.recording_mode:
            progress_w = int((len(self.current_session_frames) / self.recording_max_frames) * 300)
            cv2.rectangle(frame, (10, h - 140), (310, h - 120), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, h - 140), (10 + progress_w, h - 120), (0, 0, 255), -1)
            cv2.putText(frame, "GRAVANDO PONTOS...", (10, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            # 1. Flip do frame ANTES do processamento (Alinha visual com a física)
            display_img = cv2.flip(image, 1)
            
            # Processamento Landmarks Holistic
            res = self.holistic.process(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            
            # Tenta pegar qualquer mão detectada
            hand_lms = res.right_hand_landmarks or res.left_hand_landmarks
            
            if hand_lms:
                # Desenhar esqueleto (Pontos e Linhas)
                self.mp_draw.draw_landmarks(display_img, hand_lms, self.mp_holistic.HAND_CONNECTIONS)
                
                # Normalizar e extrair pontos
                norm_42, raw_pts = self.normalize_for_ai(hand_lms)
                    
                # 2. IA Prediz se tiver modelo carregado
                if self.model:
                    inp = np.array([norm_42], dtype=np.float32)
                    pred = self.model.predict(inp, verbose=0)[0]
                    idx = np.argmax(pred)
                    self.current_match = self.labels[idx] if idx < len(self.labels) else "???"
                    self.current_confidence = pred[idx]
                
                # 3. Se estiver gravando, salva os pontos crus
                if self.recording_mode:
                    self.current_session_frames.append(raw_pts)
                    if len(self.current_session_frames) >= self.recording_max_frames:
                        self.recording_mode = False
                        # Pequena pausa visual antes do input do terminal
                        cv2.imshow('Libras Trainer', display_img)
                        cv2.waitKey(100)
                        self.save_session_to_catalog()
            else:
                self.current_match = "IDLE"
                self.current_confidence = 0.0
                
            self.draw_hud(display_img)
            cv2.imshow('Libras Trainer', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break # ESC
            elif key == ord('r'):
                if not self.recording_mode:
                    print("\n[GRAVAÇÃO] Iniciada! Mova a mão levemente...")
                    self.recording_mode = True
                    self.current_session_frames = []

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AILiveCaptureTool()
    app.run()
