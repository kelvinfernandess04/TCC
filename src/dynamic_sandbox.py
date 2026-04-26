import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Treinamento IA")
H5_PATH = os.path.join(TRAIN_DIR, "modelo_gestos.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "labels.txt")

class DynamicSandbox:
    def __init__(self):
        print("="*50)
        print(" LIBRAS DYNAMIC SANDBOX (AI-POWERED) ")
        print("="*50)

        self.labels = []
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding='utf-8') as f:
                self.labels = [line.strip() for line in f if line.strip()]
        
        self.model = None
        if os.path.exists(H5_PATH):
            self.model = tf.keras.models.load_model(H5_PATH)
            print("[IA] Modelo Gestual Carregado.")
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mode = "IDLE"
        self.MAX_FRAMES = 60 # 2 Segundos a 30fps
        self.recorded_frames = []
        
        self.target_sign = ""
        self.typed_text = ""
        self.result_score = 0.0
        self.result_message = ""
        self.report = {}
        
        self.countdown_start_time = 0

    def normalize_hand(self, hand_landmarks):
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
        return normalized

    def process_hand(self, landmarks, frame, w, h, connections):
        self.mp_draw.draw_landmarks(frame, landmarks, connections)
        norm_coords = self.normalize_hand(landmarks)
        
        data = {"shape_prediction": "NENHUM", "confidence": 0.0}
        
        if self.model:
            inp = np.array([norm_coords], dtype=np.float32)
            pred = self.model.predict(inp, verbose=0)[0]
            idx = np.argmax(pred)
            if idx < len(self.labels):
                data["shape_prediction"] = self.labels[idx]
                data["confidence"] = float(pred[idx])
                
        return data

    def calculate_report(self):
        matches = 0
        confidences = []
        for f in self.recorded_frames:
            best_pred = "NENHUM"
            best_conf = 0.0
            
            if f["left"]["confidence"] > f["right"]["confidence"]:
                best_pred = f["left"]["shape_prediction"]
                best_conf = f["left"]["confidence"]
            else:
                best_pred = f["right"]["shape_prediction"]
                best_conf = f["right"]["confidence"]
                
            if best_pred == self.target_sign:
                matches += 1
                confidences.append(best_conf)
                
        match_rate = (matches / len(self.recorded_frames)) * 100.0 if self.recorded_frames else 0.0
        avg_conf = (sum(confidences) / len(confidences)) * 100.0 if confidences else 0.0
        
        final_score = (match_rate * 0.5) + (avg_conf * 0.5)
        
        return {
            "match_rate": match_rate,
            "avg_conf": avg_conf,
            "final_score": final_score
        }

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            frame_data = {
                "left": {"shape_prediction": "NENHUM", "confidence": 0.0},
                "right": {"shape_prediction": "NENHUM", "confidence": 0.0}
            }
            
            if results.left_hand_landmarks:
                frame_data["left"] = self.process_hand(
                    results.left_hand_landmarks, frame, w, h, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                frame_data["right"] = self.process_hand(
                    results.right_hand_landmarks, frame, w, h, self.mp_holistic.HAND_CONNECTIONS)
            
            if self.mode == "COUNTDOWN":
                elapsed = time.time() - self.countdown_start_time
                remaining = 3.0 - elapsed
                if remaining <= 0:
                    self.mode = "RECORD_TEST"
                    self.recorded_frames = []
                    self.result_score = 0.0
            elif self.mode == "RECORD_TEST":
                self.recorded_frames.append(frame_data)
                if len(self.recorded_frames) >= self.MAX_FRAMES:
                    report = self.calculate_report()
                    self.result_score = report["final_score"]
                    self.report = report
                    self.result_message = "APROVADO!" if self.result_score >= 70.0 else "REPROVADO!"
                    self.mode = "RESULT"
                        
            if self.mode == "COUNTDOWN":
                elapsed = time.time() - self.countdown_start_time
                remaining = int(3.0 - elapsed) + 1
                cv2.putText(frame, f"{remaining}", (w//2 - 40, h//2 + 40), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
            
            if self.mode != "RESULT" and self.mode != "TYPING":
                cv2.rectangle(frame, (0, 0), (350, 180), (20, 20, 20), -1)
                cv2.putText(frame, "DYNAMIC SANDBOX (AI POWERED)", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.putText(frame, "REAL-TIME AI:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.putText(frame, f"Esq: {frame_data['left']['shape_prediction']} ({frame_data['left']['confidence']*100:.0f}%)", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 1)
                cv2.putText(frame, f"Dir: {frame_data['right']['shape_prediction']} ({frame_data['right']['confidence']*100:.0f}%)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 1)
                
                if self.mode == "IDLE":
                    cv2.putText(frame, "[T] Iniciar Teste", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif self.mode == "RECORD_TEST":
                    cv2.putText(frame, f"GRAVANDO: {self.target_sign}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    prog = int((len(self.recorded_frames) / self.MAX_FRAMES) * 330)
                    cv2.rectangle(frame, (10, 160), (340, 175), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, 160), (10 + prog, 175), (0, 0, 255), -1)
            
            elif self.mode == "TYPING":
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.putText(frame, "QUAL SINAL VOCE VAI TESTAR?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"> {self.typed_text}_", (50, 180), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 3)
                cv2.putText(frame, "Digite a letra e aperte [ENTER]", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                
            elif self.mode == "RESULT":
                cv2.rectangle(frame, (0, 0), (550, 250), (30, 30, 30), -1)
                cv2.putText(frame, f"ALVO: {self.target_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                color = (0, 255, 0) if "APROVADO" in self.result_message else (0, 0, 255)
                cv2.putText(frame, self.result_message, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                
                if self.report:
                    y = 110
                    cv2.putText(frame, f"1. Precisao Temporal (Acertos nos 60 frames): {self.report['match_rate']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=25
                    cv2.putText(frame, f"2. Confianca Media da Inteligencia Artificial:  {self.report['avg_conf']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=40
                    
                    cv2.putText(frame, f"NOTA FINAL DE SIMILARIDADE: {self.report['final_score']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                cv2.putText(frame, "Pressione [Espaco] para fechar", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow('Sandbox Dinamico', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            
            if self.mode == "TYPING":
                if key == 13: # ENTER
                    if self.typed_text:
                        self.target_sign = self.typed_text.upper()
                        self.countdown_start_time = time.time()
                        self.mode = "COUNTDOWN"
                elif key == 8: # BACKSPACE
                    self.typed_text = self.typed_text[:-1]
                elif 32 <= key <= 126:
                    self.typed_text += chr(key).upper()
                    
            elif self.mode == "IDLE" and key == ord('t'):
                self.mode = "TYPING"
                self.typed_text = ""
                
            elif self.mode == "RESULT" and key == 32:
                self.mode = "IDLE"

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DynamicSandbox()
    app.run()
