import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Treinamento IA")
H5_PATH = os.path.join(TRAIN_DIR, "modelo_gestos.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "labels.txt")
SANDBOX_JSON = os.path.join(BASE_DIR, "src", "sandbox_signatures.json")

class DynamicSandbox:
    def __init__(self):
        print("="*50)
        print(" LIBRAS DYNAMIC SANDBOX (DUAL-HAND) ")
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
        self.MAX_FRAMES = 90
        self.recorded_frames = []
        
        self.visual_path_left = []
        self.visual_path_right = []
        
        self.target_sign = ""
        self.result_score = 0.0
        self.result_message = ""
        self.report = {}
        
        self.countdown_start_time = 0
        self.pending_mode = ""

        self.signatures = self.load_signatures()
        
    def load_signatures(self):
        if os.path.exists(SANDBOX_JSON):
            with open(SANDBOX_JSON, 'r') as f:
                return json.load(f)
        return {}

    def save_signatures(self):
        with open(SANDBOX_JSON, 'w') as f:
            json.dump(self.signatures, f, indent=2)

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
        return normalized, pts

    def compare_single_hand(self, base_side, test_side):
        base_shapes = [f["shape_prediction"] for f in base_side if f.get("relative_trajectory_x") is not None and f["shape_prediction"] != "NENHUM"]
        
        if not base_shapes:
            return None # Não avalia essa mão pois ela não foi usada na base
            
        test_shapes = [f["shape_prediction"] for f in test_side if f.get("relative_trajectory_x") is not None and f["shape_prediction"] != "NENHUM"]
        
        def get_top_shapes(shapes):
            counts = {}
            for s in shapes: counts[s] = counts.get(s, 0) + 1
            sorted_shapes = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            return [s[0] for s in sorted_shapes[:2]]
            
        base_top = get_top_shapes(base_shapes)
        test_top = get_top_shapes(test_shapes)
        
        correct_confidences = []
        for f in test_side:
            if f["shape_prediction"] in base_top and f["shape_prediction"] != "NENHUM":
                correct_confidences.append(f.get("confidence", 0.0))
                
        valid_test_frames = [f for f in test_side if f["shape_prediction"] != "NENHUM"]
        shape_percentage = (len(correct_confidences) / len(valid_test_frames) * 100.0) if valid_test_frames else 0.0
        shape_confidence = (sum(correct_confidences) / len(correct_confidences) * 100.0) if correct_confidences else 0.0
        
        base_traj = np.array([[f["relative_trajectory_x"], f["relative_trajectory_y"]] for f in base_side if f.get("relative_trajectory_x") is not None])
        test_traj = np.array([[f["relative_trajectory_x"], f["relative_trajectory_y"]] for f in test_side if f.get("relative_trajectory_x") is not None])
        
        traj_score = 0.0
        if len(base_traj) > 0 and len(test_traj) > 0:
            distance, path = fastdtw(base_traj, test_traj, dist=euclidean)
            norm_distance = distance / len(path)
            traj_score = max(0.0, 100.0 * (1.0 - (norm_distance / 0.4)))
            
        final_score = (shape_percentage * 0.3) + (shape_confidence * 0.2) + (traj_score * 0.5)
        
        return {
            "score": final_score,
            "shape_perc": shape_percentage,
            "shape_conf": shape_confidence,
            "traj_score": traj_score,
            "test_top": test_top
        }

    def compare_sequences(self, base_seq, test_seq):
        base_left = [f["left"] for f in base_seq]
        base_right = [f["right"] for f in base_seq]
        
        test_left = [f["left"] for f in test_seq]
        test_right = [f["right"] for f in test_seq]
        
        res_left = self.compare_single_hand(base_left, test_left)
        res_right = self.compare_single_hand(base_right, test_right)
        
        valid_hands = []
        if res_left: valid_hands.append(res_left)
        if res_right: valid_hands.append(res_right)
        
        if not valid_hands:
            # Caso anomalo
            return {
                "final_score": 0.0, "traj_score": 0.0, "shape_percentage": 0.0, "shape_confidence": 0.0, "test_top": [], "hands_used": 0
            }
            
        avg_final = sum(r["score"] for r in valid_hands) / len(valid_hands)
        avg_traj = sum(r["traj_score"] for r in valid_hands) / len(valid_hands)
        avg_shape_p = sum(r["shape_perc"] for r in valid_hands) / len(valid_hands)
        avg_shape_c = sum(r["shape_conf"] for r in valid_hands) / len(valid_hands)
        all_tops = []
        for r in valid_hands: all_tops.extend(r["test_top"])
        
        return {
            "final_score": avg_final,
            "traj_score": avg_traj,
            "shape_percentage": avg_shape_p,
            "shape_confidence": avg_shape_c,
            "test_top": list(set(all_tops)),
            "hands_used": len(valid_hands)
        }

    def process_hand(self, landmarks, frame, w, h, body_mid_x, body_mid_y, connections):
        self.mp_draw.draw_landmarks(frame, landmarks, connections)
        norm_coords, raw_pts = self.normalize_hand(landmarks)
        
        data = {"shape_prediction": "NENHUM", "confidence": 0.0, "relative_trajectory_x": None, "relative_trajectory_y": None}
        wrist_px = (int(raw_pts[0][0] * w), int(raw_pts[0][1] * h))
        
        if self.model:
            inp = np.array([norm_coords], dtype=np.float32)
            pred = self.model.predict(inp, verbose=0)[0]
            idx = np.argmax(pred)
            if idx < len(self.labels):
                data["shape_prediction"] = self.labels[idx]
                data["confidence"] = float(pred[idx])
                
        if body_mid_x is not None and body_mid_y is not None:
            data["relative_trajectory_x"] = raw_pts[0][0] - body_mid_x
            data["relative_trajectory_y"] = raw_pts[0][1] - body_mid_y
            
        return data, wrist_px

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
                "left": {"shape_prediction": "NENHUM", "confidence": 0.0, "relative_trajectory_x": None, "relative_trajectory_y": None},
                "right": {"shape_prediction": "NENHUM", "confidence": 0.0, "relative_trajectory_x": None, "relative_trajectory_y": None}
            }
            wrist_left_px = None
            wrist_right_px = None
            
            body_mid_x, body_mid_y = None, None
            if results.pose_landmarks:
                l_shoulder = results.pose_landmarks.landmark[11]
                r_shoulder = results.pose_landmarks.landmark[12]
                body_mid_x = (l_shoulder.x + r_shoulder.x) / 2.0
                body_mid_y = (l_shoulder.y + r_shoulder.y) / 2.0
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                
            if results.left_hand_landmarks:
                frame_data["left"], wrist_left_px = self.process_hand(
                    results.left_hand_landmarks, frame, w, h, body_mid_x, body_mid_y, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                frame_data["right"], wrist_right_px = self.process_hand(
                    results.right_hand_landmarks, frame, w, h, body_mid_x, body_mid_y, self.mp_holistic.HAND_CONNECTIONS)
            
            # --- LÓGICA DE ESTADOS ---
            if self.mode == "COUNTDOWN":
                elapsed = time.time() - self.countdown_start_time
                remaining = 3.0 - elapsed
                if remaining <= 0:
                    self.mode = self.pending_mode
                    self.recorded_frames = []
                    self.visual_path_left = []
                    self.visual_path_right = []
                    self.result_score = 0.0
            elif self.mode in ["RECORD_BASE", "RECORD_TEST"]:
                self.recorded_frames.append(frame_data)
                
                if wrist_left_px: self.visual_path_left.append(wrist_left_px)
                if wrist_right_px: self.visual_path_right.append(wrist_right_px)
                    
                if len(self.recorded_frames) >= self.MAX_FRAMES:
                    if self.mode == "RECORD_BASE":
                        self.signatures[self.target_sign] = {"total_frames": self.MAX_FRAMES, "sequence": self.recorded_frames}
                        self.save_signatures()
                        self.result_message = f"BASE '{self.target_sign}' SALVA!"
                        self.mode = "RESULT"
                    elif self.mode == "RECORD_TEST":
                        base_seq = self.signatures.get(self.target_sign, {}).get("sequence", [])
                        if base_seq:
                            report = self.compare_sequences(base_seq, self.recorded_frames)
                            self.result_score = report["final_score"]
                            self.report = report
                            self.result_message = f"APROVADO!" if self.result_score >= 70.0 else "REPROVADO!"
                        else:
                            self.result_message = "BASE NÃO ENCONTRADA!"
                        self.mode = "RESULT"
                        
            # --- DESENHOS ---
            if len(self.visual_path_left) > 1:
                cv2.polylines(frame, [np.array(self.visual_path_left)], False, (255, 0, 0), 4) # Azul pra Esquerda
            if len(self.visual_path_right) > 1:
                cv2.polylines(frame, [np.array(self.visual_path_right)], False, (0, 0, 255), 4) # Vermelho pra Direita
                
            if self.mode == "COUNTDOWN":
                elapsed = time.time() - self.countdown_start_time
                remaining = int(3.0 - elapsed) + 1
                cv2.putText(frame, f"{remaining}", (w//2 - 40, h//2 + 40), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
            
            if self.mode != "RESULT":
                cv2.rectangle(frame, (0, 0), (350, 150), (20, 20, 20), -1)
                cv2.putText(frame, "DYNAMIC SANDBOX", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                
                if self.mode == "IDLE":
                    cv2.putText(frame, "[B] Gravar Nova Base", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(frame, "[T] Testar Sinal", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                    cv2.putText(frame, f"Bases: {len(self.signatures)} cadastradas", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                elif self.mode in ["RECORD_BASE", "RECORD_TEST"]:
                    cv2.putText(frame, f"GRAVANDO: {self.target_sign}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    prog = int((len(self.recorded_frames) / self.MAX_FRAMES) * 330)
                    cv2.rectangle(frame, (10, 100), (340, 120), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, 100), (10 + prog, 120), (0, 0, 255), -1)
                
            elif self.mode == "RESULT":
                cv2.rectangle(frame, (0, 0), (520, 290), (30, 30, 30), -1)
                cv2.putText(frame, f"ALVO: {self.target_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                color = (0, 255, 0) if "APROVADO" in self.result_message else (0, 0, 255)
                if "BASE SALVA" in self.result_message: color = (255, 255, 0)
                
                cv2.putText(frame, self.result_message, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                
                if "BASE SALVA" not in self.result_message and self.report:
                    y = 110
                    hands_text = f"Sim (Duas Maos)" if self.report.get('hands_used') == 2 else f"Apenas Uma"
                    cv2.putText(frame, f"> Maos detectadas na validacao: {hands_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1); y+=25
                    cv2.putText(frame, f"1. Similitude da Trajetoria (Corpo): {self.report['traj_score']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=25
                    cv2.putText(frame, f"2. Assertividade da Forma (Tempo):  {self.report['shape_percentage']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=25
                    cv2.putText(frame, f"3. Certeza Interna do Modelo IA:    {self.report['shape_confidence']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=25
                    cv2.putText(frame, f"Identificados: [{', '.join(self.report['test_top'])}]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1); y+=35
                    
                    cv2.putText(frame, f"NOTA FINAL: {self.report['final_score']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                cv2.putText(frame, "Pressione [Espaco] para fechar", (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow('Sandbox Dinamico', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('b') and self.mode == "IDLE":
                nome = input("Digite o nome do Sinal para gravar a BASE: ").strip().upper()
                if nome:
                    self.target_sign = nome
                    self.pending_mode = "RECORD_BASE"
                    self.countdown_start_time = time.time()
                    self.mode = "COUNTDOWN"
                    print(f"GRAVANDO BASE EM 3, 2, 1... PREPARE-SE!")
            elif key == ord('t') and self.mode == "IDLE":
                if not self.signatures:
                    print("Nenhuma base cadastrada. Aperte B primeiro.")
                else:
                    print("Bases cadastradas:", ", ".join(self.signatures.keys()))
                    nome = input("Digite qual Sinal você quer TESTAR: ").strip().upper()
                    if nome in self.signatures:
                        self.target_sign = nome
                        self.pending_mode = "RECORD_TEST"
                        self.countdown_start_time = time.time()
                        self.mode = "COUNTDOWN"
                        print(f"TESTANDO SINAL EM 3, 2, 1... PREPARE-SE!")
                    else:
                        print("Sinal não encontrado.")
            elif key == 32 and self.mode == "RESULT":
                self.mode = "IDLE"
                self.visual_path_left = []
                self.visual_path_right = []

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DynamicSandbox()
    app.run()
