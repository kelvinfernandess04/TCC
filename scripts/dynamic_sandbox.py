import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Treinamento IA")
H5_PATH = os.path.join(TRAIN_DIR, "models", "modelo_gestos.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "models", "labels.txt")
CUSTOM_DATASET_ROOT = os.path.join(TRAIN_DIR, "data", "datasets", "dataset_custom")

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
        self.typing_intent = "TEST" # Pode ser "TEST" ou "RECORD"
        
        os.makedirs(CUSTOM_DATASET_ROOT, exist_ok=True)

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

    def calculate_body_reference(self, pose_landmarks):
        if not pose_landmarks: return None
        # MediaPipe Pose: 11 = left shoulder, 12 = right shoulder
        l_sh = pose_landmarks.landmark[11]
        r_sh = pose_landmarks.landmark[12]
        cx = (l_sh.x + r_sh.x) / 2.0
        cy = (l_sh.y + r_sh.y) / 2.0
        return {"x": cx, "y": cy}

    def compute_dtw(self, seq1, seq2):
        if not seq1 or not seq2: return 999.0
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(np.array(seq1[i - 1]) - np.array(seq2[j - 1]))
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
        return dtw_matrix[n, m] / max(n, m) # Normaliza pelo tamanho médio

    def process_hand(self, landmarks, frame, w, h, connections, body_ref):
        self.mp_draw.draw_landmarks(frame, landmarks, connections)
        norm_coords = self.normalize_hand(landmarks)
        
        # 1. Centro da Palma
        palm_idx = [0, 1, 5, 9, 13, 17]
        px = sum([landmarks.landmark[i].x for i in palm_idx]) / 6.0
        py = sum([landmarks.landmark[i].y for i in palm_idx]) / 6.0
        pz = sum([landmarks.landmark[i].z for i in palm_idx]) / 6.0
        palm_center = {"x": px, "y": py, "z": pz}
        
        # 2. Vetor Normal da Palma (Produto Vetorial base do indicador e mindinho em relação ao pulso)
        v1 = np.array([landmarks.landmark[5].x - landmarks.landmark[0].x,
                       landmarks.landmark[5].y - landmarks.landmark[0].y,
                       landmarks.landmark[5].z - landmarks.landmark[0].z])
        v2 = np.array([landmarks.landmark[17].x - landmarks.landmark[0].x,
                       landmarks.landmark[17].y - landmarks.landmark[0].y,
                       landmarks.landmark[17].z - landmarks.landmark[0].z])
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-6: normal = normal / norm_length
        palm_normal = {"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])}
        
        # 3. Referência Corporal
        rel_body = None
        if body_ref:
            rel_body = {"dx": px - body_ref["x"], "dy": py - body_ref["y"]}
            
        data = {
            "shape_prediction": "NENHUM", 
            "confidence": 0.0, 
            "raw_pts": [[lm.x, lm.y] for lm in landmarks.landmark],
            "palm_center": palm_center,
            "palm_normal": palm_normal,
            "rel_body": rel_body
        }
        
        if self.model:
            inp = np.array([norm_coords], dtype=np.float32)
            pred = self.model.predict(inp, verbose=0)[0]
            idx = np.argmax(pred)
            if idx < len(self.labels):
                data["shape_prediction"] = self.labels[idx]
                data["confidence"] = float(pred[idx])
                
        return data

    def calculate_static_report(self):
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
            "final_score": final_score,
            "is_dynamic": False
        }

    def load_template(self, label):
        import glob
        import json
        class_dir = os.path.join(CUSTOM_DATASET_ROOT, label)
        if not os.path.exists(class_dir): return None
        files = glob.glob(os.path.join(class_dir, "*.json"))
        if not files: return None
        files.sort(key=os.path.getmtime, reverse=True)
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return None

    def calculate_report(self):
        template = self.load_template(self.target_sign)
        if not template:
            return self.calculate_static_report()
            
        traj_test, traj_template = [], []
        norm_test, norm_template = [], []
        shapes_test, shapes_template = [], []
        
        for f in self.recorded_frames:
            best = f["left"] if f["left"]["confidence"] > f["right"]["confidence"] else f["right"]
            traj_test.append([best["rel_body"]["dx"], best["rel_body"]["dy"]] if best.get("rel_body") else [0.0, 0.0])
            if best.get("palm_normal"): norm_test.append(np.array([best["palm_normal"]["x"], best["palm_normal"]["y"], best["palm_normal"]["z"]]))
            shapes_test.append(best.get("shape_prediction", "NENHUM"))
            
        for f in template.get("frames", []):
            traj_template.append([f["rel_body"]["dx"], f["rel_body"]["dy"]] if f.get("rel_body") else [0.0, 0.0])
            if f.get("palm_normal"): norm_template.append(np.array([f["palm_normal"]["x"], f["palm_normal"]["y"], f["palm_normal"]["z"]]))
            shapes_template.append(f.get("shape_prediction", "NENHUM"))

        match_count = sum(1 for st in shapes_test if st in shapes_template and st != "NENHUM")
        shape_score = (match_count / max(len(shapes_test), 1)) * 100.0
        
        dtw_dist = self.compute_dtw(traj_test, traj_template)
        traj_score = max(0.0, min(100.0, 100.0 - (dtw_dist * 300.0))) 
        
        dot_products = []
        for i in range(min(len(norm_test), len(norm_template))):
            v1, v2 = norm_test[i], norm_template[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                dot_products.append(np.dot(v1, v2))
        
        orient_score = max(0.0, np.mean(dot_products)) * 100.0 if dot_products else 0.0
        
        final_score = (shape_score * 0.3) + (traj_score * 0.5) + (orient_score * 0.2)
        
        return {
            "match_rate": shape_score,
            "avg_conf": traj_score,
            "orient_score": orient_score,
            "final_score": final_score,
            "is_dynamic": True
        }

    def save_session_to_catalog(self):
        from datetime import datetime
        import json
        
        class_dir = os.path.join(CUSTOM_DATASET_ROOT, self.target_sign)
        os.makedirs(class_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captura_{timestamp}.json"
        save_path = os.path.join(class_dir, filename)
        
        export_data = {
            "metadata": {
                "label": self.target_sign,
                "timestamp": timestamp,
                "frame_count": len(self.recorded_frames)
            },
            "frames": []
        }
        
        for idx, f in enumerate(self.recorded_frames):
            # Prioriza mão com maior confiança, se não, apenas pega a primeira detectada ou vazia
            if f["left"]["raw_pts"] and f["right"]["raw_pts"]:
                best = f["left"] if f["left"]["confidence"] > f["right"]["confidence"] else f["right"]
            elif f["left"]["raw_pts"]:
                best = f["left"]
            elif f["right"]["raw_pts"]:
                best = f["right"]
            else:
                best = {"raw_pts": [], "shape_prediction": "NENHUM", "palm_center": None, "palm_normal": None, "rel_body": None}
            
            export_data["frames"].append({
                "id": idx,
                "landmarks": best["raw_pts"],
                "shape_prediction": best.get("shape_prediction", "NENHUM"),
                "palm_center": best.get("palm_center"),
                "palm_normal": best.get("palm_normal"),
                "rel_body": best.get("rel_body")
            })
            
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"[CATÁLOGO] Salvo com sucesso em: {self.target_sign}/{filename}")

    def run(self):
        self.video_source = 0
        cap = cv2.VideoCapture(self.video_source)
        
        while True:
            if not cap.isOpened():
                break
                
            success, frame = cap.read()
            if not success: 
                if self.video_source != 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            if self.video_source == 0:
                frame = cv2.flip(frame, 1)
                
            h, w, _ = frame.shape
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            body_ref = self.calculate_body_reference(results.pose_landmarks)
            
            frame_data = {
                "left": {"shape_prediction": "NENHUM", "confidence": 0.0, "raw_pts": []},
                "right": {"shape_prediction": "NENHUM", "confidence": 0.0, "raw_pts": []}
            }
            
            if results.left_hand_landmarks:
                frame_data["left"] = self.process_hand(
                    results.left_hand_landmarks, frame, w, h, self.mp_holistic.HAND_CONNECTIONS, body_ref)
            if results.right_hand_landmarks:
                frame_data["right"] = self.process_hand(
                    results.right_hand_landmarks, frame, w, h, self.mp_holistic.HAND_CONNECTIONS, body_ref)
            
            if self.mode == "COUNTDOWN":
                elapsed = time.time() - self.countdown_start_time
                remaining = 3.0 - elapsed
                if remaining <= 0:
                    self.mode = "RECORD_TEST" if self.typing_intent == "TEST" else "RECORD_NEW"
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
            elif self.mode == "RECORD_NEW":
                self.recorded_frames.append(frame_data)
                if len(self.recorded_frames) >= self.MAX_FRAMES:
                    self.save_session_to_catalog()
                    self.result_score = 100.0
                    self.report = None
                    self.result_message = "SINAL GRAVADO!"
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
                    cv2.putText(frame, "[T] Iniciar Teste", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "[G] Gravar Novo Sinal", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, "[I] Importar Midia / [C] Camera", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 200), 2)
                
                elif self.mode in ["RECORD_TEST", "RECORD_NEW"]:
                    action_text = "TESTANDO" if self.mode == "RECORD_TEST" else "GRAVANDO"
                    color = (0, 0, 255) if self.mode == "RECORD_TEST" else (0, 255, 255)
                    cv2.putText(frame, f"{action_text}: {self.target_sign}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    prog = int((len(self.recorded_frames) / self.MAX_FRAMES) * 330)
                    cv2.rectangle(frame, (10, 160), (340, 175), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, 160), (10 + prog, 175), color, -1)
            
            elif self.mode == "TYPING":
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                
                msg = "QUAL SINAL VOCE VAI TESTAR?" if self.typing_intent == "TEST" else "QUAL SINAL DESEJA GRAVAR?"
                cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"> {self.typed_text}_", (50, 180), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 3)
                cv2.putText(frame, "Digite a letra e aperte [ENTER]", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                
            elif self.mode == "RESULT":
                cv2.rectangle(frame, (0, 0), (550, 250), (30, 30, 30), -1)
                cv2.putText(frame, f"ALVO: {self.target_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                color = (0, 255, 0) if "APROVADO" in self.result_message else (0, 0, 255)
                cv2.putText(frame, self.result_message, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                
                if self.report:
                    y = 110
                    is_dyn = self.report.get("is_dynamic", False)
                    if is_dyn:
                        cv2.putText(frame, f"1. Forma Estatica Base: {self.report['match_rate']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=25
                        cv2.putText(frame, f"2. Similaridade Trajetoria (DTW): {self.report['avg_conf']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=25
                        cv2.putText(frame, f"3. Orientacao da Palma: {self.report['orient_score']:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1); y+=40
                    else:
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
                    
            elif self.mode == "IDLE":
                if key == ord('t'):
                    self.mode = "TYPING"
                    self.typing_intent = "TEST"
                    self.typed_text = ""
                elif key == ord('g'):
                    self.mode = "TYPING"
                    self.typing_intent = "RECORD"
                    self.typed_text = ""
                elif key == ord('i'):
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    path = filedialog.askopenfilename(title="Selecione Midia", filetypes=[("Midia", "*.mp4 *.avi *.jpg *.jpeg *.png")])
                    root.destroy()
                    if path:
                        self.video_source = path
                        cap.release()
                        cap = cv2.VideoCapture(self.video_source)
                elif key == ord('c'):
                    self.video_source = 0
                    cap.release()
                    cap = cv2.VideoCapture(self.video_source)
                
            elif self.mode == "RESULT" and key == 32:
                self.mode = "IDLE"

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DynamicSandbox()
    app.run()
