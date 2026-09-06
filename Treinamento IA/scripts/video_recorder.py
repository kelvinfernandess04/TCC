import os
import cv2
import time
import datetime
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RECORDINGS_DIR = os.path.join(DATA_DIR, 'recordings')
import json
os.makedirs(RECORDINGS_DIR, exist_ok=True)

COLORS = {
    'bg_main': '#181825',
    'bg_card': '#1E1E2E',
    'bg_sidebar': '#11111B',
    'accent_blue': '#89B4FA',
    'accent_green': '#A6E3A1',
    'accent_yellow': '#F9E2AF',
    'accent_red': '#F38BA8',
    'accent_purple': '#CBA6F7',
    'text_main': '#CDD6F4',
    'text_muted': '#BAC2DE'
}

class HandVideoRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gravador de Vídeo Biomecânico - LIBRAS TCC")
        self.root.geometry("1100x720")
        self.root.configure(bg=COLORS['bg_main'])

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera & Recording state
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        self.recorded_file_path = None
        self.record_start_time = 0
        self.frames_recorded = 0
        self.recorded_landmarks = []
        self.current_frame = None

        self.build_ui()
        self.start_camera()

    def build_ui(self):
        # Left Panel (Controls and instructions)
        left_panel = tk.Frame(self.root, width=420, bg=COLORS['bg_sidebar'], padx=18, pady=18)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        lbl_title = tk.Label(
            left_panel, text="🎥 GRAVAÇÃO DE MOVIMENTOS",
            fg=COLORS['accent_blue'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 12, "bold")
        )
        lbl_title.pack(anchor=tk.W)

        lbl_sub = tk.Label(
            left_panel, text="Grave sua mão executando todos os limites biomecânicos para calibrar o modelo 3D.",
            fg=COLORS['text_muted'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 8), wraplength=380, justify=tk.LEFT
        )
        lbl_sub.pack(anchor=tk.W, pady=(2, 12))

        # Instructions Card
        card_guide = tk.LabelFrame(
            left_panel, text="📋 Roteiro de Movimentos para o Vídeo",
            fg=COLORS['accent_yellow'], bg=COLORS['bg_card'], padx=12, pady=10, font=("Segoe UI", 9, "bold")
        )
        card_guide.pack(fill=tk.X, pady=(0, 12))

        guide_steps = [
            "1. Mão espalmada aberta em leque (máximo)",
            "2. Fechar dedos em paralelo (sem leque)",
            "3. Flexionar dedos em Garra (Estágio 1)",
            "4. Flexionar dedos em Gancho (Estágio 2)",
            "5. Fechar mão em Punho / Soco (Estágio 3)",
            "6. Mover polegar: aberto ao lado, ponta dobrada",
            "7. Mover polegar em oposição cruzando a palma",
            "8. Fazer gestos de LIBRAS: A, B, C, I, V, W"
        ]

        for step in guide_steps:
            lbl = tk.Label(
                card_guide, text=step, fg=COLORS['text_main'], bg=COLORS['bg_card'],
                font=("Segoe UI", 8), anchor='w'
            )
            lbl.pack(fill=tk.X, pady=1)

        # Status & Recording Info
        self.card_status = tk.Frame(left_panel, bg=COLORS['bg_card'], padx=12, pady=12)
        self.card_status.pack(fill=tk.X, pady=(0, 12))

        self.lbl_rec_status = tk.Label(
            self.card_status, text="⚪ Câmera Pronta (Aguardando Gravação)",
            fg=COLORS['accent_green'], bg=COLORS['bg_card'], font=("Segoe UI", 9, "bold")
        )
        self.lbl_rec_status.pack(anchor=tk.W)

        self.lbl_rec_time = tk.Label(
            self.card_status, text="Duração: 00:00 | Quadros: 0",
            fg=COLORS['text_muted'], bg=COLORS['bg_card'], font=("Segoe UI", 8)
        )
        self.lbl_rec_time.pack(anchor=tk.W, pady=(3, 0))

        # Action Buttons
        self.btn_toggle_rec = tk.Button(
            left_panel, text="🔴 INICIAR GRAVAÇÃO DO VÍDEO",
            bg=COLORS['accent_red'], fg='#11111B', font=("Segoe UI", 11, "bold"),
            relief='flat', pady=10, cursor='hand2', command=self.toggle_recording
        )
        self.btn_toggle_rec.pack(fill=tk.X, pady=4)

        self.btn_analyze = tk.Button(
            left_panel, text="⚡ PROCESSAR VÍDEO & EXTRAIR LIMITES",
            bg=COLORS['accent_yellow'], fg='#11111B', font=("Segoe UI", 10, "bold"),
            relief='flat', pady=8, cursor='hand2', state=tk.DISABLED, command=self.analyze_recorded_video
        )
        self.btn_analyze.pack(fill=tk.X, pady=4)

        # Right Panel: Camera View
        right_panel = tk.Frame(self.root, bg=COLORS['bg_main'], padx=15, pady=15)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.cam_canvas = tk.Label(right_panel, bg='#0B0B10')
        self.cam_canvas.pack(fill=tk.BOTH, expand=True)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.update_video_loop()

    def update_video_loop(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                annotated_frame = frame.copy()
                black_frame = np.zeros_like(frame)

                if res.multi_hand_landmarks:
                    hand_lms = res.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(annotated_frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    self.mp_drawing.draw_landmarks(black_frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    if self.is_recording:
                        # Save normalized coordinates
                        frame_pts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms.landmark]
                        self.recorded_landmarks.append(frame_pts)
                else:
                    if self.is_recording:
                        self.recorded_landmarks.append(None)

                # Write black frame (MediaPipe only) to video if recording for privacy
                if self.is_recording and self.video_writer:
                    self.video_writer.write(black_frame)
                    self.frames_recorded += 1
                    elapsed = int(time.time() - self.record_start_time)
                    mins = elapsed // 60
                    secs = elapsed % 60
                    self.lbl_rec_time.configure(
                        text=f"Duração: {mins:02d}:{secs:02d} | Quadros: {self.frames_recorded}"
                    )

                    # Draw recording badge on screen
                    cv2.circle(annotated_frame, (35, 35), 12, (0, 0, 255), -1)
                    cv2.putText(
                        annotated_frame, f"REC {mins:02d}:{secs:02d}", (55, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                # Display preview in GUI
                canvas_w = self.cam_canvas.winfo_width() or 640
                canvas_h = self.cam_canvas.winfo_height() or 480
                
                scale = min(canvas_w / w, canvas_h / h)
                disp_w = max(1, int(w * scale))
                disp_h = max(1, int(h * scale))

                resized = cv2.resize(annotated_frame, (disp_w, disp_h))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                self.cam_canvas.img_tk = img_tk
                self.cam_canvas.configure(image=img_tk)

        self.root.after(20, self.update_video_loop)

    def toggle_recording(self):
        if not self.is_recording:
            # Start Recording
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recorded_file_path = os.path.join(RECORDINGS_DIR, f"calibracao_mao_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

            self.video_writer = cv2.VideoWriter(self.recorded_file_path, fourcc, fps, (w, h))
            self.is_recording = True
            self.record_start_time = time.time()
            self.frames_recorded = 0
            self.recorded_landmarks = []

            self.btn_toggle_rec.configure(
                text="⏹️ PARAR GRAVAÇÃO", bg=COLORS['accent_green']
            )
            self.lbl_rec_status.configure(
                text="🔴 GRAVANDO VÍDEO...", fg=COLORS['accent_red']
            )
            self.btn_analyze.configure(state=tk.DISABLED)
        else:
            # Stop Recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Save landmarks JSON
            if self.recorded_file_path and self.recorded_landmarks:
                json_path = self.recorded_file_path.replace('.mp4', '_landmarks.json')
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(self.recorded_landmarks, f)
                    print(f"[LOG] Saved {len(self.recorded_landmarks)} frames to {json_path}")
                except Exception as e:
                    print(f"[ERRO] Erro ao salvar landmarks JSON: {e}")

            self.btn_toggle_rec.configure(
                text="🔴 GRAVAR OUTRO VÍDEO", bg=COLORS['accent_red']
            )
            self.lbl_rec_status.configure(
                text=f"✅ Vídeo Salvo ({self.frames_recorded} quadros)", fg=COLORS['accent_green']
            )
            self.btn_analyze.configure(state=tk.NORMAL)
            messagebox.showinfo("Vídeo Gravado", f"Vídeo gravado com sucesso!\nSalvo em:\n{self.recorded_file_path}")

    def analyze_recorded_video(self):
        if not self.recorded_file_path or not os.path.exists(self.recorded_file_path):
            messagebox.showwarning("Aviso", "Nenhum vídeo disponível para análise.")
            return

        from video_calibrator import VideoRangeCalibrator
        analyzer = VideoRangeCalibrator()
        success, msg = analyzer.process_video(self.recorded_file_path)

        if success:
            messagebox.showinfo("Calibração Concluída", f"Limites biomecânicos extraídos com sucesso a partir do vídeo!\n\n{msg}")
        else:
            messagebox.showerror("Erro na Análise", f"Falha ao analisar vídeo:\n{msg}")

    def on_closing(self):
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = HandVideoRecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
