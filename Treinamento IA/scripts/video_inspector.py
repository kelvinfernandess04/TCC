import os
import json
import math
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RECORDINGS_DIR = os.path.join(DATA_DIR, 'recordings')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'calibration_settings.json')

COLORS = {
    'bg_main': '#181825',
    'bg_card': '#1E1E2E',
    'bg_sidebar': '#11111B',
    'bg_canvas': '#0F0F17',
    'accent_blue': '#89B4FA',
    'accent_green': '#A6E3A1',
    'accent_yellow': '#F9E2AF',
    'accent_red': '#F38BA8',
    'accent_purple': '#CBA6F7',
    'text_main': '#CDD6F4',
    'text_muted': '#BAC2DE'
}

FINGER_COLORS = {
    'Thumb':  '#FF5722',
    'Index':  '#FFEB3B',
    'Middle': '#4CAF50',
    'Ring':   '#00BCD4',
    'Pinky':  '#9C27B0'
}

def vec_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def joint_flexion(p0, p1, p2):
    return 180.0 - vec_angle(p0 - p1, p2 - p1)

def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

class VideoInspectorApp:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("Inspetor de Vídeo e Validador de Keyframes - LIBRAS TCC")
        self.root.geometry("1380x860")
        self.root.configure(bg=COLORS['bg_main'])

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False

        self.current_pts_3d = None
        self.view_rot_x = 0.0
        self.view_rot_y = 0.0
        self.drag_start = None
        self.video_landmarks = []

        self.keyframe_labels = {
            'stage_0_spread': '1. Mão Espalmada Aberta (Leque)',
            'stage_0_closed': '2. Mão Espalmada Dedos Juntos',
            'stage_1':         '3. Mão em Garra Leve (Estágio 1)',
            'stage_2':         '4. Mão em Gancho / Hook (Estágio 2)',
            'stage_3':         '5. Punho Fechado / Soco (Estágio 3)',
            'thumb_opposition':'6. Polegar em Oposição (F=1)',
            'thumb_ip_flexed': '7. Polegar Ponta Dobrada (P=1)'
        }

        self.calib_keyframes = {}
        self.load_current_calibration()
        self.build_ui()

        if not self.video_path:
            # Pick latest video in recordings directory
            recordings = sorted([
                os.path.join(RECORDINGS_DIR, f) for f in os.listdir(RECORDINGS_DIR) if f.endswith('.mp4')
            ])
            if recordings:
                self.load_video(recordings[-1])
        else:
            self.load_video(self.video_path)

    def load_current_calibration(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.calib_keyframes = data.get("keyframe_indices", {})
            except Exception as e:
                print(f"[AVISO] Erro ao ler calibração: {e}")

    def build_ui(self):
        # Top Bar
        top_bar = tk.Frame(self.root, bg=COLORS['bg_card'], padx=15, pady=10)
        top_bar.pack(fill=tk.X)

        lbl_t = tk.Label(top_bar, text="INSPETOR DE VÍDEO & EXTRAÇÃO DE KEYFRAMES", fg=COLORS['accent_blue'], bg=COLORS['bg_card'], font=("Segoe UI", 12, "bold"))
        lbl_t.pack(side=tk.LEFT)

        btn_open = tk.Button(
            top_bar, text="📂 Abrir Outro Vídeo", bg=COLORS['accent_purple'], fg='#11111B',
            font=("Segoe UI", 9, "bold"), relief='flat', padx=10, pady=4, cursor='hand2',
            command=self.choose_video_file
        )
        btn_open.pack(side=tk.RIGHT, padx=5)

        btn_save_all = tk.Button(
            top_bar, text="💾 Salvar Calibração Oficial", bg=COLORS['accent_green'], fg='#11111B',
            font=("Segoe UI", 9, "bold"), relief='flat', padx=10, pady=4, cursor='hand2',
            command=self.save_official_calibration
        )
        btn_save_all.pack(side=tk.RIGHT, padx=5)

        # Main Split
        main_split = tk.Frame(self.root, bg=COLORS['bg_main'])
        main_split.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Video Player & Timeline Controls
        self.left_box = tk.Frame(main_split, bg=COLORS['bg_sidebar'], width=580, padx=10, pady=10)
        self.left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_box.pack_propagate(False)

        self.lbl_video_preview = tk.Label(self.left_box, bg='#000000')
        self.lbl_video_preview.pack(fill=tk.BOTH, expand=True)

        # Timeline Slider & Controls
        ctrl_bar = tk.Frame(self.left_box, bg=COLORS['bg_sidebar'], pady=8)
        ctrl_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_play_pause = tk.Button(
            ctrl_bar, text="▶ Play", bg=COLORS['accent_green'], fg='#11111B',
            font=("Segoe UI", 9, "bold"), relief='flat', width=8, cursor='hand2', command=self.toggle_play
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=3)

        btn_prev = tk.Button(
            ctrl_bar, text="◀ -1", bg=COLORS['bg_card'], fg=COLORS['text_main'],
            font=("Segoe UI", 8, "bold"), relief='flat', padx=6, cursor='hand2', command=lambda: self.step_frame(-1)
        )
        btn_prev.pack(side=tk.LEFT, padx=2)

        btn_next = tk.Button(
            ctrl_bar, text="+1 ▶", bg=COLORS['bg_card'], fg=COLORS['text_main'],
            font=("Segoe UI", 8, "bold"), relief='flat', padx=6, cursor='hand2', command=lambda: self.step_frame(1)
        )
        btn_next.pack(side=tk.LEFT, padx=2)

        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(ctrl_bar, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.slider_var, command=self.on_slider_moved)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.lbl_frame_info = tk.Label(ctrl_bar, text="0 / 0", fg=COLORS['text_muted'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 8))
        self.lbl_frame_info.pack(side=tk.RIGHT)

        # Center/Right: Keyframe Assignment & Live Telemetry
        right_box = tk.Frame(main_split, bg=COLORS['bg_sidebar'], width=480, padx=12, pady=10)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_box.pack_propagate(False)

        # 3D Skeleton Canvas Header with Reset Button
        f_3d_hdr = tk.Frame(right_box, bg=COLORS['bg_sidebar'])
        f_3d_hdr.pack(fill=tk.X, pady=(0, 2))

        lbl_3d_t = tk.Label(f_3d_hdr, text="Visualização 3D do Quadro Atual", fg=COLORS['accent_yellow'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 10, "bold"))
        lbl_3d_t.pack(side=tk.LEFT)

        btn_reset_3d = tk.Button(
            f_3d_hdr, text="🔄 Visão 1:1", bg=COLORS['bg_card'], fg=COLORS['accent_blue'],
            font=("Segoe UI", 8, "bold"), relief='flat', padx=6, pady=1, cursor='hand2',
            command=self.reset_3d_view
        )
        btn_reset_3d.pack(side=tk.RIGHT)

        self.canvas_3d = tk.Canvas(right_box, bg=COLORS['bg_canvas'], height=260, highlightthickness=0)
        self.canvas_3d.pack(fill=tk.X, pady=(4, 8))
        self.canvas_3d.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas_3d.bind("<B1-Motion>", self.on_drag_move)

        # Telemetry Box
        self.lbl_telemetry = tk.Label(
            right_box, text="Telemetria do Quadro:", fg=COLORS['accent_blue'], bg=COLORS['bg_card'],
            font=("Consolas", 8), justify=tk.LEFT, padx=8, pady=6
        )
        self.lbl_telemetry.pack(fill=tk.X, pady=(0, 8))

        # Keyframe Assign Panel
        card_kf = tk.LabelFrame(right_box, text="Atribuir Quadro Atual como Posição Base", fg=COLORS['accent_green'], bg=COLORS['bg_card'], padx=8, pady=6, font=("Segoe UI", 9, "bold"))
        card_kf.pack(fill=tk.BOTH, expand=True)

        for key, title in self.keyframe_labels.items():
            f_kf = tk.Frame(card_kf, bg=COLORS['bg_card'])
            f_kf.pack(fill=tk.X, pady=2)

            btn_set = tk.Button(
                f_kf, text="Definir", bg=COLORS['accent_blue'], fg='#11111B',
                font=("Segoe UI", 7, "bold"), relief='flat', padx=6, cursor='hand2',
                command=lambda k=key: self.set_current_as_keyframe(k)
            )
            btn_set.pack(side=tk.LEFT, padx=(0, 6))

            btn_goto = tk.Button(
                f_kf, text="Ir", bg=COLORS['bg_sidebar'], fg=COLORS['text_main'],
                font=("Segoe UI", 7), relief='flat', padx=4, cursor='hand2',
                command=lambda k=key: self.goto_keyframe(k)
            )
            btn_goto.pack(side=tk.LEFT, padx=(0, 6))

            lbl_name = tk.Label(f_kf, text=title, fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 8))
            lbl_name.pack(side=tk.LEFT)

            current_val = self.calib_keyframes.get(key, "--")
            lbl_val = tk.Label(f_kf, text=f"Q #{current_val}", fg=COLORS['accent_yellow'], bg=COLORS['bg_card'], font=("Segoe UI", 8, "bold"))
            lbl_val.pack(side=tk.RIGHT)
            setattr(self, f"lbl_kf_{key}", lbl_val)

    def load_video(self, path):
        if not os.path.exists(path):
            return
        self.video_path = path
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        json_path = path.replace('.mp4', '_landmarks.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.video_landmarks = json.load(f)
                print(f"[LOG] Loaded {len(self.video_landmarks)} frames of landmarks from {json_path}")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar landmarks: {e}")
                self.video_landmarks = []
        else:
            print(f"[AVISO] Arquivo {json_path} não encontrado! O vídeo pode não conter a calibração de privacidade.")
            self.video_landmarks = []

        self.slider.configure(to=max(1, self.total_frames - 1))
        self.seek_frame(0)

    def choose_video_file(self):
        v_path = filedialog.askopenfilename(
            initialdir=RECORDINGS_DIR,
            title="Escolha o vídeo gravado para inspeção",
            filetypes=[("Vídeos MP4", "*.mp4"), ("Todos os Arquivos", "*.*")]
        )
        if v_path:
            self.load_video(v_path)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.configure(text="⏸ Pause", bg=COLORS['accent_red'])
            self.play_loop()
        else:
            self.btn_play_pause.configure(text="▶ Play", bg=COLORS['accent_green'])

    def play_loop(self):
        if not self.is_playing:
            return
        if self.current_frame_idx >= self.total_frames - 1:
            self.current_frame_idx = 0
        else:
            self.current_frame_idx += 1

        self.seek_frame(self.current_frame_idx)
        self.root.after(int(1000 / self.fps), self.play_loop)

    def step_frame(self, delta):
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
        self.seek_frame(new_idx)

    def on_slider_moved(self, val):
        target = int(float(val))
        if target != self.current_frame_idx:
            self.seek_frame(target)

    def seek_frame(self, frame_idx):
        if not self.cap or not self.cap.isOpened():
            return

        self.current_frame_idx = frame_idx
        self.slider_var.set(frame_idx)
        self.lbl_frame_info.configure(text=f"Quadro {frame_idx} / {self.total_frames - 1} ({frame_idx/self.fps:.2f}s)")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            telemetry_text = f"Quadro #{frame_idx} ({frame_idx/self.fps:.2f}s)\n"

            has_landmarks = False
            pts_raw = None
            if self.video_landmarks and frame_idx < len(self.video_landmarks):
                if self.video_landmarks[frame_idx] is not None:
                    has_landmarks = True
                    # Corrigir Aspect Ratio para proporção real (Isotrópico 1:1)
                    pts_raw = np.array([[lm['x'] * w, lm['y'] * h, lm['z'] * w] for lm in self.video_landmarks[frame_idx]])
            else:
                # Fallback to MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    has_landmarks = True
                    hand_lms = res.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    pts_raw = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms.landmark])
            
            if has_landmarks:
                wrist = pts_raw[0]
                palm_len = np.linalg.norm(pts_raw[9] - wrist)
                if palm_len > 1e-6:
                    self.current_pts_3d = (pts_raw - wrist) / palm_len
                    pts_norm = self.current_pts_3d

                    idx_f = (joint_flexion(pts_norm[0], pts_norm[5], pts_norm[6]) +
                             joint_flexion(pts_norm[5], pts_norm[6], pts_norm[7]) +
                             joint_flexion(pts_norm[6], pts_norm[7], pts_norm[8]))
                    mid_f = (joint_flexion(pts_norm[0], pts_norm[9], pts_norm[10]) +
                             joint_flexion(pts_norm[9], pts_norm[10], pts_norm[11]) +
                             joint_flexion(pts_norm[10], pts_norm[11], pts_norm[12]))
                    rng_f = (joint_flexion(pts_norm[0], pts_norm[13], pts_norm[14]) +
                             joint_flexion(pts_norm[13], pts_norm[14], pts_norm[15]) +
                             joint_flexion(pts_norm[14], pts_norm[15], pts_norm[16]))
                    pnk_f = (joint_flexion(pts_norm[0], pts_norm[17], pts_norm[18]) +
                             joint_flexion(pts_norm[17], pts_norm[18], pts_norm[19]) +
                             joint_flexion(pts_norm[18], pts_norm[19], pts_norm[20]))

                    thm_ip = joint_flexion(pts_norm[2], pts_norm[3], pts_norm[4])
                    thm_opp_dist = float(np.linalg.norm(pts_norm[4] - pts_norm[9]))

                    sp_mid_idx = vec_angle(pts_norm[9] - pts_norm[0], pts_norm[5] - pts_norm[0])

                    telemetry_text += (
                        f"Flexões: Ind:{idx_f:.0f}° | Méd:{mid_f:.0f}° | Anel:{rng_f:.0f}° | Min:{pnk_f:.0f}°\n"
                        f"Polegar: Dist Oposição:{thm_opp_dist:.3f} | Flex Ponta (IP):{thm_ip:.0f}°\n"
                        f"Spread Indicador-Médio:{sp_mid_idx:.1f}°"
                    )
            else:
                self.current_pts_3d = None
                telemetry_text += "[Mão não detectada neste quadro]"

            self.lbl_telemetry.configure(text=telemetry_text)

            # Render video image
            pv_w = self.left_box.winfo_width() - 20
            pv_h = self.left_box.winfo_height() - 60
            if pv_w < 100: pv_w = 540
            if pv_h < 100: pv_h = 400
            scale = min(pv_w / w, pv_h / h)
            dw, dh = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(frame, (dw, dh))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_video_preview.img_tk = img_tk
            self.lbl_video_preview.configure(image=img_tk)

            self.redraw_3d()

    def set_current_as_keyframe(self, key):
        if self.current_pts_3d is None:
            messagebox.showwarning("Aviso", "Nenhuma mão detectada no quadro atual para extrair.")
            return

        self.calib_keyframes[key] = self.current_frame_idx
        lbl = getattr(self, f"lbl_kf_{key}", None)
        if lbl:
            lbl.configure(text=f"Q #{self.current_frame_idx}")
        messagebox.showinfo("Keyframe Definido", f"Posição '{self.keyframe_labels[key]}' definida para o Quadro #{self.current_frame_idx}!")

    def goto_keyframe(self, key):
        if key in self.calib_keyframes:
            f_idx = self.calib_keyframes[key]
            self.seek_frame(f_idx)

    def reset_3d_view(self):
        self.view_rot_x = 0.0
        self.view_rot_y = 0.0
        self.redraw_3d()

    def save_official_calibration(self):
        if not self.video_path:
            return

        extracted_poses = {}
        for pose_name, target_f in self.calib_keyframes.items():
            has_landmarks = False
            pts = None
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            h, w, _ = frame.shape
            
            if self.video_landmarks and target_f < len(self.video_landmarks):
                if self.video_landmarks[target_f] is not None:
                    has_landmarks = True
                    pts = np.array([[lm['x'] * w, lm['y'] * h, lm['z'] * w] for lm in self.video_landmarks[target_f]])
            else:
                # Fallback to MediaPipe
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    has_landmarks = True
                    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in res.multi_hand_landmarks[0].landmark])

            if has_landmarks:
                wrist = pts[0]
                palm_len = np.linalg.norm(pts[9] - wrist)
                pts_norm = (pts - wrist) / palm_len
                extracted_poses[pose_name] = {"front": pts_norm.tolist(), "profile": None}
                print(f"[LOG] Posição '{pose_name}' extraída do quadro {target_f}.")
            else:
                print(f"[ERRO] Posição '{pose_name}' no quadro {target_f} não possui landmarks salvos e nem foi detectada no vídeo pelo fallback.")

        # Compute bone lengths and palm dimensions from stage_0_spread or stage_0_closed
        ref_pose = extracted_poses.get('stage_0_spread', {}).get('front') or extracted_poses.get('stage_0_closed', {}).get('front')
        avg_lengths = {}
        avg_palm = {}
        if ref_pose:
            ref_pts = np.array(ref_pose)
            avg_lengths = {
                'Thumb':  [float(np.linalg.norm(ref_pts[2] - ref_pts[1])),
                           float(np.linalg.norm(ref_pts[3] - ref_pts[2])),
                           float(np.linalg.norm(ref_pts[4] - ref_pts[3]))],
                'Index':  [float(np.linalg.norm(ref_pts[6] - ref_pts[5])),
                           float(np.linalg.norm(ref_pts[7] - ref_pts[6])),
                           float(np.linalg.norm(ref_pts[8] - ref_pts[7]))],
                'Middle': [float(np.linalg.norm(ref_pts[10] - ref_pts[9])),
                           float(np.linalg.norm(ref_pts[11] - ref_pts[10])),
                           float(np.linalg.norm(ref_pts[12] - ref_pts[11]))],
                'Ring':   [float(np.linalg.norm(ref_pts[14] - ref_pts[13])),
                           float(np.linalg.norm(ref_pts[15] - ref_pts[14])),
                           float(np.linalg.norm(ref_pts[16] - ref_pts[15]))],
                'Pinky':  [float(np.linalg.norm(ref_pts[18] - ref_pts[17])),
                           float(np.linalg.norm(ref_pts[19] - ref_pts[18])),
                           float(np.linalg.norm(ref_pts[20] - ref_pts[19]))]
            }
            avg_palm = {
                'Thumb':  float(np.linalg.norm(ref_pts[1] - ref_pts[0])),
                'Index':  float(np.linalg.norm(ref_pts[5] - ref_pts[0])),
                'Middle': float(np.linalg.norm(ref_pts[9] - ref_pts[0])),
                'Ring':   float(np.linalg.norm(ref_pts[13] - ref_pts[0])),
                'Pinky':  float(np.linalg.norm(ref_pts[17] - ref_pts[0]))
            }

        payload = {
            "captured_poses": extracted_poses,
            "avg_lengths": avg_lengths,
            "avg_palm": avg_palm,
            "source_video": self.video_path,
            "keyframe_indices": self.calib_keyframes
        }

        with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        print(f"[SUCESSO] Calibração oficial salva em: {CALIBRATION_FILE}")
        messagebox.showinfo("Sucesso", f"Calibração oficial salva com sucesso em:\n{CALIBRATION_FILE}")

    def on_drag_start(self, event):
        self.drag_start = (event.x, event.y)

    def on_drag_move(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.view_rot_y += dx * 0.5
            self.view_rot_x += dy * 0.5
            self.drag_start = (event.x, event.y)
            self.redraw_3d()

    def redraw_3d(self):
        self.canvas_3d.delete("all")
        if self.current_pts_3d is None:
            return

        w = self.canvas_3d.winfo_width() or 400
        h = self.canvas_3d.winfo_height() or 260

        R = rot_y(self.view_rot_y).dot(rot_x(self.view_rot_x))
        pts_rot = [R.dot(p) for p in self.current_pts_3d]

        xs = [p[0] for p in pts_rot]
        ys = [p[1] for p in pts_rot]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        span_x = max(abs(max_x - min_x), 0.1)
        span_y = max(abs(max_y - min_y), 0.1)

        scale = min((w * 0.7) / span_x, (h * 0.75) / span_y, 140.0)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        u_w = w / 2.0
        v_w = h / 2.0

        pts_2d = []
        for p in pts_rot:
            u = u_w + (p[0] - center_x) * scale
            v = v_w + (p[1] - center_y) * scale
            pts_2d.append((u, v))

        segs = {
            'Thumb':  [(0,1),(1,2),(2,3),(3,4)],
            'Index':  [(0,5),(5,6),(6,7),(7,8)],
            'Middle': [(0,9),(9,10),(10,11),(11,12)],
            'Ring':   [(0,13),(13,14),(14,15),(15,16)],
            'Pinky':  [(0,17),(17,18),(18,19),(19,20)]
        }

        for p1, p2 in [(0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)]:
            self.canvas_3d.create_line(pts_2d[p1][0], pts_2d[p1][1], pts_2d[p2][0], pts_2d[p2][1], fill='#555577', width=1, dash=(2,2))

        for finger, pairs in segs.items():
            col = FINGER_COLORS[finger]
            for p1, p2 in pairs:
                self.canvas_3d.create_line(pts_2d[p1][0], pts_2d[p1][1], pts_2d[p2][0], pts_2d[p2][1], fill=col, width=3, capstyle=tk.ROUND)

        for idx, (u, v) in enumerate(pts_2d):
            r = 3.5
            self.canvas_3d.create_oval(u-r, v-r, u+r, v+r, fill='#FFFFFF', outline='#000000')

def main():
    root = tk.Tk()
    app = VideoInspectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
