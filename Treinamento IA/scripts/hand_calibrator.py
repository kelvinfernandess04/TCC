import os
import json
import math
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEEDS_DIR = os.path.join(DATA_DIR, 'seeds')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'seed_verification')
RECORDINGS_DIR = os.path.join(DATA_DIR, 'recordings')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'calibration_settings.json')
SEEDS_FILE = os.path.join(SEEDS_DIR, 'seeds.json')

os.makedirs(SEEDS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

COLORS = {
    'bg_main': '#181825',
    'bg_sidebar': '#11111B',
    'bg_card': '#1E1E2E',
    'bg_canvas': '#0F0F17',
    'accent_blue': '#89B4FA',
    'accent_purple': '#CBA6F7',
    'accent_green': '#A6E3A1',
    'accent_yellow': '#F9E2AF',
    'accent_red': '#F38BA8',
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

def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

def fuse_dual_plane_landmarks(pose_dict):
    if not isinstance(pose_dict, dict):
        return None
    front = pose_dict.get('front')
    profile = pose_dict.get('profile')

    if front is None and profile is None:
        return None

    if front is not None and profile is not None:
        fused = np.array(front)
        fused[:, 2] = np.array(profile)[:, 0]
        return fused

    return np.array(front if front is not None else profile)

def get_fallback_hand():
    return np.array([
        [0.0, 0.0, 0.0],
        [-0.06, -0.04, -0.02], [-0.11, -0.09, -0.04], [-0.15, -0.14, -0.05], [-0.18, -0.18, -0.06],
        [-0.08, -0.25, 0.00], [-0.10, -0.35, 0.00], [-0.11, -0.42, 0.00], [-0.12, -0.48, 0.00],
        [0.00, -0.26, 0.00], [0.00, -0.37, 0.00], [0.00, -0.45, 0.00], [0.00, -0.52, 0.00],
        [0.08, -0.24, 0.00], [0.10, -0.34, 0.00], [0.11, -0.41, 0.00], [0.12, -0.47, 0.00],
        [0.16, -0.20, 0.00], [0.19, -0.28, 0.00], [0.21, -0.34, 0.00], [0.22, -0.39, 0.00]
    ])

def generate_anatomical_hand_3d(finger_states, spread_states, thumb_opp, thumb_ip, captured_poses=None):
    fallback = get_fallback_hand()

    if not captured_poses:
        captured_poses = {}

    p_0_spread = fuse_dual_plane_landmarks(captured_poses.get('stage_0_spread'))
    p_0_closed = fuse_dual_plane_landmarks(captured_poses.get('stage_0_closed'))
    p_1        = fuse_dual_plane_landmarks(captured_poses.get('stage_1'))
    p_2        = fuse_dual_plane_landmarks(captured_poses.get('stage_2'))
    p_3        = fuse_dual_plane_landmarks(captured_poses.get('stage_3'))
    p_opp      = fuse_dual_plane_landmarks(captured_poses.get('thumb_opposition'))
    p_ip       = fuse_dual_plane_landmarks(captured_poses.get('thumb_ip_flexed'))

    if p_0_spread is None: p_0_spread = fallback
    if p_0_closed is None: p_0_closed = p_0_spread
    if p_1 is None: p_1 = p_0_spread
    if p_2 is None: p_2 = p_1
    if p_3 is None: p_3 = p_2

    lms = np.zeros((21, 3))
    lms[0] = p_0_spread[0]
    for idx in [1, 5, 9, 13, 17]:
        lms[idx] = p_0_spread[idx]

    def get_finger_chain(pose, mcp_idx, tip_idxs):
        mcp = pose[mcp_idx]
        return [pose[tip_idxs[0]] - mcp, pose[tip_idxs[1]] - mcp, pose[tip_idxs[2]] - mcp]

    # Fingers 4 (Index, Middle, Ring, Pinky)
    fingers_info = [
        ('Index',  5,  [6, 7, 8],   'Middle_Index',  8.0),
        ('Middle', 9,  [10, 11, 12], 'Middle_Index', 0.0),
        ('Ring',   13, [14, 15, 16], 'Ring_Middle',  -6.0),
        ('Pinky',  17, [18, 19, 20], 'Pinky_Ring',   -14.0)
    ]

    for finger, mcp_idx, tip_idxs, sp_key, abduct_angle in fingers_info:
        st = float(finger_states.get(finger, 0.0))
        sp = float(spread_states.get(sp_key, 0.0))

        # Base chain from stage_0_closed (sp=0) to stage_0_spread (sp=1)
        c_open_closed = get_finger_chain(p_0_closed, mcp_idx, tip_idxs)
        c_open_spread = get_finger_chain(p_0_spread, mcp_idx, tip_idxs)
        c_0 = [c_open_closed[i] * (1.0 - sp) + c_open_spread[i] * sp for i in range(3)]

        c_1 = get_finger_chain(p_1, mcp_idx, tip_idxs)
        c_2 = get_finger_chain(p_2, mcp_idx, tip_idxs)
        c_3 = get_finger_chain(p_3, mcp_idx, tip_idxs)

        if st <= 1.0:
            c_final = [c_0[i] * (1.0 - st) + c_1[i] * st for i in range(3)]
        elif st <= 2.0:
            w = st - 1.0
            c_final = [c_1[i] * (1.0 - w) + c_2[i] * w for i in range(3)]
        else:
            w = st - 2.0
            c_final = [c_2[i] * (1.0 - w) + c_3[i] * w for i in range(3)]

        if sp > 0.01 and st <= 1.0:
            R_sp = rot_z(abduct_angle * sp)
            c_final = [R_sp.dot(v) for v in c_final]

        mcp_pos = lms[mcp_idx]
        lms[tip_idxs[0]] = mcp_pos + c_final[0]
        lms[tip_idxs[1]] = mcp_pos + c_final[1]
        lms[tip_idxs[2]] = mcp_pos + c_final[2]

    # Thumb Kinematics (Anatomical: CMC -> MCP -> IP -> TIP)
    t_st = float(finger_states.get('Thumb', 0.0))
    t_opp = float(thumb_opp)
    t_ip = float(thumb_ip)
    t_sp = float(spread_states.get('Index_Thumb', 0.0))

    cmc_pos = lms[1]
    
    # 1. Spread (Abertura Polegar-Indicador: 0 = aberto/leque, 1 = fechado/unido)
    t_c_open_spread = [p_0_spread[idx] - p_0_spread[1] for idx in [2, 3, 4]]
    t_c_open_closed = [p_0_closed[idx] - p_0_closed[1] for idx in [2, 3, 4]]
    t_c_base = [t_c_open_spread[i] * (1.0 - t_sp) + t_c_open_closed[i] * t_sp for i in range(3)]

    # 2. Opposition (Movimento Transversal F: cruzando na frente da palma)
    if p_opp is not None:
        t_c_opp = [p_opp[idx] - p_opp[1] for idx in [2, 3, 4]]
    else:
        idx_mcp = lms[5]
        mid_mcp = lms[9]
        vec_idx = idx_mcp - cmc_pos
        vec_mid = mid_mcp - cmc_pos
        t_c_opp = [
            vec_idx * 0.5 + np.array([0.04, 0.0, -0.04]),
            vec_idx * 0.75 + np.array([0.08, 0.0, -0.05]),
            vec_mid * 0.85 + np.array([0.12, 0.0, -0.06])
        ]

    opp_weight = max(t_opp, 1.0 if t_st >= 2.5 else 0.0)
    t_c_final = [t_c_base[i] * (1.0 - opp_weight) + t_c_opp[i] * opp_weight for i in range(3)]

    # 3. IP Flexion (Ponta do Polegar P: flexão da falange distal IP->TIP)
    if t_ip > 0.01:
        if p_ip is not None:
            vec_tip_ip_flexed = p_ip[4] - p_ip[3]
            vec_tip_ip_spread = p_0_spread[4] - p_0_spread[3]
            delta_tip = vec_tip_ip_flexed - vec_tip_ip_spread
            t_c_final[2] = t_c_final[2] + delta_tip * t_ip
        else:
            t_c_final[2][1] += t_ip * 0.04
            t_c_final[2][2] -= t_ip * 0.05

    lms[2] = cmc_pos + t_c_final[0]
    lms[3] = cmc_pos + t_c_final[1]
    lms[4] = cmc_pos + t_c_final[2]

    return lms


class HandCalibratorMainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Centro de Calibração Anatômica e Vídeo 3D - LIBRAS TCC")
        self.root.geometry("1340x840")
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

        self.captured_poses = {
            'stage_0_spread': {'front': None, 'profile': None},
            'stage_0_closed': {'front': None, 'profile': None},
            'stage_1':         {'front': None, 'profile': None},
            'stage_2':         {'front': None, 'profile': None},
            'stage_3':         {'front': None, 'profile': None},
            'thumb_opposition':{'front': None, 'profile': None},
            'thumb_ip_flexed': {'front': None, 'profile': None}
        }

        self.pose_cards_config = [
            ('stage_0_spread', '1. Mão Espalmada Aberta (Leque)', 'Dedos 100% estendidos e abertos em leque'),
            ('stage_0_closed', '2. Mão Espalmada Dedos Juntos', 'Dedos 100% estendidos e encostados (paralelos)'),
            ('stage_1',         '3. Mão em Garra Leve',        'Dedos curvados em formato de garra'),
            ('stage_2',         '4. Mão em Plataforma / Hook', 'Dedos dobrados em ângulo reto / gancho'),
            ('stage_3',         '5. Punho Fechado / Soco',     'Mão completamente fechada em soco'),
            ('thumb_opposition','6. Polegar em Oposição (F=1)', 'Polegar dobrado sobre a palma da mão'),
            ('thumb_ip_flexed', '7. Polegar Ponta Dobrada (P=1)', 'Apenas a falange distal (ponta) flexionada')
        ]

        self.selected_pose_key = tk.StringVar(value='stage_0_spread')
        self.view_mode = tk.StringVar(value='animator')

        # Simulator Sliders
        self.sim_states = {
            'Index':  tk.DoubleVar(value=0.0),
            'Middle': tk.DoubleVar(value=0.0),
            'Ring':   tk.DoubleVar(value=0.0),
            'Pinky':  tk.DoubleVar(value=0.0),
            'Thumb':  tk.DoubleVar(value=0.0),
            'Thumb_Opp': tk.DoubleVar(value=0.0),
            'Thumb_IP':  tk.DoubleVar(value=0.0),
            'Index_Thumb':  tk.DoubleVar(value=0.0),
            'Middle_Index': tk.DoubleVar(value=0.0),
            'Ring_Middle':  tk.DoubleVar(value=0.0),
            'Pinky_Ring':   tk.DoubleVar(value=0.0)
        }

        # Animation Playback State
        self.is_animating = False
        self.anim_t = 0.0
        self.anim_mode = tk.StringVar(value="LIBRAS")
        self.current_anim_label = "0000000000"
        self.current_anim_title = "Mão Espalmada Aberta"

        self.anim_sequences = {
            "LIBRAS": [
                ("0000000000", "Mão Aberta Espalmada"),
                ("3030303001", "Sinal 'A' (Fechado, Polegar Encostado)"),
                ("0030303000", "Sinal 'I' (Mindinho Levantado)"),
                ("3030000000", "Sinal 'V' (Indicador e Médio em V)"),
                ("3000000000", "Sinal 'W' (Indicador, Médio e Anelar)"),
                ("3030303000", "Punho Fechado / Soco")
            ],
            "STAGES": [
                ("0000000000", "Estágio 0: Mão Espalmada Aberta"),
                ("1010101000", "Estágio 1: Garra Leve"),
                ("2020202000", "Estágio 2: Plataforma / Hook"),
                ("3030303000", "Estágio 3: Punho Fechado")
            ],
            "SPREADS": [
                ("0000000000", "Sem Abertura (Dedos Paralelos)"),
                ("0000000011", "Abertura Polegar-Indicador (A0=1)"),
                ("0000011100", "Abertura Indicador-Médio (A1=1)"),
                ("0001011000", "Abertura Total em Leque (A=1)")
            ]
        }

        # Camera angles
        self.view_rot_x = 15.0
        self.view_rot_y = -20.0
        self.drag_start = None

        self.active_capture_key = None
        self.active_capture_plane = None
        self.camera_win = None
        self.live_lms_3d = None

        self.load_calibration()
        self.build_ui()
        self.redraw_3d()

    def load_calibration(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "captured_poses" in data:
                    raw_data = data["captured_poses"]
                    for k in self.captured_poses:
                        if k in raw_data:
                            val = raw_data[k]
                            if isinstance(val, dict):
                                self.captured_poses[k] = val
                            elif isinstance(val, list):
                                self.captured_poses[k] = {'front': val, 'profile': None}
                    print(f"[CALIBRAÇÃO] Dados anatômicos carregados de {CALIBRATION_FILE}.")
            except Exception as e:
                print(f"[AVISO] Erro ao carregar calibração: {e}")

    def save_calibration(self):
        try:
            out_data = {"captured_poses": self.captured_poses}
            with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(out_data, f, indent=2)
            messagebox.showinfo("Sucesso", f"Calibração salva com sucesso em:\n{CALIBRATION_FILE}")
            print(f"[SUCESSO] Calibração gravada em {CALIBRATION_FILE}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar calibração: {e}")

    def build_ui(self):
        # Sidebar Container
        sidebar = tk.Frame(self.root, width=540, bg=COLORS['bg_sidebar'], padx=15, pady=15)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Title Block
        lbl_t = tk.Label(sidebar, text="CALIBRADOR BIOMECÂNICO & VÍDEO 3D", fg=COLORS['accent_blue'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 12, "bold"))
        lbl_t.pack(anchor=tk.W)
        lbl_sub = tk.Label(sidebar, text="Gravação de vídeo, extração automática e simulação em tempo real", fg=COLORS['text_muted'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 8))
        lbl_sub.pack(anchor=tk.W, pady=(0, 8))

        # Action Buttons at Bottom
        btn_area = tk.Frame(sidebar, bg=COLORS['bg_sidebar'], pady=8)
        btn_area.pack(side=tk.BOTTOM, fill=tk.X)

        btn_record = tk.Button(
            btn_area, text="🎥 GRAVAR NOVO VÍDEO BIOMECÂNICO", bg=COLORS['accent_purple'], fg='#11111B',
            font=("Segoe UI", 10, "bold"), relief='flat', pady=7, cursor='hand2',
            command=self.open_video_recorder
        )
        btn_record.pack(fill=tk.X, pady=2)

        btn_inspect = tk.Button(
            btn_area, text="🔍 INSPECIONAR GRAVAÇÃO (TIMELINE & KEYFRAMES)", bg=COLORS['accent_green'], fg='#11111B',
            font=("Segoe UI", 10, "bold"), relief='flat', pady=7, cursor='hand2',
            command=self.open_video_inspector
        )
        btn_inspect.pack(fill=tk.X, pady=2)

        btn_analyze_video = tk.Button(
            btn_area, text="📂 PROCESSAR VÍDEO GRAVADO (EXTRAIR LIMITES)", bg=COLORS['accent_blue'], fg='#11111B',
            font=("Segoe UI", 10, "bold"), relief='flat', pady=7, cursor='hand2',
            command=self.select_and_process_video
        )
        btn_analyze_video.pack(fill=tk.X, pady=2)

        btn_gen = tk.Button(
            btn_area, text="⚡ GERAR SEMENTES & RELATÓRIOS VISUAIS", bg=COLORS['accent_yellow'], fg='#11111B',
            font=("Segoe UI", 10, "bold"), relief='flat', pady=7, cursor='hand2',
            command=self.generate_seeds_and_reports
        )
        btn_gen.pack(fill=tk.X, pady=2)

        # Tabs
        notebook = ttk.Notebook(sidebar)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tab_anim = tk.Frame(notebook, bg=COLORS['bg_sidebar'])
        tab_captures = tk.Frame(notebook, bg=COLORS['bg_sidebar'])
        tab_sim = tk.Frame(notebook, bg=COLORS['bg_sidebar'])

        notebook.add(tab_anim, text="▶️ Animação")
        notebook.add(tab_captures, text="📷 Poses Registradas")
        notebook.add(tab_sim, text="🎛️ Sliders Manuais")

        # TAB 1: ANIMATION
        pnl_anim = tk.Frame(tab_anim, bg=COLORS['bg_card'], padx=12, pady=12)
        pnl_anim.pack(fill=tk.BOTH, expand=True, pady=5)

        lbl_a_t = tk.Label(pnl_anim, text="Simulação de Movimento das Classes", fg=COLORS['accent_yellow'], bg=COLORS['bg_card'], font=("Segoe UI", 11, "bold"))
        lbl_a_t.pack(anchor=tk.W, pady=(0, 6))

        self.btn_play = tk.Button(
            pnl_anim, text="▶️ INICIAR ANIMAÇÃO DA BASE", bg=COLORS['accent_green'], fg='#11111B',
            font=("Segoe UI", 11, "bold"), relief='flat', pady=10, cursor='hand2',
            command=self.toggle_animation
        )
        self.btn_play.pack(fill=tk.X, pady=6)

        lbl_mode_t = tk.Label(pnl_anim, text="Modo da Animação:", fg=COLORS['accent_blue'], bg=COLORS['bg_card'], font=("Segoe UI", 9, "bold"))
        lbl_mode_t.pack(anchor=tk.W, pady=(10, 4))

        modes = [
            ("Demonstração de Gestos de LIBRAS (A, I, V, W)", "LIBRAS"),
            ("Varredura de Estágios dos Dedos (0 -> 3)", "STAGES"),
            ("Varredura de Aberturas Laterais / Spreads", "SPREADS")
        ]

        for m_text, m_val in modes:
            rb = tk.Radiobutton(
                pnl_anim, text=m_text, variable=self.anim_mode, value=m_val,
                fg=COLORS['text_main'], bg=COLORS['bg_card'], selectcolor=COLORS['bg_main'],
                font=("Segoe UI", 8), anchor='w'
            )
            rb.pack(fill=tk.X, pady=2)

        self.lbl_anim_hud = tk.Label(
            pnl_anim, text="Status: Pausado\nClasse: 0000000000", fg=COLORS['accent_purple'], bg=COLORS['bg_main'],
            font=("Segoe UI", 9, "bold"), pady=8, padx=8, justify=tk.LEFT
        )
        self.lbl_anim_hud.pack(fill=tk.X, pady=(15, 0))

        # TAB 2: CAPTURED POSES
        canvas_scroll = tk.Canvas(tab_captures, bg=COLORS['bg_sidebar'], highlightthickness=0)
        scroller = ttk.Scrollbar(tab_captures, orient="vertical", command=canvas_scroll.yview)
        scroll_frame = tk.Frame(canvas_scroll, bg=COLORS['bg_sidebar'])

        scroll_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scroller.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroller.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_labels = {}
        self.card_frames = {}

        for key, title, desc in self.pose_cards_config:
            card = tk.Frame(scroll_frame, bg=COLORS['bg_card'], padx=10, pady=8, cursor='hand2')
            card.pack(fill=tk.X, pady=4)
            self.card_frames[key] = card

            card.bind("<Button-1>", lambda e, k=key: self.select_pose_for_viewing(k))

            lbl_card_t = tk.Label(card, text=title, fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 9, "bold"), cursor='hand2')
            lbl_card_t.pack(anchor=tk.W)
            lbl_card_t.bind("<Button-1>", lambda e, k=key: self.select_pose_for_viewing(k))

            lbl_card_d = tk.Label(card, text=desc, fg=COLORS['text_muted'], bg=COLORS['bg_card'], font=("Segoe UI", 8), cursor='hand2')
            lbl_card_d.pack(anchor=tk.W, pady=(0, 4))
            lbl_card_d.bind("<Button-1>", lambda e, k=key: self.select_pose_for_viewing(k))

            btn_f = tk.Frame(card, bg=COLORS['bg_card'])
            btn_f.pack(fill=tk.X)

            btn_front = tk.Button(
                btn_f, text="Frente (0°)", bg=COLORS['accent_blue'], fg='#11111B',
                font=("Segoe UI", 8, "bold"), relief='flat', padx=8, pady=3, cursor='hand2',
                command=lambda k=key: self.open_camera_for_pose(k, 'front')
            )
            btn_front.pack(side=tk.LEFT, padx=(0, 4))

            btn_profile = tk.Button(
                btn_f, text="Perfil (90°)", bg=COLORS['accent_purple'], fg='#11111B',
                font=("Segoe UI", 8, "bold"), relief='flat', padx=8, pady=3, cursor='hand2',
                command=lambda k=key: self.open_camera_for_pose(k, 'profile')
            )
            btn_profile.pack(side=tk.LEFT)

            lbl_st = tk.Label(btn_f, text=self.get_status_text(key), fg=COLORS['accent_green'], bg=COLORS['bg_card'], font=("Segoe UI", 8, "bold"))
            lbl_st.pack(side=tk.RIGHT)
            self.status_labels[key] = lbl_st

        # TAB 3: MANUAL SLIDERS
        sim_scroll = tk.Canvas(tab_sim, bg=COLORS['bg_sidebar'], highlightthickness=0)
        sim_scroller = ttk.Scrollbar(tab_sim, orient="vertical", command=sim_scroll.yview)
        sim_frame = tk.Frame(sim_scroll, bg=COLORS['bg_sidebar'])

        sim_frame.bind("<Configure>", lambda e: sim_scroll.configure(scrollregion=sim_scroll.bbox("all")))
        sim_scroll.create_window((0, 0), window=sim_frame, anchor="nw")
        sim_scroll.configure(yscrollcommand=sim_scroller.set)

        sim_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sim_scroller.pack(side=tk.RIGHT, fill=tk.Y)

        slider_groups = [
            ("Flexão dos Dedos (0: Extendido -> 3: Dobrado)", [
                ("Indicador (D1)", 'Index', 0.0, 3.0),
                ("Médio (D2)", 'Middle', 0.0, 3.0),
                ("Anelar (D3)", 'Ring', 0.0, 3.0),
                ("Mindinho (D4)", 'Pinky', 0.0, 3.0),
                ("Polegar (D_Thumb)", 'Thumb', 0.0, 3.0)
            ]),
            ("Estados do Polegar", [
                ("Polegar Oposição (F)", 'Thumb_Opp', 0.0, 1.0),
                ("Polegar Ponta Flex (P)", 'Thumb_IP', 0.0, 1.0)
            ]),
            ("Aberturas Laterais (Spreads)", [
                ("Mindinho - Anelar (A3)", 'Pinky_Ring', 0.0, 1.0),
                ("Anelar - Médio (A2)", 'Ring_Middle', 0.0, 1.0),
                ("Médio - Indicador (A1)", 'Middle_Index', 0.0, 1.0),
                ("Indicador - Polegar (A0)", 'Index_Thumb', 0.0, 1.0)
            ])
        ]

        for grp_title, sliders in slider_groups:
            grp = tk.LabelFrame(sim_frame, text=grp_title, fg=COLORS['accent_blue'], bg=COLORS['bg_card'], padx=10, pady=8, font=("Segoe UI", 9, "bold"))
            grp.pack(fill=tk.X, pady=6)

            for s_name, s_key, s_min, s_max in sliders:
                f_s = tk.Frame(grp, bg=COLORS['bg_card'])
                f_s.pack(fill=tk.X, pady=2)

                lbl_s = tk.Label(f_s, text=s_name, fg=COLORS['text_main'], bg=COLORS['bg_card'], width=20, anchor='w', font=("Segoe UI", 8))
                lbl_s.pack(side=tk.LEFT)

                scale = tk.Scale(
                    f_s, from_=s_min, to=s_max, resolution=0.1, orient=tk.HORIZONTAL,
                    variable=self.sim_states[s_key], bg=COLORS['bg_card'], fg=COLORS['accent_green'],
                    troughcolor=COLORS['bg_main'], highlightthickness=0, length=180,
                    command=lambda val: self.on_slider_changed()
                )
                scale.pack(side=tk.RIGHT)

        # 3D Canvas Area
        canvas_box = tk.Frame(self.root, bg=COLORS['bg_canvas'])
        canvas_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        hdr_frame = tk.Frame(canvas_box, bg=COLORS['bg_card'], padx=15, pady=8)
        hdr_frame.pack(fill=tk.X)

        lbl_head = tk.Label(hdr_frame, text="Exibindo:", fg=COLORS['accent_blue'], bg=COLORS['bg_card'], font=("Segoe UI", 9, "bold"))
        lbl_head.pack(side=tk.LEFT, padx=(0, 10))

        rb_anim = tk.Radiobutton(
            hdr_frame, text="▶️ Animação de Classes", variable=self.view_mode, value="animator",
            fg=COLORS['accent_green'], bg=COLORS['bg_card'], selectcolor=COLORS['bg_main'],
            font=("Segoe UI", 9, "bold"), command=self.redraw_3d
        )
        rb_anim.pack(side=tk.LEFT, padx=5)

        rb_sim = tk.Radiobutton(
            hdr_frame, text="🎛️ Sliders Manuais", variable=self.view_mode, value="simulator",
            fg=COLORS['accent_yellow'], bg=COLORS['bg_card'], selectcolor=COLORS['bg_main'],
            font=("Segoe UI", 9, "bold"), command=self.redraw_3d
        )
        rb_sim.pack(side=tk.LEFT, padx=5)

        rb_pose = tk.Radiobutton(
            hdr_frame, text="📷 Pose Registrada", variable=self.view_mode, value="pose",
            fg=COLORS['accent_blue'], bg=COLORS['bg_card'], selectcolor=COLORS['bg_main'],
            font=("Segoe UI", 9, "bold"), command=self.redraw_3d
        )
        rb_pose.pack(side=tk.LEFT, padx=5)

        pose_options = [title for k, title, d in self.pose_cards_config]
        self.pose_combo = ttk.Combobox(hdr_frame, values=pose_options, state="readonly", width=25, font=("Segoe UI", 8))
        self.pose_combo.current(0)
        self.pose_combo.pack(side=tk.LEFT, padx=5)
        self.pose_combo.bind("<<ComboboxSelected>>", self.on_combo_pose_selected)

        lbl_hint = tk.Label(canvas_box, text="Arraste com o mouse para rotacionar a câmera 3D | Visualização 100% enquadrada", fg=COLORS['text_muted'], bg=COLORS['bg_canvas'], font=("Segoe UI", 8))
        lbl_hint.pack(pady=(4, 0))

        self.canvas_3d = tk.Canvas(canvas_box, bg=COLORS['bg_canvas'], highlightthickness=0)
        self.canvas_3d.pack(fill=tk.BOTH, expand=True)

        self.canvas_3d.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas_3d.bind("<B1-Motion>", self.on_drag_move)

    def open_video_recorder(self):
        cmd = f'python "{os.path.join(BASE_DIR, "scripts", "video_recorder.py")}"'
        subprocess.Popen(cmd, shell=True)

    def open_video_inspector(self):
        cmd = f'python "{os.path.join(BASE_DIR, "scripts", "video_inspector.py")}"'
        subprocess.Popen(cmd, shell=True)

    def select_and_process_video(self):
        video_path = filedialog.askopenfilename(
            initialdir=RECORDINGS_DIR,
            title="Selecione o vídeo da mão gravado",
            filetypes=[("Vídeos MP4", "*.mp4"), ("Todos os Arquivos", "*.*")]
        )
        if video_path:
            from video_calibrator import VideoRangeCalibrator
            analyzer = VideoRangeCalibrator()
            success, msg = analyzer.process_video(video_path)
            if success:
                self.load_calibration()
                for k in self.status_labels:
                    self.status_labels[k].configure(text=self.get_status_text(k))
                self.redraw_3d()
                messagebox.showinfo("Calibração Concluída", f"Limites extraídos com sucesso do vídeo!\n\n{msg}")
            else:
                messagebox.showerror("Erro na Análise", f"Falha ao analisar vídeo:\n{msg}")

    def toggle_animation(self):
        self.is_animating = not self.is_animating
        if self.is_animating:
            self.load_calibration()
            self.view_mode.set("animator")
            self.btn_play.configure(text="⏸️ PAUSAR ANIMAÇÃO", bg=COLORS['accent_red'])
            self.tick_animation()
        else:
            self.btn_play.configure(text="▶️ INICIAR ANIMAÇÃO DA BASE", bg=COLORS['accent_green'])
            self.lbl_anim_hud.configure(text=f"Status: Pausado\nClasse Atual: {self.current_anim_label}")

    def tick_animation(self):
        if not self.is_animating:
            return

        seq = self.anim_sequences.get(self.anim_mode.get(), self.anim_sequences["LIBRAS"])
        total_steps = len(seq)
        
        self.anim_t += 0.04
        seq_idx = int(self.anim_t) % total_steps
        next_idx = (seq_idx + 1) % total_steps
        blend = self.anim_t - int(self.anim_t)

        t_ease = blend * blend * (3 - 2 * blend)

        label_curr, title_curr = seq[seq_idx]
        label_next, _ = seq[next_idx]

        self.current_anim_label = label_curr
        self.current_anim_title = title_curr

        def decode(l):
            d4, a3, d3, a2, d2, a1, d1, a0, f, p = [int(c) for c in l]
            return {
                'Pinky': d4, 'Ring': d3, 'Middle': d2, 'Index': d1, 'Thumb': p,
                'Pinky_Ring': a3, 'Ring_Middle': a2, 'Middle_Index': a1, 'Index_Thumb': a0,
                'Thumb_Opp': f, 'Thumb_IP': p
            }

        st_c = decode(label_curr)
        st_n = decode(label_next)

        interp = {}
        for k in st_c:
            interp[k] = st_c[k] + (st_n[k] - st_c[k]) * t_ease

        for k in self.sim_states:
            if k in interp:
                self.sim_states[k].set(round(interp[k], 2))

        self.lbl_anim_hud.configure(
            text=f"Status: ▶ REPRODUZINDO\nClasse: {label_curr}\n{title_curr}"
        )

        self.redraw_3d()
        self.root.after(33, self.tick_animation)

    def on_slider_changed(self):
        if self.view_mode.get() != "animator":
            self.view_mode.set("simulator")
        self.redraw_3d()

    def get_status_text(self, key):
        pose = self.captured_poses.get(key, {})
        has_f = pose.get('front') is not None
        has_p = pose.get('profile') is not None
        if has_f and has_p:
            return "Frente: OK | Perfil: OK"
        elif has_f:
            return "Frente: OK | Perfil: --"
        elif has_p:
            return "Frente: -- | Perfil: OK"
        else:
            return "[Pendente]"

    def select_pose_for_viewing(self, pose_key):
        self.selected_pose_key.set(pose_key)
        self.view_mode.set("pose")
        for idx, (k, t, d) in enumerate(self.pose_cards_config):
            if k == pose_key:
                self.pose_combo.current(idx)
                break
        self.redraw_3d()

    def on_combo_pose_selected(self, event):
        idx = self.pose_combo.current()
        if 0 <= idx < len(self.pose_cards_config):
            key = self.pose_cards_config[idx][0]
            self.selected_pose_key.set(key)
            self.view_mode.set("pose")
            self.redraw_3d()

    def open_camera_for_pose(self, pose_key, plane='front'):
        self.active_capture_key = pose_key
        self.active_capture_plane = plane
        pose_title = [t for k, t, d in self.pose_cards_config if k == pose_key][0]
        plane_str = "FRENTE (0°)" if plane == 'front' else "PERFIL (90° virado de lado)"

        self.camera_win = tk.Toplevel(self.root)
        self.camera_win.title(f"Captura da Câmera: {pose_title} - Plano: {plane_str}")
        self.camera_win.geometry("720x560")
        self.camera_win.configure(bg=COLORS['bg_main'])

        lbl_info = tk.Label(self.camera_win, text=f"Captura para: {pose_title}\nPlano: {plane_str} | Pressione [ESPAÇO] para capturar", fg=COLORS['accent_blue'], bg=COLORS['bg_main'], font=("Segoe UI", 10, "bold"))
        lbl_info.pack(pady=8)

        self.cam_label = tk.Label(self.camera_win, bg='#000000')
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_cap = tk.Button(self.camera_win, text=f"CAPTURAR {plane_str} AGORA [ESPAÇO]", bg=COLORS['accent_green'], fg='#11111B', font=("Segoe UI", 11, "bold"), pady=8, cursor='hand2', command=self.confirm_current_capture)
        btn_cap.pack(fill=tk.X, padx=10, pady=10)

        self.camera_win.bind("<space>", lambda e: self.confirm_current_capture())
        self.camera_win.protocol("WM_DELETE_WINDOW", self.close_camera_win)

        self.cap = cv2.VideoCapture(0)
        self.update_camera_stream()

    def update_camera_stream(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened() or self.camera_win is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )

                pts_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
                wrist = pts_raw[0]
                palm_len = np.linalg.norm(pts_raw[9] - wrist)
                if palm_len > 1e-6:
                    self.live_lms_3d = (pts_raw - wrist) / palm_len

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            self.cam_label.img_tk = img_tk
            self.cam_label.configure(image=img_tk)

        if self.camera_win is not None:
            self.root.after(20, self.update_camera_stream)

    def confirm_current_capture(self):
        if self.live_lms_3d is not None and self.active_capture_key is not None:
            pose = self.captured_poses.get(self.active_capture_key, {})
            pose[self.active_capture_plane] = self.live_lms_3d.tolist()
            self.captured_poses[self.active_capture_key] = pose

            self.status_labels[self.active_capture_key].configure(text=self.get_status_text(self.active_capture_key))
            print(f"[SUCESSO] Capturada pose '{self.active_capture_key}' no plano '{self.active_capture_plane}'.")
            self.select_pose_for_viewing(self.active_capture_key)
            self.save_calibration()
            self.close_camera_win()
        else:
            messagebox.showwarning("Aviso", "Nenhuma mão detectada na câmera. Posicione sua mão em frente à webcam e tente novamente.")

    def close_camera_win(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if self.camera_win is not None:
            self.camera_win.destroy()
            self.camera_win = None

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

    def get_display_3d_landmarks(self):
        if self.view_mode.get() in ["simulator", "animator"]:
            f_states = {
                'Index':  self.sim_states['Index'].get(),
                'Middle': self.sim_states['Middle'].get(),
                'Ring':   self.sim_states['Ring'].get(),
                'Pinky':  self.sim_states['Pinky'].get(),
                'Thumb':  self.sim_states['Thumb'].get()
            }
            s_states = {
                'Index_Thumb':  self.sim_states['Index_Thumb'].get(),
                'Middle_Index': self.sim_states['Middle_Index'].get(),
                'Ring_Middle':  self.sim_states['Ring_Middle'].get(),
                'Pinky_Ring':   self.sim_states['Pinky_Ring'].get()
            }
            t_opp = self.sim_states['Thumb_Opp'].get()
            t_ip  = self.sim_states['Thumb_IP'].get()
            return generate_anatomical_hand_3d(f_states, s_states, t_opp, t_ip, self.captured_poses)
        else:
            key = self.selected_pose_key.get()
            pose_dict = self.captured_poses.get(key, {})
            front = pose_dict.get('front')
            profile = pose_dict.get('profile')

            if front is not None or profile is not None:
                if front is not None and profile is not None:
                    fused = np.zeros_like(front)
                    fused[:, 0] = np.array(front)[:, 0]
                    fused[:, 1] = np.array(front)[:, 1]
                    fused[:, 2] = np.array(profile)[:, 0]
                    return fused
                return np.array(front if front is not None else profile)

            return generate_anatomical_hand_3d({}, {}, 0, 0, self.captured_poses)

    def redraw_3d(self):
        self.canvas_3d.delete("all")
        w = self.canvas_3d.winfo_width() or 700
        h = self.canvas_3d.winfo_height() or 650

        pts_3d = self.get_display_3d_landmarks()
        R = rot_y(self.view_rot_y).dot(rot_x(self.view_rot_x))
        pts_rot = [R.dot(p) for p in pts_3d]

        xs = [p[0] for p in pts_rot]
        ys = [p[1] for p in pts_rot]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        span_x = max(abs(max_x - min_x), 0.1)
        span_y = max(abs(max_y - min_y), 0.1)

        scale = min((w * 0.65) / span_x, (h * 0.70) / span_y, 240.0)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        u_center = w / 2.0
        v_center = h / 2.0

        pts_2d = []
        for p in pts_rot:
            u = u_center + (p[0] - center_x) * scale
            v = v_center + (p[1] - center_y) * scale
            pts_2d.append((u, v))

        segment_indices = {
            'Thumb':  [(0,1),(1,2),(2,3),(3,4)],
            'Index':  [(0,5),(5,6),(6,7),(7,8)],
            'Middle': [(0,9),(9,10),(10,11),(11,12)],
            'Ring':   [(0,13),(13,14),(14,15),(15,16)],
            'Pinky':  [(0,17),(17,18),(18,19),(19,20)]
        }

        for start, end in [(0,1), (0,5), (0,9), (0,13), (0,17), (5,9), (9,13), (13,17)]:
            u1, v1 = pts_2d[start]
            u2, v2 = pts_2d[end]
            self.canvas_3d.create_line(u1, v1, u2, v2, fill='#555577', width=2, dash=(2, 2))

        for finger, segs in segment_indices.items():
            color = FINGER_COLORS[finger]
            for start, end in segs:
                u1, v1 = pts_2d[start]
                u2, v2 = pts_2d[end]
                self.canvas_3d.create_line(u1, v1, u2, v2, fill=color, width=4, capstyle=tk.ROUND)

        for idx, (u, v) in enumerate(pts_2d):
            r = 5
            self.canvas_3d.create_oval(u-r, v-r, u+r, v+r, fill='#FFFFFF', outline='#000000', width=1.5)
            self.canvas_3d.create_text(u+10, v, text=str(idx), fill='#A6ADC8', font=("Segoe UI", 8))

        if self.view_mode.get() == "animator":
            self.canvas_3d.create_rectangle(15, 15, 340, 75, fill='#11111B', outline=COLORS['accent_green'], width=2)
            self.canvas_3d.create_text(25, 30, text="▶ ANIMAÇÃO DE CLASSES AO VIVO", fill=COLORS['accent_green'], font=("Segoe UI", 9, "bold"), anchor='w')
            self.canvas_3d.create_text(25, 48, text=f"Classe: {self.current_anim_label}", fill=COLORS['accent_yellow'], font=("Segoe UI", 10, "bold"), anchor='w')
            self.canvas_3d.create_text(25, 63, text=f"{self.current_anim_title}", fill=COLORS['text_main'], font=("Segoe UI", 8), anchor='w')

    def generate_seeds_and_reports(self):
        try:
            self.load_calibration()
            print("\n[PROCESSANDO] Extraindo sementes e gerando relatório gráfico...")
            cmd = f'python "{os.path.join(BASE_DIR, "scripts", "seed_extractor.py")}"'
            os.system(cmd)

            cmd_vis = f'python "{os.path.join(BASE_DIR, "scripts", "generate_seed_limit_visualizations.py")}"'
            os.system(cmd_vis)

            messagebox.showinfo("Sucesso", "Sementes e Relatórios Visuais gerados com sucesso!\nVerifique a pasta: Treinamento IA/reports/seed_verification/")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar sementes: {e}")

def main():
    root = tk.Tk()
    app = HandCalibratorMainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
