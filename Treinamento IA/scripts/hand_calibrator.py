import os
import json
import cv2
import math
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Configurações de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, 'data', 'seeds', 'seeds.json')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'data', 'calibration_settings.json')

# Conexões da mão
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # Mínimo
    (5, 9), (9, 13), (13, 17)              # Palma
]

# Design System Colors (Dracula/Mocha theme)
COLORS = {
    'bg_main': '#1E1E2E',
    'bg_sidebar': '#181825',
    'bg_card': '#252538',
    'bg_canvas': '#11111B',
    'text_main': '#CDD6F4',
    'text_muted': '#A6ADC8',
    'accent_blue': '#89B4FA',
    'accent_green': '#A6E3A1',
    'accent_yellow': '#F9E2AF',
    'accent_red': '#F38BA8',
    'Thumb':  '#ffffff',   # Branco
    'Index':  '#cba6f7',   # Roxo
    'Middle': '#f9e2af',   # Amarelo
    'Ring':   '#a6e3a1',   # Verde
    'Pinky':  '#89b4fa',   # Azul
    'Palm': '#585B70'
}

# Mapeamentos e Constantes para o modelo de Yaw & Pitch por Ponto
JOINT_OPTIONS = [
    "Polegar - CMC (Junta 1)",
    "Polegar - MCP (Junta 2)",
    "Polegar - IP (Junta 3)",
    "Indicador - MCP (Junta 1)",
    "Indicador - PIP (Junta 2)",
    "Indicador - DIP (Junta 3)",
    "Médio - MCP (Junta 1)",
    "Médio - PIP (Junta 2)",
    "Médio - DIP (Junta 3)",
    "Anelar - MCP (Junta 1)",
    "Anelar - PIP (Junta 2)",
    "Anelar - DIP (Junta 3)",
    "Mindinho - MCP (Junta 1)",
    "Mindinho - PIP (Junta 2)",
    "Mindinho - DIP (Junta 3)"
]

JOINT_MAPPING = {
    "Polegar - CMC (Junta 1)":   ("Thumb",  "J1"),
    "Polegar - MCP (Junta 2)":   ("Thumb",  "J2"),
    "Polegar - IP (Junta 3)":    ("Thumb",  "J3"),
    "Indicador - MCP (Junta 1)": ("Index",  "J1"),
    "Indicador - PIP (Junta 2)": ("Index",  "J2"),
    "Indicador - DIP (Junta 3)": ("Index",  "J3"),
    "Médio - MCP (Junta 1)":     ("Middle", "J1"),
    "Médio - PIP (Junta 2)":     ("Middle", "J2"),
    "Médio - DIP (Junta 3)":     ("Middle", "J3"),
    "Anelar - MCP (Junta 1)":    ("Ring",   "J1"),
    "Anelar - PIP (Junta 2)":    ("Ring",   "J2"),
    "Anelar - DIP (Junta 3)":    ("Ring",   "J3"),
    "Mindinho - MCP (Junta 1)":  ("Pinky",  "J1"),
    "Mindinho - PIP (Junta 2)":  ("Pinky",  "J2"),
    "Mindinho - DIP (Junta 3)":  ("Pinky",  "J3")
}

JOINT_TO_LANDMARK = {
    ("Thumb",  "J1"): 1,
    ("Thumb",  "J2"): 2,
    ("Thumb",  "J3"): 3,
    ("Thumb",  "J4"): 4,
    ("Index",  "J1"): 5,
    ("Index",  "J2"): 6,
    ("Index",  "J3"): 7,
    ("Index",  "J4"): 8,
    ("Middle", "J1"): 9,
    ("Middle", "J2"): 10,
    ("Middle", "J3"): 11,
    ("Middle", "J4"): 12,
    ("Ring",   "J1"): 13,
    ("Ring",   "J2"): 14,
    ("Ring",   "J3"): 15,
    ("Ring",   "J4"): 16,
    ("Pinky",  "J1"): 17,
    ("Pinky",  "J2"): 18,
    ("Pinky",  "J3"): 19,
    ("Pinky",  "J4"): 20
}

LANDMARK_TO_JOINT = {
    1: ("Thumb",  "J1"),
    2: ("Thumb",  "J2"),
    3: ("Thumb",  "J3"),
    4: ("Thumb",  "J4"),
    5: ("Index",  "J1"),
    6: ("Index",  "J2"),
    7: ("Index",  "J3"),
    8: ("Index",  "J4"),
    9: ("Middle", "J1"),
    10: ("Middle", "J2"),
    11: ("Middle", "J3"),
    12: ("Middle", "J4"),
    13: ("Ring",   "J1"),
    14: ("Ring",   "J2"),
    15: ("Ring",   "J3"),
    16: ("Ring",   "J4"),
    17: ("Pinky",  "J1"),
    18: ("Pinky",  "J2"),
    19: ("Pinky",  "J3"),
    20: ("Pinky",  "J4")
}

# ---------------------------------------------------------
# MATRIZES DE ROTAÇÃO E KINEMATICS
# ---------------------------------------------------------
def rot_x(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

def rot_y(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

def rot_z(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

def lerp(a, b, t):
    return a + (b - a) * t

def calc_finger_chain_yaw_pitch(base_pos, base_rot_mat, lengths, yaws, pitches):
    points = [base_pos.copy()]
    current_mat = base_rot_mat.copy()
    current_pos = base_pos.copy()

    for length, yaw, pitch in zip(lengths, yaws, pitches):
        joint_rot = rot_z(yaw).dot(rot_x(pitch))
        current_mat = current_mat.dot(joint_rot)
        local_bone = np.array([0, length, 0])
        world_bone = current_mat.dot(local_bone)
        current_pos = current_pos + world_bone
        points.append(current_pos.copy())

    return points

# ---------------------------------------------------------
# CÁLCULOS ANGULARES DE SENSOR
# ---------------------------------------------------------
def vec_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))

def joint_flexion(p_prev, p_joint, p_next):
    v1 = p_prev - p_joint
    v2 = p_next - p_joint
    return 180.0 - vec_angle(v1, v2)

# ---------------------------------------------------------
# CLASSE DO CALIBRADOR INTERATIVO
# ---------------------------------------------------------
class HandCalibratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibrador Anatômico LIBRAS 3D")
        self.root.geometry("1366x768")
        self.root.configure(bg=COLORS['bg_main'])
        
        # Maximizar janela ao iniciar
        try:
            self.root.state('zoomed')
        except:
            pass

        # Parâmetros padrão da Mão
        self.avg_lengths = {
            'Thumb':  [0.0914, 0.0771, 0.0621],
            'Index':  [0.0998, 0.0640, 0.0532],
            'Middle': [0.1102, 0.0769, 0.0578],
            'Ring':   [0.1001, 0.0700, 0.0553],
            'Pinky':  [0.0768, 0.0517, 0.0454]
        }
        self.avg_palm = {
            'Thumb': 0.070, 'Index': 0.240, 'Middle': 0.245, 'Ring': 0.235, 'Pinky': 0.210
        }
        
        self.load_anatomical_proportions()

        # Configurações de Estado de Calibração por Estágio (0-3) baseadas em Juntas Yaw & Pitch
        self.stages = {}
        self.ranges = {} # Legacy ranges variable to avoid breakages

        default_ranges = {
            'Thumb':  {'MCP': [10.0, 50.0], 'PIP_DIP': [5.0, 60.0]},
            'Index':  {'MCP': [5.0, 60.0],  'PIP_DIP': [5.0, 90.0]},
            'Middle': {'MCP': [5.0, 75.0],  'PIP_DIP': [5.0, 110.0]},
            'Ring':   {'MCP': [5.0, 80.0],  'PIP_DIP': [5.0, 105.0]},
            'Pinky':  {'MCP': [5.0, 85.0],  'PIP_DIP': [5.0, 100.0]}
        }

        # Inicializar o dicionário self.stages com defaults
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            self.stages[f] = {}
            mcp_min, mcp_max = default_ranges[f]['MCP']
            pip_min, pip_max = default_ranges[f]['PIP_DIP']
            
            def_yaw = {'Thumb': -25.0, 'Index': 5.0, 'Middle': 0.0, 'Ring': -5.0, 'Pinky': -15.0}
            
            for s in range(4):
                state = str(s)
                mcp_val = mcp_min
                pip_val = pip_min
                if state == '1':
                    mcp_val = lerp(mcp_min, mcp_max, 0.15)
                    pip_val = lerp(pip_min, pip_max, 0.5)
                elif state == '2':
                    mcp_val = mcp_min
                    pip_val = pip_max
                elif state == '3':
                    mcp_val = mcp_max
                    pip_val = pip_max

                if f != 'Thumb':
                    self.stages[f][state] = {
                        'J1_Yaw': def_yaw[f],
                        'J1_Pitch': mcp_val,
                        'J2_Yaw': 0.0,
                        'J2_Pitch': pip_val,
                        'J3_Yaw': 0.0,
                        'J3_Pitch': pip_val,
                        'J4_Yaw': 0.0,
                        'J4_Pitch': pip_val
                    }
                else:
                    cy_stages = {'0': -25.0, '1': -31.6, '2': -36.1, '3': -21.2}
                    cp_stages = {'0': 5.4, '1': 14.5, '2': 21.1, '3': 37.3}
                    self.stages[f][state] = {
                        'J1_Yaw': cy_stages[state],
                        'J1_Pitch': cp_stages[state],
                        'J2_Yaw': 0.0,
                        'J2_Pitch': mcp_val,
                        'J3_Yaw': 0.0,
                        'J3_Pitch': pip_val,
                        'J4_Yaw': 0.0,
                        'J4_Pitch': pip_val
                    }

        # Estado da edição atual
        self.active_finger = 'Index'
        self.active_stage = '0'
        self.active_joint = 'J1'
        self.active_landmark_idx = 5  # Index MCP por padrão
        self.updating_gui = False  # Lock para evitar recursão infinita
        self.calibration_frozen = False  # Estado de congelamento da câmera
        
        self.spread_overrides = None
        self.thumb_opp_override = None
        self.thumb_ip_override = None

        # Estados Dinâmicos de Simulação (Visualização)
        self.finger_states = {
            'Pinky': 0, 'Ring': 0, 'Middle': 0, 'Index': 0, 'Thumb': 0
        }

        # Carregar calibração persistida se existir
        self.load_calibration_file()

        # Configurações de View 3D da Mão
        self.view_pitch = 340.0
        self.view_yaw = 325.0
        self.view_roll = 0.0

        # Variáveis de interação por mouse
        self.lms_2d = []
        self.mouse_hover_idx = None
        self.mouse_drag_idx = None
        self.camera_drag = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Variáveis da Câmera
        self.camera_active = False
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.live_ranges = {}
        self.reset_live_ranges()

        # Construir UI Completa
        self.build_ui()
        self.update_gui_from_model()
        
        # Aguardar um instante para o canvas ter tamanho real e então renderizar
        self.root.after(100, self.redraw_hand)

    # ---------------------------------------------------------
    # CARGA / SALVAMENTO DE CONFIGURAÇÕES
    # ---------------------------------------------------------
    def load_anatomical_proportions(self):
        if os.path.exists(SEEDS_FILE):
            try:
                with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
                    seeds = json.load(f)
                metadata = seeds.get("__metadata__", {})
                if "avg_lengths" in metadata:
                    self.avg_lengths = metadata["avg_lengths"]
                if "avg_palm" in metadata:
                    self.avg_palm = metadata["avg_palm"]
                print("[SISTEMA] Metadados anatômicos herdados do seeds.json.")
            except Exception as e:
                print(f"[Aviso] Falha ao ler seeds.json: {e}")

    def load_calibration_file(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                if "stages" in saved:
                    loaded_stages = saved["stages"]
                    # Sanitização, defaults anatômicos e retrocompatibilidade
                    for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
                        if f not in loaded_stages:
                            continue
                        for s in self.stages[f].keys():
                            if s not in loaded_stages[f]:
                                continue
                            
                            item = loaded_stages[f][s]
                            
                            # Retrocompatibilidade
                            if 'MCP' in item and 'J1_Pitch' not in item:
                                if f == 'Thumb':
                                    item['J1_Yaw'] = item.get('CMC_Yaw', -25.0)
                                    item['J1_Pitch'] = item.get('CMC_Pitch', 5.4)
                                    item['J2_Yaw'] = 0.0
                                    item['J2_Pitch'] = item.get('MCP', 10.0)
                                    item['J3_Yaw'] = 0.0
                                    item['J3_Pitch'] = item.get('PIP', 5.0)
                                else:
                                    def_y = {'Index': 5.0, 'Middle': 0.0, 'Ring': -5.0, 'Pinky': -15.0}
                                    item['J1_Yaw'] = def_y.get(f, 0.0)
                                    item['J1_Pitch'] = item.get('MCP', 5.0)
                                    item['J2_Yaw'] = 0.0
                                    item['J2_Pitch'] = item.get('PIP', 5.0)
                                    item['J3_Yaw'] = 0.0
                                    item['J3_Pitch'] = item.get('PIP', 5.0)
                                    item['J4_Yaw'] = 0.0
                                    item['J4_Pitch'] = item.get('PIP', 5.0)
                            
                            # Preenchimento inteligente de limitações embutidas caso não existam no JSON
                            if f == 'Thumb':
                                item.setdefault('J1_Yaw', 0.0)
                                item.setdefault('J1_Pitch', 0.0)
                                item.setdefault('J2_Yaw', 0.0)
                                item.setdefault('J2_Pitch', 0.0)
                                item.setdefault('J3_Yaw', 0.0)
                                item.setdefault('J3_Pitch', 0.0)
                                item.setdefault('J4_Yaw', 0.0)
                                item.setdefault('J4_Pitch', item.get('J3_Pitch', 0.0))
                            else:
                                item.setdefault('J1_Yaw', 0.0)
                                item.setdefault('J1_Pitch', 0.0)  # Limitação embutida
                                item.setdefault('J2_Yaw', 0.0)    # Limitação embutida
                                item.setdefault('J2_Pitch', 0.0)
                                item.setdefault('J3_Yaw', 0.0)    # Limitação embutida
                                item.setdefault('J3_Pitch', 0.0)
                                item.setdefault('J4_Yaw', 0.0)    # Limitação embutida
                                item.setdefault('J4_Pitch', item.get('J3_Pitch', 0.0)) # Acoplamento de tendão
                                
                            # Atualiza a memória de forma segura, preservando chaves default se faltou estágio
                            self.stages[f][s].update(item)
                
                if "avg_lengths" in saved:
                    self.avg_lengths = saved["avg_lengths"]
                if "avg_palm" in saved:
                    self.avg_palm = saved["avg_palm"]
                if "spreads" in saved:
                    self.spread_limits = saved["spreads"]
                else:
                    self.spread_limits = None
                if "thumb_fold" in saved:
                    self.thumb_fold_limits = saved["thumb_fold"]
                else:
                    self.thumb_fold_limits = None
                if "stage_raw_landmarks" in saved:
                    self.stage_raw_landmarks = saved["stage_raw_landmarks"]
                
                # Sincronizar estados da simulação com estágio ativo inicial
                for f in self.finger_states:
                    self.finger_states[f] = int(self.active_stage)
                    
                print(f"[SISTEMA] Calibração carregada de: {CALIBRATION_FILE}")
            except Exception as e:
                print(f"[Aviso] Falha ao ler calibração: {e}")

    def save_calibration_file(self):
        saved_data = {"stages": {}}
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
            except:
                pass

        if "stages" not in saved_data:
            saved_data["stages"] = {}

        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            saved_data["stages"][f] = self.stages[f]

        saved_data["avg_lengths"] = self.avg_lengths
        saved_data["avg_palm"] = self.avg_palm
        if hasattr(self, 'spread_limits') and self.spread_limits:
            saved_data["spread_limits"] = self.spread_limits
        if hasattr(self, 'thumb_fold_limits') and self.thumb_fold_limits:
            saved_data["thumb_fold_limits"] = self.thumb_fold_limits
        if hasattr(self, 'stage_raw_landmarks') and self.stage_raw_landmarks:
            saved_data["stage_raw_landmarks"] = self.stage_raw_landmarks

        try:
            os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
            with open(CALIBRATION_FILE, 'w', encoding='utf-8') as file:
                json.dump(saved_data, file, indent=4)
            print(f"[SUCESSO] Calibração salva diretamente em {CALIBRATION_FILE}")
            messagebox.showinfo("Sucesso", f"Calibração salva com sucesso em:\n{CALIBRATION_FILE}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar calibração: {e}")

    # ---------------------------------------------------------
    # CONSTRUÇÃO DA INTERFACE (TKINTER SIDEBAR + CANVAS)
    # ---------------------------------------------------------
    def build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=COLORS['bg_main'], foreground=COLORS['text_main'], font=("Segoe UI", 10))
        style.configure('TFrame', background=COLORS['bg_main'])
        style.configure('Card.TFrame', background=COLORS['bg_card'], relief='flat')
        style.configure('TLabel', background=COLORS['bg_main'], foreground=COLORS['text_main'])
        style.configure('Card.TLabel', background=COLORS['bg_card'], foreground=COLORS['text_main'])
        
        # Layout Principal: Sidebar (Esquerda) e Canvas (Direita)
        self.sidebar_frame = tk.Frame(self.root, width=420, bg=COLORS['bg_sidebar'], padx=15, pady=15)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        self.canvas_container = tk.Frame(self.root, bg=COLORS['bg_canvas'])
        self.canvas_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Cabeçalho Principal
        header_frame = tk.Frame(self.sidebar_frame, bg=COLORS['bg_sidebar'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        lbl_title = tk.Label(header_frame, text="CALIBRADOR ANATÔMICO 3D", fg=COLORS['accent_blue'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 14, "bold"))
        lbl_title.pack(anchor=tk.W)
        lbl_subtitle = tk.Label(header_frame, text="Controle 4-Vias (Combobox, Sliders, Entrada & Arrasto)", fg=COLORS['text_muted'], bg=COLORS['bg_sidebar'], font=("Segoe UI", 9))
        lbl_subtitle.pack(anchor=tk.W)

        # Container Rolar
        scroll_canvas = tk.Canvas(self.sidebar_frame, bg=COLORS['bg_sidebar'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.sidebar_frame, orient="vertical", command=scroll_canvas.yview)
        self.scrollable_frame = tk.Frame(scroll_canvas, bg=COLORS['bg_sidebar'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=380)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. SELEÇÃO DE DEDO E ESTÁGIO ATIVO
        self.build_selector_cards()

        # 2. CARD PREMIUM DE CONTROLE DE JUNTA
        self.build_joint_sliders_card()

        # 3. CONFIGURAR ABERTURAS
        self.build_spread_configurator_card()

        # 4. TESTADOR DE POSE (DADADADAFP)
        self.build_pose_tester_card()

        # 5. AÇÕES DO SISTEMA
        self.build_action_buttons()

        # ---------------------------------------------------------
        # CANVAS PRINCIPAL (DIREITA)
        # ---------------------------------------------------------
        self.canvas = tk.Canvas(self.canvas_container, bg=COLORS['bg_canvas'], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings de interação do Canvas
        self.canvas.bind("<Motion>", self.on_canvas_mouse_move)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", lambda e: self.redraw_hand())

    def build_selector_cards(self):
        card = tk.Frame(self.scrollable_frame, bg=COLORS['bg_card'], padx=10, pady=10)
        card.pack(fill=tk.X, pady=(0, 10))

        # Seleção de Dedo (Segmented Buttons)
        lbl_f = tk.Label(card, text="1. Selecione o Dedo", fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 10, "bold"))
        lbl_f.pack(anchor=tk.W, pady=(0, 6))

        finger_btn_frame = tk.Frame(card, bg=COLORS['bg_card'])
        finger_btn_frame.pack(fill=tk.X)

        self.finger_buttons = {}
        fingers = [('Polegar', 'Thumb'), ('Indicador', 'Index'), ('Médio', 'Middle'), ('Anelar', 'Ring'), ('Mindinho', 'Pinky')]
        for label, code in fingers:
            btn = tk.Button(
                finger_btn_frame, text=label, bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'],
                activebackground=COLORS['accent_blue'], activeforeground='#11111B', relief='flat', 
                font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2",
                command=lambda c=code: self.set_active_finger(c)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=2)
            self.finger_buttons[code] = btn

        # Seleção de Estágio (Segmented Buttons)
        lbl_s = tk.Label(card, text="2. Selecione o Estágio", fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 10, "bold"))
        lbl_s.pack(anchor=tk.W, pady=(8, 6))

        stage_btn_frame = tk.Frame(card, bg=COLORS['bg_card'])
        stage_btn_frame.pack(fill=tk.X)

        self.stage_buttons = {}
        for s in range(4):
            code = str(s)
            btn = tk.Button(
                stage_btn_frame, text=f"Estágio {s}", bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'],
                activebackground=COLORS['accent_green'], activeforeground='#11111B', relief='flat', 
                font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2",
                command=lambda c=code: self.set_active_stage(c)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=2)
            self.stage_buttons[code] = btn

    def build_joint_sliders_card(self):
        self.slider_card = tk.Frame(self.scrollable_frame, bg=COLORS['bg_card'], padx=10, pady=10)
        self.slider_card.pack(fill=tk.X, pady=(0, 10))

        self.lbl_joint_header = tk.Label(self.slider_card, text="Controle por Junta (Yaw & Pitch)",
                                         fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 10, "bold"))
        self.lbl_joint_header.pack(anchor=tk.W, pady=(0, 8))

        # Joint Selector Dropdown
        self.joint_selector = ttk.Combobox(
            self.slider_card, values=JOINT_OPTIONS, state="readonly", font=("Segoe UI", 9)
        )
        self.joint_selector.pack(fill=tk.X, pady=(0, 10))
        self.joint_selector.bind("<<ComboboxSelected>>", self.on_joint_combobox_changed)
        self.joint_selector.set("Indicador - MCP (Junta 1)") # Valor padrão

        # Yaw (Lateral) Frame
        self.yaw_frame = tk.Frame(self.slider_card, bg=COLORS['bg_card'])
        self.yaw_frame.pack(fill=tk.X, pady=4)
        self.lbl_yaw_title = tk.Label(self.yaw_frame, text="Yaw (Lateral):", fg=COLORS['text_muted'], bg=COLORS['bg_card'], width=15, anchor=tk.W)
        self.lbl_yaw_title.pack(side=tk.LEFT)
        self.yaw_slider = tk.Scale(
            self.yaw_frame, from_=-180, to=180, orient=tk.HORIZONTAL, bg=COLORS['bg_card'],
            fg=COLORS['text_main'], highlightthickness=0, troughcolor=COLORS['bg_sidebar'],
            activebackground=COLORS['accent_blue'], showvalue=False, command=self.on_yaw_slider_move
        )
        self.yaw_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.yaw_entry = tk.Entry(
            self.yaw_frame, width=6, bg=COLORS['bg_sidebar'], fg=COLORS['text_main'],
            insertbackground=COLORS['text_main'], relief='flat', font=("Segoe UI", 9, "bold"), justify='center'
        )
        self.yaw_entry.pack(side=tk.RIGHT)
        self.yaw_entry.bind("<Return>", self.on_yaw_entry_commit)
        self.yaw_entry.bind("<FocusOut>", self.on_yaw_entry_commit)

        # Pitch (Flexão) Frame
        self.pitch_frame = tk.Frame(self.slider_card, bg=COLORS['bg_card'])
        self.pitch_frame.pack(fill=tk.X, pady=4)
        self.lbl_pitch_title = tk.Label(self.pitch_frame, text="Pitch (Flexão):", fg=COLORS['text_muted'], bg=COLORS['bg_card'], width=15, anchor=tk.W)
        self.lbl_pitch_title.pack(side=tk.LEFT)
        self.pitch_slider = tk.Scale(
            self.pitch_frame, from_=-180, to=180, orient=tk.HORIZONTAL, bg=COLORS['bg_card'],
            fg=COLORS['text_main'], highlightthickness=0, troughcolor=COLORS['bg_sidebar'],
            activebackground=COLORS['accent_blue'], showvalue=False, command=self.on_pitch_slider_move
        )
        self.pitch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.pitch_entry = tk.Entry(
            self.pitch_frame, width=6, bg=COLORS['bg_sidebar'], fg=COLORS['text_main'],
            insertbackground=COLORS['text_main'], relief='flat', font=("Segoe UI", 9, "bold"), justify='center'
        )
        self.pitch_entry.pack(side=tk.RIGHT)
        self.pitch_entry.bind("<Return>", self.on_pitch_entry_commit)
        self.pitch_entry.bind("<FocusOut>", self.on_pitch_entry_commit)

    def build_spread_configurator_card(self):
        card = tk.Frame(self.scrollable_frame, bg=COLORS['bg_card'], padx=10, pady=10)
        card.pack(fill=tk.X, pady=(0, 10))

        lbl = tk.Label(card, text="3. Selecione a Abertura", fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 10, "bold"))
        lbl.pack(anchor=tk.W, pady=(0, 8))
        
        spread_btn_frame = tk.Frame(card, bg=COLORS['bg_card'])
        spread_btn_frame.pack(fill=tk.X)

        self.spread_buttons = {}
        spreads = [('Pol. - Ind.', 'Index_Thumb'), ('Ind - Med', 'Middle_Index'), ('Med - Ane', 'Ring_Middle'), ('Ane - Min.', 'Pinky_Ring'), ('Mov. P', 'Thumb_Opp')]
        for label, code in spreads:
            btn = tk.Button(
                spread_btn_frame, text=label, bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'],
                activebackground=COLORS['accent_yellow'], activeforeground='#11111B', relief='flat', 
                font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2",
                command=lambda c=code: self.set_active_spread(c)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=2)
            self.spread_buttons[code] = btn
            
        spread_stage_frame = tk.Frame(card, bg=COLORS['bg_card'])
        spread_stage_frame.pack(fill=tk.X, pady=(8, 4))
        
        self.spread_stage_buttons = {}
        for s, label in [(0, "Estágio 0"), (1, "Estágio 1")]:
            btn = tk.Button(
                spread_stage_frame, text=label, bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'],
                activebackground=COLORS['accent_green'], activeforeground='#11111B', relief='flat', 
                font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2",
                command=lambda stage=s: self.set_spread_stage(stage)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=2)
            self.spread_stage_buttons[s] = btn
            
        slider_frame = tk.Frame(card, bg=COLORS['bg_card'])
        slider_frame.pack(fill=tk.X, pady=(8, 0))
        
        lbl_slider = tk.Label(slider_frame, text="Valor da Abertura:", fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 8))
        lbl_slider.pack(side=tk.LEFT)
        
        self.spread_val_slider = tk.Scale(
            slider_frame, from_=-45.0, to=45.0, resolution=0.1, orient=tk.HORIZONTAL,
            bg=COLORS['bg_card'], fg=COLORS['text_main'], highlightthickness=0, bd=0, troughcolor=COLORS['bg_sidebar'],
            activebackground=COLORS['accent_yellow'], showvalue=False, command=self.on_spread_slider_move
        )
        self.spread_val_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.spread_val_entry = tk.Entry(
            slider_frame, width=6, bg=COLORS['bg_sidebar'], fg=COLORS['text_main'],
            insertbackground=COLORS['text_main'], relief='flat', font=("Segoe UI", 9, "bold"), justify='center'
        )
        self.spread_val_entry.pack(side=tk.RIGHT)
        self.spread_val_entry.bind("<Return>", self.on_spread_entry_commit)
        self.spread_val_entry.bind("<FocusOut>", self.on_spread_entry_commit)
        
        self.active_spread = None
        self.active_spread_stage = 0
        self.set_spread_stage(0)

    def set_active_spread(self, spread_code):
        self.active_spread = spread_code
        print(f"[AÇÃO] Seleção de Abertura - Abertura Ativa alterada para: {spread_code}")
        
        for code, btn in self.spread_buttons.items():
            if code == spread_code:
                btn.configure(bg=COLORS['accent_yellow'], fg='#11111B')
            else:
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
                
        self.update_spread_ui()
        
    def set_spread_stage(self, stage):
        self.active_spread_stage = stage
        for s, btn in self.spread_stage_buttons.items():
            if s == stage:
                btn.configure(bg=COLORS['accent_green'], fg='#11111B')
            else:
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
                
        self.update_spread_ui()
        
    def update_spread_ui(self):
        if not self.active_spread:
            return
            
        if not hasattr(self, 'spread_limits') or self.spread_limits is None:
            self.spread_limits = {'Pinky_Ring': [0.0,0.0], 'Ring_Middle': [0.0,0.0], 'Middle_Index': [0.0,0.0], 'Index_Thumb': [0.0,0.0]}
            
        spread_data = self.spread_limits.get(self.active_spread, [0.0, 0.0])
        val = 0.0
        if isinstance(spread_data, list) and len(spread_data) > self.active_spread_stage:
            val = float(spread_data[self.active_spread_stage])
        elif isinstance(spread_data, dict):
            val = float(spread_data.get(self.active_spread_stage, 0.0))
        
        self.updating_gui = True
        self.spread_val_slider.set(val)
        self.spread_val_entry.delete(0, tk.END)
        self.spread_val_entry.insert(0, f"{val:.1f}")
        self.updating_gui = False
        
    def on_spread_slider_move(self, val):
        if self.updating_gui or not self.active_spread:
            return
        v = float(val)
        self.spread_limits[self.active_spread][self.active_spread_stage] = v
        self.updating_gui = True
        self.spread_val_entry.delete(0, tk.END)
        self.spread_val_entry.insert(0, f"{v:.1f}")
        self.updating_gui = False
        self.redraw_hand()
        
    def on_spread_entry_commit(self, event=None):
        if not self.active_spread:
            return
        try:
            val_str = self.spread_val_entry.get().replace(',', '.')
            v = float(val_str)
            self.updating_gui = True
            self.spread_val_slider.set(v)
            self.updating_gui = False
            self.on_spread_slider_move(v)
        except ValueError:
            self.update_spread_ui()

    def build_pose_tester_card(self):
        card = tk.Frame(self.scrollable_frame, bg=COLORS['bg_card'], padx=10, pady=10)
        card.pack(fill=tk.X, pady=(0, 10))

        lbl = tk.Label(card, text="Testador DADADADAFP", fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 10, "bold"))
        lbl.pack(anchor=tk.W, pady=(0, 8))
        
        frm = tk.Frame(card, bg=COLORS['bg_card'])
        frm.pack(fill=tk.X)
        
        self.pose_code_entry = tk.Entry(
            frm, width=15, bg=COLORS['bg_sidebar'], fg=COLORS['text_main'],
            insertbackground=COLORS['text_main'], relief='flat', font=("Consolas", 10, "bold"), justify='center'
        )
        self.pose_code_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.pose_code_entry.insert(0, "0000000000")
        
        btn = tk.Button(
            frm, text="Aplicar", bg=COLORS['accent_blue'], fg='#11111B',
            activebackground='#a3c8fc', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 8, "bold"), cursor="hand2", command=self.apply_dadadadafp_code
        )
        btn.pack(side=tk.RIGHT)

    def apply_dadadadafp_code(self):
        code = self.pose_code_entry.get().strip()
        if len(code) != 10 or not code.isdigit():
            messagebox.showwarning("Aviso", "O código deve ter exatamente 10 dígitos numéricos.")
            return
            
        pinky_s, pr_spread, ring_s, rm_spread, middle_s, mi_spread, index_s, it_spread, thumb_opp, thumb_ip = [int(c) for c in code]
        
        self.finger_states['Pinky'] = pinky_s
        self.finger_states['Ring'] = ring_s
        self.finger_states['Middle'] = middle_s
        self.finger_states['Index'] = index_s
        self.finger_states['Thumb'] = 0  # MCP fixo em aberto, IP controlado por P
        
        self.thumb_ip_override = thumb_ip
        
        self.spread_overrides = {
            'Pinky_Ring': pr_spread,
            'Ring_Middle': rm_spread,
            'Middle_Index': mi_spread,
            'Index_Thumb': it_spread
        }
        self.thumb_opp_override = thumb_opp
        
        print(f"[AÇÃO] Visualizador de Pose - Código aplicado: {code}")
        self.redraw_hand()

    def build_action_buttons(self):
        frm_actions = tk.Frame(self.scrollable_frame, bg=COLORS['bg_sidebar'])
        frm_actions.pack(fill=tk.X, pady=(5, 15))

        btn_save = tk.Button(
            frm_actions, text="SALVAR CALIBRAÇÃO", bg=COLORS['accent_green'], fg='#11111B',
            activebackground='#8be087', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 10, "bold"), pady=6, cursor="hand2", command=self.save_calibration_file
        )
        btn_save.pack(fill=tk.X, pady=4)

        btn_cam = tk.Button(
            frm_actions, text="CALIBRAR COM A CÂMERA", bg=COLORS['accent_blue'], fg='#11111B',
            activebackground='#a3c8fc', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 10, "bold"), pady=6, cursor="hand2", command=self.open_camera_calibration
        )
        btn_cam.pack(fill=tk.X, pady=4)

        btn_reset = tk.Button(
            frm_actions, text="Resetar Ângulos da Câmera 3D", bg='#44445c', fg=COLORS['text_main'],
            activebackground='#585b70', activeforeground=COLORS['text_main'], relief='flat',
            font=("Segoe UI", 9), pady=4, cursor="hand2", command=self.reset_view_angles
        )
        btn_reset.pack(fill=tk.X, pady=4)

    # ---------------------------------------------------------
    # SINCRONIZAÇÃO BIDIRECIONAL DE 4 VIAS
    # ---------------------------------------------------------
    def translate_finger_name(self, finger):
        mapping = {'Thumb': 'Polegar', 'Index': 'Indicador', 'Middle': 'Médio', 'Ring': 'Anelar', 'Pinky': 'Mindinho'}
        return mapping.get(finger, finger)

    def translate_joint_name(self, finger, joint):
        if finger == 'Thumb':
            mapping = {'J1': 'CMC', 'J2': 'MCP', 'J3': 'IP'}
        else:
            mapping = {'J1': 'MCP', 'J2': 'PIP', 'J3': 'DIP'}
        return mapping.get(joint, joint)

    def set_active_finger(self, finger_code):
        self.active_finger = finger_code
        self.active_spread = None
        print(f"[AÇÃO] Seleção de Dedo - Dedo Ativo alterado para: {finger_code}")
        
        # Estilizar botões de dedo
        for code, btn in self.finger_buttons.items():
            if code == finger_code:
                btn.configure(bg=COLORS[finger_code], fg='#11111B')
            else:
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
                
        # Desmarcar botões de spread
        if hasattr(self, 'spread_buttons'):
            for code, btn in self.spread_buttons.items():
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])

        # Restaurar textos dos estágios
        for code, btn in self.stage_buttons.items():
            btn.configure(text=f"Estágio {code}")

        # Atualizar a simulação para focar no estágio atual deste dedo
        self.spread_overrides = None
        self.thumb_opp_override = None
        self.thumb_ip_override = None
        self.thumb_ip_override = None
        
        self.finger_states[finger_code] = int(self.active_stage)

        # Sincronizar o Combobox com a Junta 1 (MCP ou CMC) do novo dedo
        joint_name = "CMC" if finger_code == 'Thumb' else "MCP"
        joint_label = f"{self.translate_finger_name(finger_code)} - {joint_name} (Junta 1)"
        self.joint_selector.set(joint_label)

        self.update_gui_from_model()
        self.redraw_hand()

    def set_active_spread(self, spread_code):
        self.active_spread = spread_code
        self.active_finger = None
        print(f"[AÇÃO] Seleção de Abertura - Abertura Ativa alterada para: {spread_code}")
        
        # Estilizar botões de spread
        for code, btn in self.spread_buttons.items():
            if code == spread_code:
                btn.configure(bg=COLORS['accent_yellow'], fg='#11111B')
            else:
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
                
        # Desmarcar botões de dedo
        for code, btn in self.finger_buttons.items():
            btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
            
        # Mudar textos dos estágios para 0 (Aberto) e 1 (Fechado) e desabilitar 2/3
        for code, btn in self.stage_buttons.items():
            if code == '0':
                btn.configure(text="0 (Aberto)", state='normal')
            elif code == '1':
                btn.configure(text="1 (Fechado)", state='normal')
            else:
                btn.configure(text=f"-", state='disabled')
                
        # Se estava no 2 ou 3, forçar pro 0
        if int(self.active_stage) > 1:
            self.set_active_stage('0')
            return # set_active_stage chamará update_gui_from_model
            
        self.update_gui_from_model()
        self.redraw_hand()

    def set_active_stage(self, stage_code):
        self.active_stage = stage_code
        print(f"[AÇÃO] Seleção de Estágio - Estágio Ativo alterado para: {stage_code}")
        
        # Estilizar botões de estágio
        for code, btn in self.stage_buttons.items():
            if code == stage_code:
                btn.configure(bg=COLORS['accent_green'], fg='#11111B')
            else:
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])

        # Forçar a pose inteira a simular o estágio atual
        self.spread_overrides = None
        self.thumb_opp_override = None
        self.thumb_ip_override = None
        
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            self.finger_states[f] = int(stage_code)

        self.update_gui_from_model()
        self.redraw_hand()

    def update_gui_from_model(self):
        if self.updating_gui:
            return
        self.updating_gui = True

        selected_text = self.joint_selector.get()
        res = JOINT_MAPPING.get(selected_text)
        if res:
            finger, joint = res
            self.active_finger = finger
            self.active_joint = joint
            
            # Sincronizar o landmark correspondente
            self.active_landmark_idx = JOINT_TO_LANDMARK.get((finger, joint), 1)

            # Get angles
            state = self.active_stage
            yaw_v = self.stages[finger][state][f"{joint}_Yaw"]
            pitch_v = self.stages[finger][state][f"{joint}_Pitch"]

            # Update sliders and entries
            self.yaw_slider.set(int(yaw_v))
            self.yaw_entry.delete(0, tk.END)
            self.yaw_entry.insert(0, f"{yaw_v:.1f}")

            self.pitch_slider.set(int(pitch_v))
            self.pitch_entry.delete(0, tk.END)
            self.pitch_entry.insert(0, f"{pitch_v:.1f}")

            # Todos os controles habilitados para total controle da cinemática
            self.yaw_slider.configure(state='normal')
            self.yaw_entry.configure(state='normal')
            self.lbl_yaw_title.configure(text="Yaw (Lateral):")
                
            self.lbl_joint_header.configure(text=f"Ajuste da Junta: {selected_text}")

        self.updating_gui = False

    def on_joint_combobox_changed(self, event):
        selected_text = self.joint_selector.get()
        print(f"[AÇÃO] Dropdown Junta - Junta selecionada: {selected_text}")
        res = JOINT_MAPPING.get(selected_text)
        if res:
            finger, joint = res
            self.active_finger = finger
            self.active_joint = joint
            
            # Sincronizar botões do dedo ativo
            for code, btn in self.finger_buttons.items():
                if code == finger:
                    btn.configure(bg=COLORS[finger], fg='#11111B')
                else:
                    btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
            
            self.active_landmark_idx = JOINT_TO_LANDMARK.get((finger, joint), 1)

            self.update_gui_from_model()
            self.redraw_hand()

    def on_yaw_slider_move(self, val):
        if self.updating_gui:
            return
        selected_text = self.joint_selector.get()
        res = JOINT_MAPPING.get(selected_text)
        if not res:
            return
        finger, joint = res
        state = self.active_stage

        v = float(val)

        # Forçar limites anatômicos nos Dedos D
        if finger != 'Thumb' and joint in ['J2', 'J3', 'J4']:
            v = 0.0
        elif finger == 'Thumb' and joint == 'J4':
            v = 0.0

        self.stages[finger][state][f"{joint}_Yaw"] = v
        print(f"[SISTEMA YAW] {finger} {joint} Yaw no Estágio {state} alterado para: {v:.1f}°")

        self.updating_gui = True
        self.yaw_entry.delete(0, tk.END)
        self.yaw_entry.insert(0, f"{v:.1f}")
        self.updating_gui = False

        self.redraw_hand()

    def on_pitch_slider_move(self, val):
        if self.updating_gui:
            return
            
        v = float(val)
        state_int = int(self.active_stage)
        
        if getattr(self, 'active_spread', None):
            spread_code = self.active_spread
            if spread_code == 'Thumb_Opp':
                if state_int == 0:
                    self.thumb_fold_limits['J1_Pitch_rest'] = v
                else:
                    self.thumb_fold_limits['J1_Pitch_offset'] = v
                print(f"[SISTEMA PITCH] Movimentação Transversal do Polegar no Estado {state_int} alterada para: {v:.1f}°")
                
            self.updating_gui = True
            self.pitch_entry.delete(0, tk.END)
            self.pitch_entry.insert(0, f"{v:.1f}")
            self.updating_gui = False
            self.redraw_hand()
            return
            
        selected_text = self.joint_selector.get()
        res = JOINT_MAPPING.get(selected_text)
        if not res:
            return
        finger, joint = res
        state = self.active_stage

        v = float(val)
        
        # Forçar limites anatômicos nos Dedos D
        if finger != 'Thumb' and joint == 'J1':
            v = 0.0
        
        self.stages[finger][state][f"{joint}_Pitch"] = v

        # Acoplamento de Tendão
        if finger != 'Thumb':
            if joint == 'J3':
                self.stages[finger][state]['J4_Pitch'] = v
            elif joint == 'J4':
                self.stages[finger][state]['J3_Pitch'] = v
            
            if joint == 'J2' and finger == 'Ring':
                self.stages['Pinky'][state]['J2_Pitch'] = v
            elif joint == 'J2' and finger == 'Pinky':
                self.stages['Ring'][state]['J2_Pitch'] = v

        print(f"[SISTEMA PITCH] {finger} {joint} Pitch no Estágio {state} alterado para: {v:.1f}°")
        
        self.updating_gui = True
        self.pitch_entry.delete(0, tk.END)
        self.pitch_entry.insert(0, f"{v:.1f}")
        self.updating_gui = False

        self.redraw_hand()

    def on_yaw_entry_commit(self, event=None):
        if self.updating_gui:
            return
        selected_text = self.joint_selector.get()
        res = JOINT_MAPPING.get(selected_text)
        if not res:
            return
        finger, joint = res
        state = self.active_stage

        try:
            val = float(self.yaw_entry.get())
            val = np.clip(val, -180.0, 180.0)

            # Forçar limites anatômicos nos Dedos D
            if finger != 'Thumb' and joint in ['J2', 'J3', 'J4']:
                val = 0.0
            elif finger == 'Thumb' and joint == 'J4':
                val = 0.0

            self.stages[finger][state][f"{joint}_Yaw"] = val
            print(f"[SISTEMA INPUT] {finger} {joint} Yaw no Estágio {state} digitado: {val:.1f}°")

            self.updating_gui = True
            self.yaw_slider.set(int(val))
            self.yaw_entry.delete(0, tk.END)
            self.yaw_entry.insert(0, f"{val:.1f}")
            self.updating_gui = False

            self.redraw_hand()
        except ValueError:
            self.update_gui_from_model()

    def on_pitch_entry_commit(self, event=None):
        if self.updating_gui:
            return
        selected_text = self.joint_selector.get()
        res = JOINT_MAPPING.get(selected_text)
        if not res:
            return
        finger, joint = res
        state = self.active_stage

        try:
            val = float(self.pitch_entry.get())
            val = np.clip(val, -180.0, 180.0)

            # Forçar limites anatômicos nos Dedos D
            if finger != 'Thumb' and joint == 'J1':
                val = 0.0

            self.stages[finger][state][f"{joint}_Pitch"] = val

            # Acoplamento de Tendão
            if finger != 'Thumb':
                if joint == 'J3':
                    self.stages[finger][state]['J4_Pitch'] = val
                elif joint == 'J4':
                    self.stages[finger][state]['J3_Pitch'] = val
                
                if joint == 'J2' and finger == 'Ring':
                    self.stages['Pinky'][state]['J2_Pitch'] = val
                elif joint == 'J2' and finger == 'Pinky':
                    self.stages['Ring'][state]['J2_Pitch'] = val

            print(f"[SISTEMA INPUT] {finger} {joint} Pitch no Estágio {state} digitado: {val:.1f}°")

            self.updating_gui = True
            self.pitch_slider.set(int(val))
            self.pitch_entry.delete(0, tk.END)
            self.pitch_entry.insert(0, f"{val:.1f}")
            self.updating_gui = False

            self.redraw_hand()
        except ValueError:
            self.update_gui_from_model()

    def reset_view_angles(self):
        self.view_pitch = 340.0
        self.view_yaw = 325.0
        self.view_roll = 0.0
        print("[AÇÃO] Câmera 3D - Ângulos de visualização resetados.")
        self.redraw_hand()

    # --- Método Centralizador de Alterações Biomecânicas (Arrasto Tridimensional Isolado) ---
    def apply_local_deltas(self, finger, drag_idx, dx_local, dy_local, dz_local):
        res = LANDMARK_TO_JOINT.get(drag_idx)
        if not res:
            return
        finger, joint = res
        state = self.active_stage

        delta_yaw = dx_local * 400.0
        delta_pitch = -dy_local * 400.0 + dz_local * 300.0

        # Para cruzar a mão fechada na palma, o Polegar deve ler a rotação do osso 1-2 e não 0-1
        # (Lógica integrada na geração)
        
        y_key = f"{joint}_Yaw"
        p_key = f"{joint}_Pitch"
        
        # Forçar limites anatômicos nos Dedos D durante o drag
        if finger != 'Thumb':
            if joint == 'J1':
                delta_pitch = 0.0
            elif joint in ['J2', 'J3', 'J4']:
                delta_yaw = 0.0
        elif finger == 'Thumb':
            if joint == 'J4':
                delta_yaw = 0.0

        # Isolado e independente
        curr_yaw = self.stages[finger][state][y_key]
        curr_pitch = self.stages[finger][state][p_key]

        new_yaw = np.clip(curr_yaw + delta_yaw, -360.0, 360.0)
        new_pitch = np.clip(curr_pitch + delta_pitch, -360.0, 360.0)

        self.stages[finger][state][y_key] = new_yaw
        self.stages[finger][state][p_key] = new_pitch
        
        # Acoplamento de Tendão durante o drag
        if finger != 'Thumb':
            if joint == 'J3':
                self.stages[finger][state]['J4_Pitch'] = new_pitch
            elif joint == 'J4':
                self.stages[finger][state]['J3_Pitch'] = new_pitch
            
            if joint == 'J2' and finger == 'Ring':
                self.stages['Pinky'][state]['J2_Pitch'] = new_pitch
            elif joint == 'J2' and finger == 'Pinky':
                self.stages['Ring'][state]['J2_Pitch'] = new_pitch

        print(f"[ARRASADO] {finger} {joint}: Yaw={new_yaw:.1f}°, Pitch={new_pitch:.1f}°")

    # ---------------------------------------------------------
    # CINEMÁTICA DIRETA RECURSIVA PURA YAW & PITCH
    # ---------------------------------------------------------
    def generate_simulated_hand_3d(self):
        # Se o estágio ativo foi calibrado via câmera, priorizar os 21 pontos 3D exatos do MediaPipe (sem distorção FK)
        active_stage_str = str(getattr(self, 'active_stage', 0))
        if hasattr(self, 'stage_raw_landmarks') and active_stage_str in self.stage_raw_landmarks:
            raw_pts = np.array(self.stage_raw_landmarks[active_stage_str])
            # Retorna os 21 pontos 3D anatômicos reais capturados da webcam
            return [pt * 0.45 for pt in raw_pts]

        palm_bases = {
            'Thumb':  np.array([-0.16, 0.08, 0.0]),
            'Index':  np.array([-0.08, 0.45, 0.0]),
            'Middle': np.array([ 0.00, 0.48, 0.0]),
            'Ring':   np.array([ 0.08, 0.45, 0.0]),
            'Pinky':  np.array([ 0.16, 0.38, 0.0])
        }
        for finger in palm_bases:
            direction = palm_bases[finger] / max(np.linalg.norm(palm_bases[finger]), 1e-9)
            palm_bases[finger] = direction * self.avg_palm[finger]

        landmarks_3d = [np.array([0.0, 0.0, 0.0])]  # Wrist
        fingers_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

        f_states = self.finger_states.copy()

        for finger in fingers_order:
            state = str(f_states[finger])
            lengths = self.avg_lengths[finger]

            if finger == 'Thumb':
                thumb_lengths = {
                    (0, 0): [0.0982, 0.0758, 0.0572],
                    (0, 2): [0.0859, 0.0666, 0.0424],
                    (0, 3): [0.0873, 0.0761, 0.0546],
                    (1, 0): [0.0780, 0.0613, 0.0523],
                    (1, 2): [0.0672, 0.0672, 0.0362],
                    (1, 3): [0.0609, 0.0470, 0.0383]
                }
                if self.thumb_opp_override is not None:
                    opp_factor = float(self.thumb_opp_override)
                else:
                    opp_factor = float(state) / 3.0
                    
                p_idx = 0 if state == '0' else (2 if state == '2' else 3)
                if state == '1': p_idx = 2
                
                if getattr(self, 'thumb_ip_override', None) is not None:
                    p_idx = 3 if self.thumb_ip_override == 1 else 0

                lens_L0 = thumb_lengths[(0, p_idx)]
                lens_L1 = thumb_lengths[(1, p_idx)]
                lengths = [lerp(l0, l1, opp_factor) for l0, l1 in zip(lens_L0, lens_L1)]

            j1_y = self.stages[finger][state]['J1_Yaw']
            j1_p = self.stages[finger][state]['J1_Pitch']
            
            if finger == 'Thumb' and self.thumb_opp_override is not None:
                t_st = self.thumb_opp_override
                if hasattr(self, 'thumb_fold_limits') and self.thumb_fold_limits:
                    j1_y = self.thumb_fold_limits.get('J1_Yaw_offset', j1_y) if t_st == 1 else self.thumb_fold_limits.get('J1_Yaw_rest', j1_y)
                    j1_p = self.thumb_fold_limits.get('J1_Pitch_offset', j1_p) if t_st == 1 else self.thumb_fold_limits.get('J1_Pitch_rest', j1_p)

            j2_y = self.stages[finger][state]['J2_Yaw']
            j2_p = self.stages[finger][state]['J2_Pitch']
            j3_y = self.stages[finger][state].get('J3_Yaw', 0.0)
            j3_p = self.stages[finger][state].get('J3_Pitch', 0.0)
            j4_y = self.stages[finger][state].get('J4_Yaw', 0.0)
            j4_p = self.stages[finger][state].get('J4_Pitch', j3_p)
            
            if finger == 'Thumb' and getattr(self, 'thumb_ip_override', None) is not None:
                target_state = '3' if self.thumb_ip_override == 1 else '0'
                j3_y = self.stages[finger][target_state].get('J3_Yaw', 0.0)
                j3_p = self.stages[finger][target_state].get('J3_Pitch', 0.0)
                j4_y = self.stages[finger][target_state].get('J4_Yaw', 0.0)
                j4_p = self.stages[finger][target_state].get('J4_Pitch', j3_p)

            if finger != 'Thumb':
                if self.spread_overrides is not None:
                    ranges = {'Spread': self.spread_limits} if hasattr(self, 'spread_limits') and self.spread_limits else {
                        'Spread': {
                            'Pinky_Ring': [0.0, 20.0],
                            'Ring_Middle': [0.0, 18.0],
                            'Middle_Index': [0.0, 20.0],
                            'Index_Thumb': [2.0, 60.0]
                        }
                    }
                    mi_sp = self.spread_overrides['Middle_Index']
                    rm_sp = self.spread_overrides['Ring_Middle']
                    pr_sp = self.spread_overrides['Pinky_Ring']
                    it_sp = self.spread_overrides['Index_Thumb']

                    idx_th_ang = lerp(ranges['Spread']['Index_Thumb'][0], ranges['Spread']['Index_Thumb'][1], it_sp)
                    mi_ind_ang = lerp(ranges['Spread']['Middle_Index'][0], ranges['Spread']['Middle_Index'][1], mi_sp)
                    rg_mi_ang = lerp(ranges['Spread']['Ring_Middle'][0], ranges['Spread']['Ring_Middle'][1], rm_sp)
                    pk_rg_ang = lerp(ranges['Spread']['Pinky_Ring'][0], ranges['Spread']['Pinky_Ring'][1], pr_sp)
                else:
                    state_int = int(state) if state in ['0', '1'] else 0
                    
                    def get_sp(code):
                        if not hasattr(self, 'spread_limits') or not self.spread_limits:
                            return 0.0
                        spread_data = self.spread_limits.get(code, [0.0, 0.0])
                        
                        s_idx = getattr(self, 'active_spread_stage', 0) if getattr(self, 'active_spread', None) == code else state_int
                        
                        if isinstance(spread_data, list) and len(spread_data) > s_idx:
                            return float(spread_data[s_idx])
                        elif isinstance(spread_data, dict):
                            return float(spread_data.get(s_idx, 0.0))
                        elif isinstance(spread_data, dict) and str(s_idx) in spread_data:
                            return float(spread_data[str(s_idx)])
                        return 0.0

                    idx_th_ang = get_sp('Index_Thumb')
                    mi_ind_ang = get_sp('Middle_Index')
                    rg_mi_ang = get_sp('Ring_Middle')
                    pk_rg_ang = get_sp('Pinky_Ring')

                if finger == 'Index':
                    j1_y += mi_ind_ang * 0.5
                    j1_y += idx_th_ang * 0.5
                elif finger == 'Middle':
                    j1_y -= mi_ind_ang * 0.5
                    j1_y += rg_mi_ang * 0.5
                elif finger == 'Ring':
                    j1_y -= rg_mi_ang * 0.5
                    j1_y += pk_rg_ang * 0.5
                elif finger == 'Pinky':
                    j1_y -= pk_rg_ang * 0.5

            if finger == 'Thumb':
                v = palm_bases['Thumb']
                L_palm = np.linalg.norm(v)
                yaw_base = math.degrees(math.atan2(-v[0], v[1]))
                pitch_base = math.degrees(math.atan2(-v[2], math.hypot(v[0], v[1])))
                R_base = rot_z(yaw_base).dot(rot_x(pitch_base))

                # J1 (CMC) controla rotação base e oposição 0-1
                R_palm = R_base.dot(rot_z(j1_y).dot(rot_x(j1_p)))
                p1 = R_palm.dot(np.array([0.0, L_palm, 0.0]))
                
                # J2 (MCP): Dobra anatômica do polegar flexiona para dentro em direção à palma/indicador e levemente para frente
                rot_thumb_j2 = rot_z(-j2_p * 0.75).dot(rot_x(j2_p * 0.35)).dot(rot_z(j2_y))
                R1 = R_palm.dot(rot_thumb_j2)
                p2 = p1 + R1.dot(np.array([0.0, lengths[0], 0.0]))
                
                # J3 (IP): Ponta do polegar flexiona para dentro em direção à palma
                rot_thumb_j3 = rot_z(-j3_p * 0.75).dot(rot_x(j3_p * 0.35)).dot(rot_z(j3_y))
                R2 = R1.dot(rot_thumb_j3)
                p3 = p2 + R2.dot(np.array([0.0, lengths[1], 0.0]))

                # J4 (Tip)
                rot_thumb_j4 = rot_z(-j4_p * 0.75).dot(rot_x(j4_p * 0.25)).dot(rot_z(j4_y))
                R3 = R2.dot(rot_thumb_j4)
                p4 = p3 + R3.dot(np.array([0.0, lengths[2], 0.0]))
            else:
                v = palm_bases[finger]
                yaw_base = math.degrees(math.atan2(-v[0], v[1]))
                pitch_base = math.degrees(math.atan2(-v[2], math.hypot(v[0], v[1])))
                R_base = rot_z(yaw_base).dot(rot_x(pitch_base))
                
                # J1 controls 0-5
                R_palm = R_base.dot(rot_z(j1_y).dot(rot_x(j1_p)))
                p1 = R_palm.dot(np.array([0.0, self.avg_palm[finger], 0.0]))
                
                # J2 controls 5-6
                R1 = R_palm.dot(rot_z(j2_y).dot(rot_x(j2_p)))
                p2 = p1 + R1.dot(np.array([0.0, lengths[0], 0.0]))
                
                # J3 controls 6-7
                R2 = R1.dot(rot_z(j3_y).dot(rot_x(j3_p)))
                p3 = p2 + R2.dot(np.array([0.0, lengths[1], 0.0]))
                
                # J4 controls 7-8
                R3 = R2.dot(rot_z(j4_y).dot(rot_x(j4_p)))
                p4 = p3 + R3.dot(np.array([0.0, lengths[2], 0.0]))

            chain = [p1, p2, p3, p4]
            landmarks_3d.extend(chain)

        return landmarks_3d

    def project_and_render_3d(self, lms_3d, width, height):
        pts = np.array(lms_3d)
        pts = pts - pts[0]

        Rx = rot_x(self.view_pitch)
        Ry = rot_y(self.view_yaw)
        Rz = rot_z(self.view_roll)
        R = Rz.dot(Ry).dot(Rx)

        pts_2d = []
        for pt in pts:
            pt_rot = R.dot(pt)
            z_offset = 6.0
            z_factor = z_offset / (z_offset - pt_rot[2])
            x2d = pt_rot[0] * z_factor
            y2d = pt_rot[1] * z_factor
            pts_2d.append([x2d, y2d])

        pts_2d = np.array(pts_2d)
        xs = pts_2d[:, 0]
        ys = pts_2d[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        size = max(max_x - min_x, max_y - min_y, 1e-6)

        norm_pts = []
        for x, y in pts_2d:
            nx = int(width * (0.2 + 0.6 * (x - min_x) / size))
            ny = int(height * (0.25 + 0.6 * (y - min_y) / size))
            norm_pts.append((nx, ny))

        self.lms_2d = norm_pts
        return norm_pts

    def redraw_hand(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        self.canvas.delete("all")

        # Gerar e Projetar Mão
        lms_3d = self.generate_simulated_hand_3d()
        lms_2d = self.project_and_render_3d(lms_3d, w, h)

        # 1. Desenhar Esqueleto
        for start, end in CONNECTIONS:
            color = self.get_connection_color(start, end)
            self.canvas.create_line(
                lms_2d[start][0], lms_2d[start][1], lms_2d[end][0], lms_2d[end][1],
                fill=color, width=4, capstyle='round', joinstyle='round'
            )

        # Labels de Abertura (A) nas retas escuras da base (5-9, 9-13, 13-17)
        for start, end in [(5, 9), (9, 13), (13, 17)]:
            if start < len(lms_3d) and end < len(lms_3d):
                p_start = np.array(lms_3d[start])
                p_end = np.array(lms_3d[end])
                p_wrist = np.array(lms_3d[0])
                v1 = p_start - p_wrist
                v2 = p_end - p_wrist
                ang = vec_angle(v1, v2)

                p2d_mid = ((lms_2d[start][0] + lms_2d[end][0]) / 2, (lms_2d[start][1] + lms_2d[end][1]) / 2)
                self.canvas.create_text(p2d_mid[0], p2d_mid[1] - 12, text=f"{ang:.1f}°", fill="#b0bec5", font=("Segoe UI", 8, "bold"))

        # Guia visual pontilhada se estiver arrastando
        if self.mouse_drag_idx is not None and self.mouse_drag_idx in range(1, 21):
            drag_idx = self.mouse_drag_idx
            base_pt = None
            if drag_idx in [1, 2, 3, 4]: base_pt = lms_2d[1]
            elif drag_idx in [5, 6, 7, 8]: base_pt = lms_2d[5]
            elif drag_idx in [9, 10, 11, 12]: base_pt = lms_2d[9]
            elif drag_idx in [13, 14, 15, 16]: base_pt = lms_2d[13]
            elif drag_idx in [17, 18, 19, 20]: base_pt = lms_2d[17]

            if base_pt is not None:
                self.canvas.create_line(
                    base_pt[0], base_pt[1], lms_2d[drag_idx][0], lms_2d[drag_idx][1],
                    fill=COLORS['accent_green'], width=1.5, dash=(4, 4)
                )

        # 2. Desenhar Landmarks (Pontos) Interativos
        for idx, pt in enumerate(lms_2d):
            is_hover = (self.mouse_hover_idx == idx)
            is_drag = (self.mouse_drag_idx == idx)

            finger_name = self.get_finger_name_by_idx(idx)
            is_active_finger = (finger_name == self.active_finger)

            if idx == 0:
                color = '#7287FD'  # Wrist
                radius = 8
            elif is_drag:
                color = COLORS['accent_green']
                radius = 10
            elif is_hover:
                color = COLORS['accent_blue']
                radius = 10
            elif is_active_finger:
                color = COLORS[finger_name]
                radius = 7
            else:
                color = '#E6E9EF'
                radius = 5

            # Efeito glow
            if is_drag:
                self.canvas.create_oval(pt[0]-16, pt[1]-16, pt[0]+16, pt[1]+16, outline=COLORS['accent_green'], width=2)
            elif is_hover and idx != 0:
                self.canvas.create_oval(pt[0]-14, pt[1]-14, pt[0]+14, pt[1]+14, outline=COLORS['accent_blue'], width=1.5)

            # Círculo central
            self.canvas.create_oval(
                pt[0]-radius, pt[1]-radius, pt[0]+radius, pt[1]+radius,
                fill=color, outline='#11111B', width=1.5
            )

            # Legenda numérica discreta
            txt_color = COLORS['accent_blue'] if (is_hover or is_drag) else COLORS['text_muted']
            self.canvas.create_text(
                pt[0]+12, pt[1], text=str(idx), anchor=tk.W,
                fill=txt_color, font=("Segoe UI", 8, "bold")
            )

        # 3. Painel Flutuante de Feedback (Superior Direito)
        self.canvas.create_rectangle(w-280, 20, w-20, 105, fill=COLORS['bg_card'], outline=COLORS['Palm'], width=1)
        self.canvas.create_text(w-150, 35, text="ESTADO DA SIMULAÇÃO", fill=COLORS['accent_blue'], font=("Segoe UI", 9, "bold"))
        self.canvas.create_text(w-260, 60, text=f"Dedo Ativo: {self.translate_finger_name(self.active_finger)}", fill=COLORS['text_main'], anchor=tk.W, font=("Segoe UI", 9))
        self.canvas.create_text(w-260, 76, text=f"Estágio Calibração: {self.active_stage}", fill=COLORS['text_main'], anchor=tk.W, font=("Segoe UI", 9))
        yaw_disp = int(self.view_yaw) % 360
        pitch_disp = int(self.view_pitch) % 360
        self.canvas.create_text(w-260, 92, text=f"Rot: Yaw {yaw_disp}° / Pitch {pitch_disp}°", fill=COLORS['text_muted'], anchor=tk.W, font=("Segoe UI", 8))

        # Barra de Status inferior
        self.canvas.create_rectangle(0, h-35, w, h, fill=COLORS['bg_sidebar'], outline="")
        self.canvas.create_text(
            20, h-18, text="CONTROLES: Arraste o fundo do canvas para rotacionar a câmera. Clique e arraste os pontos livremente para posear as juntas.",
            fill=COLORS['text_muted'], anchor=tk.W, font=("Segoe UI", 9)
        )

    def get_connection_color(self, start, end):
        if start in [0,1,2,3] and end in [0,1,2,3,4]: return COLORS['Thumb']
        elif start in [0,5,6,7] and end in [0,5,6,7,8]: return COLORS['Index']
        elif start in [0,9,10,11] and end in [0,9,10,11,12]: return COLORS['Middle']
        elif start in [0,13,14,15] and end in [0,13,14,15,16]: return COLORS['Ring']
        elif start in [0,17,18,19] and end in [0,17,18,19,20]: return COLORS['Pinky']
        return COLORS['Palm']

    def get_finger_name_by_idx(self, idx):
        if idx in [1, 2, 3, 4]: return 'Thumb'
        elif idx in [5, 6, 7, 8]: return 'Index'
        elif idx in [9, 10, 11, 12]: return 'Middle'
        elif idx in [13, 14, 15, 16]: return 'Ring'
        elif idx in [17, 18, 19, 20]: return 'Pinky'
        return None

    # ---------------------------------------------------------
    # INTERAÇÕES DO MOUSE
    # ---------------------------------------------------------
    def on_canvas_mouse_move(self, event):
        x, y = event.x, event.y
        old_hover = self.mouse_hover_idx

        self.mouse_hover_idx = None
        if len(self.lms_2d) > 0:
            for idx, pt in enumerate(self.lms_2d):
                if math.hypot(x - pt[0], y - pt[1]) < 14:
                    self.mouse_hover_idx = idx
                    break

        if self.mouse_hover_idx is not None and self.mouse_hover_idx != 0:
            self.canvas.config(cursor="hand2")
        else:
            self.canvas.config(cursor="arrow")

        if old_hover != self.mouse_hover_idx:
            if self.mouse_hover_idx is not None:
                finger_name = self.get_finger_name_by_idx(self.mouse_hover_idx) or "Wrist"
                print(f"[INTERAÇÃO] Hover - Landmark {self.mouse_hover_idx} ({finger_name})")
            self.redraw_hand()

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        self.last_mouse_x = x
        self.last_mouse_y = y
        if self.mouse_hover_idx is not None and self.mouse_hover_idx != 0:
            self.mouse_drag_idx = self.mouse_hover_idx
            self.active_landmark_idx = self.mouse_drag_idx
            
            # Foco imediato na junta correspondente ao ponto clicado!
            res = LANDMARK_TO_JOINT.get(self.active_landmark_idx)
            if res:
                finger, joint = res
                self.active_finger = finger
                self.active_joint = joint
                
                # Atualizar dropdown Combobox
                joint_label = f"{self.translate_finger_name(finger)} - {self.translate_joint_name(finger, joint)} (Junta {joint[1]})"
                self.joint_selector.set(joint_label)
                
                print(f"[INTERAÇÃO] Ponto selecionado - Landmark {self.mouse_drag_idx} ({finger} {joint})")
                
                # Sincronizar botões de dedo
                for code, btn in self.finger_buttons.items():
                    if code == finger:
                        btn.configure(bg=COLORS[finger], fg='#11111B')
                    else:
                        btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])
            
            self.update_gui_from_model()
        else:
            self.camera_drag = True
            print(f"[INTERAÇÃO] Início de Rotação da Câmera 3D via Drag no Fundo do Canvas")

        self.redraw_hand()

    def on_canvas_drag(self, event):
        x, y = event.x, event.y

        if self.mouse_drag_idx is not None:
            drag_idx = self.mouse_drag_idx
            finger = self.get_finger_name_by_idx(drag_idx)
            if not finger:
                self.last_mouse_x = x
                self.last_mouse_y = y
                return

            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w < 10 or h < 10:
                return

            # Obter matriz de câmera
            Rx = rot_x(self.view_pitch)
            Ry = rot_y(self.view_yaw)
            Rz = rot_z(self.view_roll)
            R = Rz.dot(Ry).dot(Rx)
            Rt = R.T

            lms_3d = self.generate_simulated_hand_3d()
            pts = np.array(lms_3d) - lms_3d[0]
            pts_rot = [R.dot(pt) for pt in pts]
            pts_2d = []
            z_offset = 3.5
            for pt_rot in pts_rot:
                z_factor = z_offset / (z_offset - pt_rot[2])
                pts_2d.append([pt_rot[0] * z_factor, pt_rot[1] * z_factor])
            pts_2d = np.array(pts_2d)
            xs = pts_2d[:, 0]
            ys = pts_2d[:, 1]
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            size = max(max_x - min_x, max_y - min_y, 1e-6)

            # Converter delta da tela para delta local 3D da mão
            dx_proj = (x - self.last_mouse_x) * size / (0.6 * w)
            dy_proj = -(y - self.last_mouse_y) * size / (0.6 * h)

            d_cam = np.array([dx_proj, dy_proj, 0.0])
            d_local = Rt.dot(d_cam)
            dx_local = d_local[0]
            dy_local = d_local[1]
            dz_local = d_local[2]

            self.apply_local_deltas(finger, drag_idx, dx_local, dy_local, dz_local)

            self.update_gui_from_model()
            self.redraw_hand()

            self.last_mouse_x = x
            self.last_mouse_y = y

        elif self.camera_drag:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            self.view_yaw = (self.view_yaw + dx * 0.4) % 360.0
            self.view_pitch = (self.view_pitch + dy * 0.4) % 360.0
            
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.redraw_hand()

    def on_canvas_release(self, event):
        self.mouse_drag_idx = None
        self.camera_drag = False
        self.redraw_hand()

    # ---------------------------------------------------------
    # INTEGRAÇÃO DE CÂMERA REAL-TIME COM MEDIAPIPE (POPUP WINDOW)
    # ---------------------------------------------------------
    def open_camera_calibration(self):
        if self.camera_active:
            return
        self.camera_active = True

        self.cam_win = tk.Toplevel(self.root)
        self.cam_win.title("Calibração Real-Time via Câmera (MediaPipe)")
        self.cam_win.geometry("800x650")
        self.cam_win.configure(bg=COLORS['bg_main'])
        
        self.cam_win.transient(self.root)
        self.cam_win.grab_set()

        lbl_top = tk.Label(self.cam_win, text="CALIBRAÇÃO REAL-TIME VIA CÂMERA", fg=COLORS['accent_blue'], bg=COLORS['bg_main'], font=("Segoe UI", 12, "bold"))
        lbl_top.pack(pady=(10, 5))

        self.video_label = tk.Label(self.cam_win, bg=COLORS['bg_canvas'])
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        lbl_info = tk.Label(self.cam_win, text="[ESPAÇO] Adota limites observados da mão e salva | [ESC / C] Fechar Webcam",
                             fg=COLORS['text_muted'], bg=COLORS['bg_main'], font=("Segoe UI", 9, "bold"))
        lbl_info.pack(pady=10)

        # Abrir Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a webcam.")
            self.camera_active = False
            self.cam_win.destroy()
            return

        self.reset_live_ranges()

        # Configurar atalhos
        self.cam_win.bind("<space>", lambda e: self.open_save_calibration_dialog())
        self.cam_win.bind("<Escape>", lambda e: self.close_camera())
        self.cam_win.bind("c", lambda e: self.close_camera())
        self.cam_win.bind("C", lambda e: self.close_camera())
        self.cam_win.protocol("WM_DELETE_WINDOW", self.close_camera)

        self.root.after(10, self.update_camera_frame)

    def close_camera(self, event=None):
        self.camera_active = False
        self.calibration_frozen = False
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'cam_win') and self.cam_win is not None:
            self.cam_win.destroy()
            self.cam_win = None
        print("[CAMERA] Câmera fechada.")

    def reset_live_ranges(self):
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        self.live_lengths = {}
        self.live_palm = {}
        self.live_ranges = {}
        self.live_angles = {}
        self.live_spreads = {}
        for f in fingers:
            self.live_ranges[f] = {
                'MCP': [999.0, -999.0],
                'PIP': [999.0, -999.0]
            }

    def update_live_joint(self, finger, joint, val):
        self.live_ranges[finger][joint][0] = min(self.live_ranges[finger][joint][0], val)
        self.live_ranges[finger][joint][1] = max(self.live_ranges[finger][joint][1], val)

    def update_camera_frame(self):
        if not self.camera_active or not hasattr(self, 'cap') or self.cap is None:
            return

        if self.calibration_frozen and hasattr(self, 'frozen_frame_imgtk'):
            self.video_label.img_tk = self.frozen_frame_imgtk
            self.video_label.configure(image=self.frozen_frame_imgtk)
            self.root.after(30, self.update_camera_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_camera_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)

        cv2.rectangle(frame, (0, 0), (280, h), (30, 30, 40), -1)
        if self.calibration_frozen:
            cv2.putText(frame, "PONTOS CONGELADOS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
        else:
            cv2.putText(frame, "VALORES MEDIDOS:", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms.landmark])

            # Armazenar 21 pontos 3D reais do MediaPipe normalizados no pulso
            pts_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            w_raw = pts_raw[0]
            p_len_raw = np.linalg.norm(pts_raw[9] - w_raw)
            if p_len_raw > 1e-6:
                self.live_raw_landmarks_3d = (pts_raw - w_raw) / p_len_raw

            # Calcular tamanhos fisicos proporcionais da mão (comprimento dos ossos)
            palm_len = np.linalg.norm(pts[9] - pts[0])
            scale = 0.245 / palm_len if palm_len > 1e-6 else 1.0
            
            bone_idxs = {
                'Thumb': [1, 2, 3, 4],
                'Index': [5, 6, 7, 8],
                'Middle': [9, 10, 11, 12],
                'Ring': [13, 14, 15, 16],
                'Pinky': [17, 18, 19, 20]
            }
            palm_idxs = {'Thumb': 1, 'Index': 5, 'Middle': 9, 'Ring': 13, 'Pinky': 17}

            # Configurar Eixo Local para extrair Yaw (Spread) real
            wrist = pts[0]
            y_axis = (pts[5] + pts[17]) / 2.0 - wrist
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            z_axis = np.cross(pts[17] - wrist, pts[5] - wrist)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

            palm_bases = {
                'Thumb':  np.array([-0.16, 0.08, 0.0]),
                'Index':  np.array([-0.08, 0.45, 0.0]),
                'Middle': np.array([ 0.00, 0.48, 0.0]),
                'Ring':   np.array([ 0.08, 0.45, 0.0]),
                'Pinky':  np.array([ 0.16, 0.38, 0.0])
            }

            if not hasattr(self, 'exact_live_lengths'):
                self.exact_live_lengths = {}
                self.exact_live_palm = {}
                self.exact_live_angles = {}

            current_yaw_list = {}

            for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
                idxs = bone_idxs[f]
                
                # Extração EXATA instantânea (sem suavização)
                l0 = np.clip(np.linalg.norm(pts[idxs[1]] - pts[idxs[0]]) * scale, 0.02, 0.15)
                l1 = np.clip(np.linalg.norm(pts[idxs[2]] - pts[idxs[1]]) * scale, 0.02, 0.12)
                l2 = np.clip(np.linalg.norm(pts[idxs[3]] - pts[idxs[2]]) * scale, 0.01, 0.09)
                p_len = np.clip(np.linalg.norm(pts[palm_idxs[f]] - pts[0]) * scale, 0.05, 0.3)
                
                self.exact_live_lengths[f] = [l0, l1, l2]
                self.exact_live_palm[f] = p_len
                
                if f == 'Thumb':
                    # Extração do J1 (Osso 0-1)
                    v1 = pts[1] - wrist
                    vx1 = np.dot(v1, x_axis)
                    vy1 = np.dot(v1, y_axis)
                    yaw_obs1 = math.degrees(math.atan2(-vx1, vy1))
                    v_base = palm_bases['Thumb']
                    yaw_base = math.degrees(math.atan2(-v_base[0], v_base[1]))
                    j1_yaw = yaw_obs1 - yaw_base

                    # Extração do J2 (Osso 1-2)
                    v2 = pts[2] - pts[1]
                    vx2 = np.dot(v2, x_axis)
                    vy2 = np.dot(v2, y_axis)
                    yaw_obs2 = math.degrees(math.atan2(-vx2, vy2))
                    j2_yaw = yaw_obs2 - yaw_obs1
                else:
                    v = pts[palm_idxs[f]] - wrist
                    vx = np.dot(v, x_axis)
                    vy = np.dot(v, y_axis)
                    yaw_observed = math.degrees(math.atan2(-vx, vy))
                    v_base = palm_bases[f]
                    yaw_base = math.degrees(math.atan2(-v_base[0], v_base[1]))
                    j1_yaw = yaw_observed - yaw_base
                    j2_yaw = 0.0

                current_yaw_list[f] = j1_yaw

                # Extração de Pitches precisos
                if f == 'Thumb':
                    j1_pitch = 0.0
                    j2_pitch = joint_flexion(wrist, pts[1], pts[2])
                    j3_pitch = joint_flexion(pts[1], pts[2], pts[3])
                    j4_pitch = joint_flexion(pts[2], pts[3], pts[4])
                else:
                    j1_pitch = 0.0 # O metacarpo usa Yaw apenas
                    j2_pitch = joint_flexion(wrist, pts[idxs[0]], pts[idxs[1]])
                    j3_pitch = joint_flexion(pts[idxs[0]], pts[idxs[1]], pts[idxs[2]])
                    j4_pitch = j3_pitch # Tendão acoplado (na vida real dip-tip dobra com pip-dip)

                self.exact_live_angles[f] = {
                    'J1_Yaw': j1_yaw,
                    'J1_Pitch': j1_pitch,
                    'J2_Yaw': j2_yaw,
                    'J2_Pitch': j2_pitch,
                    'J3_Pitch': j3_pitch,
                    'J4_Pitch': j4_pitch
                }
                
                if not self.calibration_frozen:
                    self.live_angles[f] = {'MCP': j1_pitch, 'PIP': j2_pitch}
                    self.update_live_joint(f, 'MCP', j1_pitch)
                    self.update_live_joint(f, 'PIP', j2_pitch)

            # Calcular Spreads atuais
            self.live_spreads['Index_Thumb'] = abs(current_yaw_list['Index'] - current_yaw_list['Thumb'])
            self.live_spreads['Middle_Index'] = abs(current_yaw_list['Middle'] - current_yaw_list['Index'])
            self.live_spreads['Ring_Middle'] = abs(current_yaw_list['Ring'] - current_yaw_list['Middle'])
            self.live_spreads['Pinky_Ring'] = abs(current_yaw_list['Pinky'] - current_yaw_list['Ring'])

            y_pos = 65
            for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
                mcp_f = self.exact_live_angles[f]['J1_Pitch']
                pip_f = self.exact_live_angles[f]['J2_Pitch']
                
                bgr_color = {
                    'Thumb': (255, 255, 255),
                    'Index': (247, 166, 203),
                    'Middle': (175, 226, 249),
                    'Ring': (161, 227, 166),
                    'Pinky': (250, 180, 137)
                }[f]
                
                cv2.putText(frame, f"{self.translate_finger_name(f)}:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1)
                cv2.putText(frame, f" MCP: {mcp_f:.0f} (Obs: {self.live_ranges[f]['MCP'][0]:.0f}-{self.live_ranges[f]['MCP'][1]:.0f})", (15, y_pos+14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1)
                cv2.putText(frame, f" PIP: {pip_f:.0f} (Obs: {self.live_ranges[f]['PIP'][0]:.0f}-{self.live_ranges[f]['PIP'][1]:.0f})", (15, y_pos+26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1)
                y_pos += 44
        else:
            cv2.putText(frame, "Aguardando mao...", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((770, 520), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.video_label.img_tk = img_tk
        self.video_label.configure(image=img_tk)
        
        self.last_img_tk = img_tk

        self.root.after(10, self.update_camera_frame)

    def open_save_calibration_dialog(self):
        if self.calibration_frozen:
            return
        self.calibration_frozen = True
        if hasattr(self, 'last_img_tk'):
            self.frozen_frame_imgtk = self.last_img_tk

        # Janela modal para configurar o salvamento da calibração via câmera
        dialog = tk.Toplevel(self.cam_win)
        dialog.title("Salvar Calibração Customizada")
        dialog.geometry("900x550")
        dialog.configure(bg=COLORS['bg_main'])
        dialog.transient(self.cam_win)
        dialog.grab_set()

        # Comportamento customizado para descongelar a camera ao fechar a janela
        def on_dialog_close():
            self.calibration_frozen = False
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        # Centralizar diálogo com relação à janela da câmera
        dialog.update_idletasks()
        cw_w = self.cam_win.winfo_width()
        cw_h = self.cam_win.winfo_height()
        cw_x = self.cam_win.winfo_rootx()
        cw_y = self.cam_win.winfo_rooty()
        d_w = dialog.winfo_width()
        d_h = dialog.winfo_height()
        x = cw_x + (cw_w - d_w) // 2
        y = cw_y + (cw_h - d_h) // 2
        dialog.geometry(f"+{x}+{y}")

        # Título
        lbl_title = tk.Label(
            dialog, text="SALVAR CONFIGURAÇÃO DA CÂMERA",
            fg=COLORS['accent_blue'], bg=COLORS['bg_main'],
            font=("Segoe UI", 12, "bold")
        )
        lbl_title.pack(pady=(15, 15))

        # UI de Seleção de Estado (Spread e Fold)
        settings_frame = tk.Frame(dialog, bg=COLORS['bg_card'], padx=10, pady=10)
        settings_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        self.spread_state_var = tk.StringVar(value="1 - Máximo (Aberto)")
        tk.Label(settings_frame, text="Mapear Spread como:", bg=COLORS['bg_card']).pack(side=tk.LEFT)
        ttk.Combobox(settings_frame, textvariable=self.spread_state_var, values=["0 - Mínimo (Fechado)", "1 - Máximo (Aberto)"], state="readonly").pack(side=tk.LEFT, padx=10)

        self.fold_state_var = tk.StringVar(value="0 - Oposição (Offset)")
        tk.Label(settings_frame, text="Mapear Fold como:", bg=COLORS['bg_card']).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(settings_frame, textvariable=self.fold_state_var, values=["0 - Oposição (Offset)", "1 - Repouso (Rest)"], state="readonly").pack(side=tk.LEFT, padx=10)

        # Container principal Lado a Lado
        container = tk.Frame(dialog, bg=COLORS['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20)

        # Coluna Esquerda: Dedos
        left_col = tk.Frame(container, bg=COLORS['bg_main'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        lbl_fingers = tk.Label(
            left_col, text="1. Dedos a Salvar:",
            fg=COLORS['text_main'], bg=COLORS['bg_main'],
            font=("Segoe UI", 10, "bold")
        )
        lbl_fingers.pack(anchor=tk.W, pady=(0, 5))

        fingers_card = tk.Frame(left_col, bg=COLORS['bg_card'], padx=15, pady=10)
        fingers_card.pack(fill=tk.BOTH, expand=True)

        finger_vars = {}
        fingers_list = [
            ('Thumb', 'Polegar'),
            ('Index', 'Indicador'),
            ('Middle', 'Médio'),
            ('Ring', 'Anelar'),
            ('Pinky', 'Mindinho')
        ]
        
        for code, label in fingers_list:
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(
                fingers_card, text=label, variable=var,
                bg=COLORS['bg_card'], fg=COLORS['text_main'],
                activebackground=COLORS['bg_card'], activeforeground=COLORS['text_main'],
                selectcolor=COLORS['bg_sidebar'], font=("Segoe UI", 9)
            )
            chk.pack(anchor=tk.W, pady=2)
            finger_vars[code] = var

        # Coluna Central: Estágios
        right_col = tk.Frame(container, bg=COLORS['bg_main'])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 10))

        lbl_stages = tk.Label(
            right_col, text="2. Estágios a Salvar:",
            fg=COLORS['text_main'], bg=COLORS['bg_main'],
            font=("Segoe UI", 10, "bold")
        )
        lbl_stages.pack(anchor=tk.W, pady=(0, 5))

        stages_card = tk.Frame(right_col, bg=COLORS['bg_card'], padx=15, pady=10)
        stages_card.pack(fill=tk.BOTH, expand=True)

        stage_vars = {}
        stages_list = [
            ('0', 'Estágio 0 (Aberto)'),
            ('1', 'Estágio 1 (Garra Leve)'),
            ('2', 'Estágio 2 (Plataforma)'),
            ('3', 'Estágio 3 (Soco / Fechado)')
        ]

        for code, label in stages_list:
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(
                stages_card, text=label, variable=var,
                bg=COLORS['bg_card'], fg=COLORS['text_main'],
                activebackground=COLORS['bg_card'], activeforeground=COLORS['text_main'],
                selectcolor=COLORS['bg_sidebar'], font=("Segoe UI", 9)
            )
            chk.pack(anchor=tk.W, pady=2)
            stage_vars[code] = var

        # Coluna Direita: Extras (Spreads e Fold)
        extra_col = tk.Frame(container, bg=COLORS['bg_main'])
        extra_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 0))

        lbl_extras = tk.Label(
            extra_col, text="3. Extras (A/F):",
            fg=COLORS['text_main'], bg=COLORS['bg_main'],
            font=("Segoe UI", 10, "bold")
        )
        lbl_extras.pack(anchor=tk.W, pady=(0, 5))

        extras_card = tk.Frame(extra_col, bg=COLORS['bg_card'], padx=15, pady=10)
        extras_card.pack(fill=tk.BOTH, expand=True)

        self.spread_vars = {}
        spreads_list = ['Polegar-Indicador', 'Indicador-Médio', 'Médio-Anelar', 'Anelar-Mindinho']

        for label in spreads_list:
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(
                extras_card, text=label, variable=var,
                bg=COLORS['bg_card'], fg=COLORS['text_main'],
                activebackground=COLORS['bg_card'], activeforeground=COLORS['text_main'],
                selectcolor=COLORS['bg_sidebar'], font=("Segoe UI", 9)
            )
            chk.pack(anchor=tk.W, pady=2)
            self.spread_vars[label] = var
            
        self.fold_var = tk.BooleanVar(value=True)
        chk_fold = tk.Checkbutton(
            extras_card, text="Thumb Fold (Oposição)", variable=self.fold_var,
            bg=COLORS['bg_card'], fg=COLORS['text_main'],
            activebackground=COLORS['bg_card'], activeforeground=COLORS['text_main'],
            selectcolor=COLORS['bg_sidebar'], font=("Segoe UI", 9)
        )
        chk_fold.pack(anchor=tk.W, pady=(15, 2))

        # Botões de Ação
        btn_actions = tk.Frame(dialog, bg=COLORS['bg_main'])
        btn_actions.pack(fill=tk.X, padx=20, pady=(15, 20))

        def confirm_and_apply():
            selected_fingers = [code for code, var in finger_vars.items() if var.get()]
            selected_stages = [code for code, var in stage_vars.items() if var.get()]

            self.apply_live_camera_calibration_filtered(selected_fingers, selected_stages)
            self.update_gui_from_model()
            on_dialog_close()

        btn_confirm = tk.Button(
            btn_actions, text="SALVAR CALIBRAÇÃO", bg=COLORS['accent_green'], fg='#11111B',
            activebackground='#8be087', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 10, "bold"), pady=8, cursor="hand2",
            command=confirm_and_apply
        )
        btn_confirm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_cancel = tk.Button(
            btn_actions, text="CANCELAR", bg=COLORS['accent_red'], fg='#11111B',
            activebackground='#fc9d9d', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 10, "bold"), pady=8, cursor="hand2",
            command=on_dialog_close
        )
        btn_cancel.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    def apply_live_camera_calibration_filtered(self, selected_fingers, selected_stages):
        applied_fingers = []
        for f in selected_fingers:
            if not hasattr(self, 'exact_live_angles') or f not in self.exact_live_angles:
                continue

            angles = self.exact_live_angles[f]
            applied_fingers.append(self.translate_finger_name(f))

            for stage in selected_stages:
                self.stages[f][stage]['J1_Yaw'] = angles['J1_Yaw']
                self.stages[f][stage]['J1_Pitch'] = angles['J1_Pitch']
                self.stages[f][stage]['J2_Pitch'] = angles['J2_Pitch']
                self.stages[f][stage]['J3_Pitch'] = angles['J3_Pitch']
                self.stages[f][stage]['J4_Pitch'] = angles['J4_Pitch']
                
                # Força a zerar torções laterais nas falanges que não existem na vida real
                # (limpa a bagunça se o usuário tiver rotacionado no 3D manualmente)
                self.stages[f][stage]['J2_Yaw'] = 0.0
                self.stages[f][stage]['J3_Yaw'] = 0.0
                self.stages[f][stage]['J4_Yaw'] = 0.0

        # Armazenar os 21 pontos 3D exatos do MediaPipe diretamente no estágio
        if hasattr(self, 'live_raw_landmarks_3d') and self.live_raw_landmarks_3d is not None:
            if not hasattr(self, 'stage_raw_landmarks'):
                self.stage_raw_landmarks = {}
            for stage in selected_stages:
                self.stage_raw_landmarks[str(stage)] = self.live_raw_landmarks_3d.tolist()

        # Extração de Spreads e Fold
        if hasattr(self, 'exact_live_angles'):
            if not hasattr(self, 'spread_limits') or self.spread_limits is None:
                self.spread_limits = {
                    'Pinky_Ring': [0.0, 20.0],
                    'Ring_Middle': [0.0, 18.0],
                    'Middle_Index': [0.0, 20.0],
                    'Index_Thumb': [2.0, 60.0]
                }
            if not hasattr(self, 'thumb_fold_limits') or self.thumb_fold_limits is None:
                self.thumb_fold_limits = {'J1_Yaw_offset': -20.0, 'J1_Pitch_offset': 10.0, 'J1_Yaw_rest': -20.0, 'J1_Pitch_rest': 10.0}
            
            # Definir índices baseados na escolha do usuário
            spread_idx = int(self.spread_state_var.get()[0])
            fold_idx = int(self.fold_state_var.get()[0])

            selected_spreads = [name for name, var in self.spread_vars.items() if var.get()]
            for spread in selected_spreads:
                if spread == 'Polegar-Indicador':
                    val = self.live_spreads.get('Index_Thumb', 0.0)
                    self.spread_limits['Index_Thumb'][spread_idx] = round(val, 1)
                elif spread == 'Indicador-Médio':
                    val = self.live_spreads.get('Middle_Index', 0.0)
                    self.spread_limits['Middle_Index'][spread_idx] = round(val, 1)
                elif spread == 'Médio-Anelar':
                    val = self.live_spreads.get('Ring_Middle', 0.0)
                    self.spread_limits['Ring_Middle'][spread_idx] = round(val, 1)
                elif spread == 'Anelar-Mindinho':
                    val = self.live_spreads.get('Pinky_Ring', 0.0)
                    self.spread_limits['Pinky_Ring'][spread_idx] = round(val, 1)

            if self.fold_var.get() and 'Thumb' in self.exact_live_angles:
                exact = self.exact_live_angles['Thumb']
                if fold_idx == 0:
                    self.thumb_fold_limits['J1_Yaw_offset'] = exact['J1_Yaw']
                    self.thumb_fold_limits['J1_Pitch_offset'] = exact['J1_Pitch']
                else:
                    self.thumb_fold_limits['J1_Yaw_rest'] = exact['J1_Yaw']
                    self.thumb_fold_limits['J1_Pitch_rest'] = exact['J1_Pitch']

        if not applied_fingers:
            messagebox.showwarning(
                "Aviso",
                "Nenhum dado válido de câmera foi observado para os itens selecionados.\n"
                "Por favor, posicione sua mão claramente em frente à webcam e tente novamente."
            )
            return

        # Focar no primeiro dedo e estágio salvos na tela principal após aplicar a calibração da câmera
        if selected_fingers and selected_stages:
            self.set_active_finger(selected_fingers[0])
            self.set_active_stage(selected_stages[0])

        self.save_calibration_file()
        self.update_gui_from_model()
        self.redraw_hand()
        
        self.close_camera()
        
        fingers_str = ", ".join(applied_fingers)
        stages_str = ", ".join(selected_stages)
        print(f"[SUCESSO] Limites da câmera adotados para: {fingers_str} nos estágios: {stages_str}")
        messagebox.showinfo(
            "Sucesso",
            f"Calibração via câmera aplicada com sucesso!\n\n"
            f"Dedos atualizados: {fingers_str}\n"
            f"Estágios aplicados: {stages_str}"
        )

    def open_json_ingestion_dialog(self):
        # Janela modal para entrada e ingestão inteligente de JSON
        dialog = tk.Toplevel(self.root)
        dialog.title("Ingestão Inteligente de Calibração via JSON")
        dialog.geometry("1100x680")
        dialog.configure(bg=COLORS['bg_main'])
        dialog.transient(self.root)
        dialog.grab_set()

        # Centralizar diálogo com relação à janela principal
        dialog.update_idletasks()
        r_w = self.root.winfo_width()
        r_h = self.root.winfo_height()
        r_x = self.root.winfo_rootx()
        r_y = self.root.winfo_rooty()
        d_w = dialog.winfo_width()
        d_h = dialog.winfo_height()
        x = r_x + (r_w - d_w) // 2
        y = r_y + (r_h - d_h) // 2
        dialog.geometry(f"+{x}+{y}")

        # Título
        lbl_title = tk.Label(
            dialog, text="INGESTÃO INTELIGENTE DE JSON",
            fg=COLORS['accent_blue'], bg=COLORS['bg_main'],
            font=("Segoe UI", 12, "bold")
        )
        lbl_title.pack(pady=(15, 5))

        lbl_desc = tk.Label(
            dialog,
            text="Cole o JSON de calibração abaixo. Chaves em português, minúsculas, e intervalos [min, max] são aceitos e convertidos automaticamente.",
            fg=COLORS['text_muted'], bg=COLORS['bg_main'],
            font=("Segoe UI", 9), wraplength=1000, justify='center'
        )
        lbl_desc.pack(pady=(0, 10))

        # Container Lado a Lado
        cols_frame = tk.Frame(dialog, bg=COLORS['bg_main'])
        cols_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # Coluna Esquerda: Documentação de Ajuda
        help_frame = tk.Frame(cols_frame, bg=COLORS['bg_sidebar'], padx=15, pady=15)
        help_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        lbl_help_title = tk.Label(
            help_frame, text="GUIA DE CONFIGURAÇÃO & INGESTÃO",
            fg=COLORS['accent_blue'], bg=COLORS['bg_sidebar'],
            font=("Segoe UI", 10, "bold")
        )
        lbl_help_title.pack(anchor=tk.W, pady=(0, 10))

        help_scroll = ttk.Scrollbar(help_frame)
        help_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        txt_help = tk.Text(
            help_frame, bg=COLORS['bg_sidebar'], fg=COLORS['text_main'],
            insertbackground=COLORS['text_main'], font=("Segoe UI", 9),
            bd=0, relief='flat', wrap=tk.WORD, yscrollcommand=help_scroll.set
        )
        txt_help.pack(fill=tk.BOTH, expand=True)
        help_scroll.config(command=txt_help.yview)

        help_text = (
            "=== INGESTÃO INTELIGENTE DE CALIBRAÇÃO ===\n\n"
            "O calibrador suporta a importação de arquivos de calibração formatados em JSON.\n"
            "Ele é inteligente e tolerante, traduzindo e mapeando automaticamente chaves\n"
            "em português, acentos, letras maiúsculas/minúsculas e termos cinesiológicos.\n\n"
            
            "--- 1. MODELO DE INTERVALOS [MIN, MAX] (RECOMENDADO) ---\n"
            "Ao invés de definir individualmente cada um dos 4 estágios, você pode apenas\n"
            "fornecer os limites [mínimo, máximo] observados das juntas MCP e PIP.\n"
            "O sistema fará a Interpolação Linear (LERP) automática dos 4 estágios:\n"
            "  * Estágio 0 (Aberto): Juntas no valor mínimo (totalmente estendido)\n"
            "  * Estágio 1 (Garra Leve): MCP = 15% flexão | PIP = 50% flexão\n"
            "  * Estágio 2 (Plataforma): MCP = mínimo | PIP = máximo\n"
            "  * Estágio 3 (Fechado/Soco): MCP = máximo | PIP = máximo\n\n"
            
            "--- 2. MODELO EXPLÍCITO (ESTÁGIOS DETALHADOS) ---\n"
            "Se preferir controle absoluto, você pode especificar os valores exatos de\n"
            "ângulo (em graus) de Yaw e Pitch para as juntas de cada estágio (0, 1, 2, 3).\n\n"
            
            "--- 3. MAPEAMENTO DE DEDOS E CHAVES ---\n"
            "O parser mapeia os seguintes sinônimos (sem distinção de acentos/caixa):\n"
            "  * Polegar: 'polegar', 'thumb'\n"
            "  * Indicador: 'indicador', 'index'\n"
            "  * Médio: 'medio', 'médio', 'middle'\n"
            "  * Anelar: 'anelar', 'ring'\n"
            "  * Mindinho: 'mindinho', 'pinky'\n\n"
            
            "--- 4. JUNTAS E DIREÇÃO ---\n"
            "A) Dedos Longos (Indicador a Mindinho):\n"
            "  * MCP (Junta 1): 'MCP', 'J1', 'flexao mcp', 'lateral mcp'\n"
            "  * PIP (Junta 2): 'PIP', 'J2', 'flexao pip', 'lateral pip'\n"
            "  * DIP (Junta 3): 'DIP', 'J3', 'flexao dip', 'lateral dip'\n"
            "B) Polegar:\n"
            "  * CMC (Junta 1): 'CMC', 'J1', 'flexao cmc', 'lateral cmc'\n"
            "  * MCP (Junta 2): 'MCP', 'J2', 'flexao mcp'\n"
            "  * IP  (Junta 3): 'IP', 'J3', 'flexao ip'\n\n"
            
            "--- 5. REGRAS BIOMECÂNICAS E ACOPLAMENTO ---\n"
            "As seguintes regras biomecânicas ativas são preservadas após a ingestão:\n"
            "  * Acoplamento do Anelar: Se ativo, as juntas J2/J3 do anelar seguirão as do mindinho.\n"
            "  * Restrição de Spread (Yaw): O afastamento lateral diminui automaticamente\n"
            "    conforme o dedo se flexiona, simulando a biomecânica humana natural.\n"
        )
        txt_help.insert("1.0", help_text)
        txt_help.configure(state=tk.DISABLED)

        # Coluna Direita: Editor de JSON
        editor_frame = tk.Frame(cols_frame, bg=COLORS['bg_main'])
        editor_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        lbl_editor_title = tk.Label(
            editor_frame, text="EDITOR / ÁREA DE COLAGEM",
            fg=COLORS['accent_blue'], bg=COLORS['bg_main'],
            font=("Segoe UI", 10, "bold")
        )
        lbl_editor_title.pack(anchor=tk.W, pady=(0, 5))

        text_frame = tk.Frame(editor_frame, bg=COLORS['bg_canvas'])
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        txt_json = tk.Text(
            text_frame, bg=COLORS['bg_canvas'], fg=COLORS['text_main'],
            insertbackground=COLORS['text_main'], font=("Consolas", 10),
            bd=0, relief='flat', padx=10, pady=10, yscrollcommand=scrollbar.set
        )
        txt_json.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=txt_json.yview)

        # Botões auxiliares de texto
        btn_row_helper = tk.Frame(editor_frame, bg=COLORS['bg_main'])
        btn_row_helper.pack(fill=tk.X, pady=5)

        example_json = {
            "stages": {
                "_comment_1": "FORMATO POR INTERVALO (AUTOMÁTICO LERP PARA ESTÁGIOS 0-3)",
                "indicador": {
                    "MCP": [5.0, 85.0],
                    "PIP": [5.0, 110.0]
                },
                "medio": {
                    "MCP": [5.0, 90.0],
                    "PIP": [5.0, 115.0]
                },
                "anelar": {
                    "MCP": [5.0, 80.0],
                    "PIP": [5.0, 105.0]
                },
                "mindinho": {
                    "MCP": [5.0, 85.0],
                    "PIP": [5.0, 100.0]
                },
                "_comment_2": "FORMATO EXPLÍCITO (ESTÁGIOS ESPECÍFICOS)",
                "polegar": {
                    "estagio_0": {
                        "CMC_Yaw": -25.0,
                        "CMC_Pitch": 5.4,
                        "MCP_Pitch": 10.0,
                        "IP_Pitch": 5.0
                    },
                    "estagio_3": {
                        "CMC_Yaw": -21.2,
                        "CMC_Pitch": 37.3,
                        "MCP_Pitch": 50.0,
                        "IP_Pitch": 60.0
                    }
                }
            }
        }

        def load_example():
            txt_json.delete("1.0", tk.END)
            txt_json.insert("1.0", json.dumps(example_json, indent=4, ensure_ascii=False))

        btn_example = tk.Button(
            btn_row_helper, text="Carregar Exemplo com Placeholders", bg=COLORS['bg_card'], fg=COLORS['text_muted'],
            activebackground=COLORS['accent_blue'], activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=10, pady=4,
            command=load_example
        )
        btn_example.pack(side=tk.LEFT, padx=(0, 5))

        def clear_text():
            txt_json.delete("1.0", tk.END)

        btn_clear = tk.Button(
            btn_row_helper, text="Limpar Código", bg=COLORS['bg_card'], fg=COLORS['text_muted'],
            activebackground=COLORS['accent_blue'], activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=10, pady=4,
            command=clear_text
        )
        btn_clear.pack(side=tk.LEFT)

        # Botões de Ação
        btn_actions = tk.Frame(dialog, bg=COLORS['bg_main'])
        btn_actions.pack(fill=tk.X, padx=20, pady=(15, 20))

        def process_ingestion():
            json_str = txt_json.get("1.0", tk.END).strip()
            
            # Limpador inteligente de Markdown do LLM
            if json_str.startswith('```'):
                lines = json_str.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                json_str = '\n'.join(lines).strip()
                
            if not json_str:
                messagebox.showwarning("Aviso", "Por favor, cole um código JSON.")
                return

            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError as err:
                print(f"[ERRO DE SINTAXE JSON]\n{err}\nTexto recebido:\n{json_str}")
                messagebox.showerror("Erro de Sintaxe JSON", f"O JSON inserido possui um erro estrutural (vírgula faltando, aspas duplas, etc):\n\n{err}")
                return

            try:
                updated_count, details = self.parse_and_ingest_json(parsed_data)
                if updated_count > 0:
                    self.save_calibration_file()
                    self.update_gui_from_model()
                    self.redraw_hand()
                    dialog.destroy()
                    
                    details_str = "\n".join([f" - {d}" for d in details])
                    messagebox.showinfo(
                        "Sucesso",
                        f"Ingestão inteligente realizada com sucesso!\n\n"
                        f"Dedos atualizados:\n{details_str}"
                    )
                else:
                    messagebox.showwarning(
                        "Aviso",
                        "Nenhuma configuração compatível de dedo ou junta foi identificada no JSON."
                    )
            except Exception as ex:
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERRO DE INGESTÃO JSON]\n{error_trace}")
                messagebox.showerror("Erro na Ingestão", f"Falha ao interpretar a calibração:\n\n{error_trace}")

        btn_confirm = tk.Button(
            btn_actions, text="PROCESSAR E INGERIR", bg=COLORS['accent_green'], fg='#11111B',
            activebackground='#8be087', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 10, "bold"), pady=8, cursor="hand2",
            command=process_ingestion
        )
        btn_confirm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_cancel = tk.Button(
            btn_actions, text="CANCELAR / FECHAR", bg=COLORS['accent_red'], fg='#11111B',
            activebackground='#fc9d9d', activeforeground='#11111B', relief='flat',
            font=("Segoe UI", 10, "bold"), pady=8, cursor="hand2",
            command=dialog.destroy
        )
        btn_cancel.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    def parse_and_ingest_json(self, data):
        import unicodedata
        updates = {} # finger -> stage -> joint -> value
        
        def normalize_str(s):
            s = str(s).lower().strip()
            s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            return s

        finger_map = {
            'thumb': 'Thumb', 'polegar': 'Thumb', 'polegar - cmc (junta 1)': 'Thumb',
            'index': 'Index', 'indicador': 'Index', 'indicador - mcp (junta 1)': 'Index',
            'middle': 'Middle', 'medio': 'Middle', 'médio': 'Middle', 'médio - mcp (junta 1)': 'Middle',
            'ring': 'Ring', 'anelar': 'Ring', 'anelar - mcp (junta 1)': 'Ring',
            'pinky': 'Pinky', 'mindinho': 'Pinky', 'mindinho - mcp (junta 1)': 'Pinky'
        }

        stage_map = {
            '0': '0', 'aberto': '0', 'totalmente aberto': '0',
            '1': '1', 'garra': '1', 'garra leve': '1',
            '2': '2', 'plataforma': '2',
            '3': '3', 'soco': '3', 'fechado': '3', 'totalmente fechado': '3'
        }

        # Check if "stages" is nested in the JSON root
        if isinstance(data, dict) and "stages" in data:
            raw_stages = data["stages"]
        else:
            raw_stages = data

        if not isinstance(raw_stages, dict):
            raise ValueError("O JSON fornecido deve ser um objeto/dicionário.")

        for k, val in raw_stages.items():
            norm_k = normalize_str(k)
            
            # Check if key is a stage (stage-first format)
            if norm_k in stage_map or (norm_k.startswith('estagio') or norm_k.startswith('stage')):
                stage_num = None
                if norm_k in stage_map:
                    stage_num = stage_map[norm_k]
                else:
                    digits = [c for c in norm_k if c.isdigit()]
                    if digits: stage_num = digits[0]
                
                if stage_num and isinstance(val, dict):
                    for f_k, f_val in val.items():
                        norm_f = normalize_str(f_k)
                        if norm_f in finger_map and isinstance(f_val, dict):
                            finger = finger_map[norm_f]
                            if finger not in updates: updates[finger] = {}
                            if stage_num not in updates[finger]: updates[finger][stage_num] = {}
                            self._parse_joints_dict(f_val, updates[finger][stage_num], finger)
                            
            # Check if key is a finger (finger-first format)
            elif norm_k in finger_map or (norm_k.startswith('dedo') or norm_k.startswith('finger')):
                finger = finger_map[norm_k]
                if finger not in updates: updates[finger] = {}
                if isinstance(val, dict):
                    # Check if there are nested stage keys, or direct joint keys
                    is_stage_nested = False
                    for inner_k, inner_v in val.items():
                        norm_inner = normalize_str(inner_k)
                        if norm_inner in stage_map or norm_inner.isdigit() or norm_inner.startswith('estagio') or norm_inner.startswith('stage'):
                            is_stage_nested = True
                            break
                    
                    if is_stage_nested:
                        # nested stages: finger -> stage -> joints
                        for st_k, st_val in val.items():
                            norm_st = normalize_str(st_k)
                            stage_num = None
                            if norm_st in stage_map:
                                stage_num = stage_map[norm_st]
                            else:
                                digits = [c for c in norm_st if c.isdigit()]
                                if digits: stage_num = digits[0]
                            
                            if stage_num and isinstance(st_val, dict):
                                if stage_num not in updates[finger]: updates[finger][stage_num] = {}
                                self._parse_joints_dict(st_val, updates[finger][stage_num], finger)
                    else:
                        # direct joints: e.g. finger -> joints (can be values or ranges)
                        mcp_range = None
                        pip_range = None
                        for jk, jv in val.items():
                            norm_jk = normalize_str(jk)
                            if isinstance(jv, (list, tuple)) and len(jv) == 2:
                                if 'mcp' in norm_jk:
                                    mcp_range = [float(jv[0]), float(jv[1])]
                                elif 'pip' in norm_jk or 'dip' in norm_jk:
                                    pip_range = [float(jv[0]), float(jv[1])]
                                elif 'cmc' in norm_jk and finger == 'Thumb':
                                    mcp_range = [float(jv[0]), float(jv[1])]
                                elif 'ip' in norm_jk and finger == 'Thumb':
                                    pip_range = [float(jv[0]), float(jv[1])]
                        
                        if mcp_range or pip_range:
                            mcp_min, mcp_max = mcp_range if mcp_range else [5.0, 90.0]
                            pip_min, pip_max = pip_range if pip_range else [5.0, 90.0]
                            
                            mcp_key = 'J2_Pitch' if finger == 'Thumb' else 'J1_Pitch'
                            pip_key = 'J3_Pitch' if finger == 'Thumb' else 'J2_Pitch'
                            
                            for s in ['0', '1', '2', '3']:
                                if s not in updates[finger]: updates[finger][s] = {}
                                if s == '0':
                                    updates[finger][s][mcp_key] = mcp_min
                                    updates[finger][s][pip_key] = pip_min
                                elif s == '1':
                                    updates[finger][s][mcp_key] = lerp(mcp_min, mcp_max, 0.15)
                                    updates[finger][s][pip_key] = lerp(pip_min, pip_max, 0.5)
                                elif s == '2':
                                    updates[finger][s][mcp_key] = mcp_min
                                    updates[finger][s][pip_key] = pip_max
                                elif s == '3':
                                    updates[finger][s][mcp_key] = mcp_max
                                    updates[finger][s][pip_key] = pip_max
                                    
                                if finger != 'Thumb':
                                    updates[finger][s]['J3_Pitch'] = updates[finger][s][pip_key]
                        
                        single_joints = {}
                        self._parse_joints_dict(val, single_joints, finger)
                        for sj_key, sj_val in single_joints.items():
                            for s in ['0', '1', '2', '3']:
                                if s not in updates[finger]: updates[finger][s] = {}
                                updates[finger][s][sj_key] = sj_val

        # Apply updates to self.stages
        updated_count = 0
        updated_details = []
        for finger, f_stages in updates.items():
            if f_stages:
                updated_count += 1
                stages_list = sorted(list(f_stages.keys()))
                stages_desc = ", ".join([f"Estágio {st}" for st in stages_list])
                updated_details.append(f"{self.translate_finger_name(finger)} ({stages_desc})")
                for stage, s_joints in f_stages.items():
                    for joint_key, val in s_joints.items():
                        self.stages[finger][stage][joint_key] = val
                        
                if finger == 'Pinky':
                    for stage in f_stages:
                        for j in ['J2', 'J3']:
                            if f"{j}_Pitch" in self.stages['Pinky'][stage]:
                                self.stages['Ring'][stage][f"{j}_Pitch"] = self.stages['Pinky'][stage][f"{j}_Pitch"]
                                
        return updated_count, updated_details

    def _parse_joints_dict(self, raw_dict, target_dict, finger):
        import unicodedata
        def normalize_str(s):
            s = str(s).lower().strip()
            s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            return s

        for jk, jv in raw_dict.items():
            if isinstance(jv, (list, tuple)):
                continue
            try:
                val = float(jv)
            except (ValueError, TypeError):
                continue
                
            norm_jk = normalize_str(jk)
            
            is_yaw = 'yaw' in norm_jk or '_y' in norm_jk or 'lateral' in norm_jk
            is_pitch = 'pitch' in norm_jk or '_p' in norm_jk or 'flex' in norm_jk or 'mcp' in norm_jk or 'pip' in norm_jk or 'dip' in norm_jk or 'ip' in norm_jk or 'cmc' in norm_jk
            
            if not is_yaw and not is_pitch:
                is_pitch = True
                
            joint_num = None
            if '1' in norm_jk or 'cmc' in norm_jk or ('mcp' in norm_jk and finger != 'Thumb'):
                joint_num = 'J1'
            elif '2' in norm_jk or ('mcp' in norm_jk and finger == 'Thumb') or ('pip' in norm_jk and finger != 'Thumb'):
                joint_num = 'J2'
            elif '3' in norm_jk or ('ip' in norm_jk and finger == 'Thumb') or ('dip' in norm_jk and finger != 'Thumb'):
                joint_num = 'J3'
            
            if 'j1_yaw' in norm_jk: joint_num, is_yaw = 'J1', True
            elif 'j1_pitch' in norm_jk: joint_num, is_pitch = 'J1', True
            elif 'j2_yaw' in norm_jk: joint_num, is_yaw = 'J2', True
            elif 'j2_pitch' in norm_jk: joint_num, is_pitch = 'J2', True
            elif 'j3_yaw' in norm_jk: joint_num, is_yaw = 'J3', True
            elif 'j3_pitch' in norm_jk: joint_num, is_pitch = 'J3', True

            if joint_num:
                suffix = "Yaw" if is_yaw else "Pitch"
                key = f"{joint_num}_{suffix}"
                target_dict[key] = val

# ---------------------------------------------------------
# EXECUÇÃO DA APLICAÇÃO
# ---------------------------------------------------------
def main():
    root = tk.Tk()
    app = HandCalibratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
