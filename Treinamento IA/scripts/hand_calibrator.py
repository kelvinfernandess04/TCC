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
    'Thumb': '#89B4FA',
    'Index': '#A6E3A1',
    'Middle': '#F9E2AF',
    'Ring': '#F5C2E7',
    'Pinky': '#F38BA8',
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
    ("Index",  "J1"): 5,
    ("Index",  "J2"): 6,
    ("Index",  "J3"): 7,
    ("Middle", "J1"): 9,
    ("Middle", "J2"): 10,
    ("Middle", "J3"): 11,
    ("Ring",   "J1"): 13,
    ("Ring",   "J2"): 14,
    ("Ring",   "J3"): 15,
    ("Pinky",  "J1"): 17,
    ("Pinky",  "J2"): 18,
    ("Pinky",  "J3"): 19
}

LANDMARK_TO_JOINT = {
    1: ("Thumb",  "J1"),
    2: ("Thumb",  "J2"),
    3: ("Thumb",  "J3"),
    4: ("Thumb",  "J3"),
    5: ("Index",  "J1"),
    6: ("Index",  "J2"),
    7: ("Index",  "J3"),
    8: ("Index",  "J3"),
    9: ("Middle", "J1"),
    10: ("Middle", "J2"),
    11: ("Middle", "J3"),
    12: ("Middle", "J3"),
    13: ("Ring",   "J1"),
    14: ("Ring",   "J2"),
    15: ("Ring",   "J3"),
    16: ("Ring",   "J3"),
    17: ("Pinky",  "J1"),
    18: ("Pinky",  "J2"),
    19: ("Pinky",  "J3"),
    20: ("Pinky",  "J3")
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
                        'J3_Pitch': pip_val
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
                        'J3_Pitch': pip_val
                    }

        self.rule_spread_constraint = True
        self.rule_tendon_pinky_ring = True

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

        # Estado da edição atual
        self.active_finger = 'Index'
        self.active_stage = '0'
        self.active_joint = 'J1'
        self.active_landmark_idx = 5  # Index MCP por padrão
        self.updating_gui = False  # Lock para evitar recursão infinita

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
                    # Compatibilidade retroativa para converter chaves clássicas
                    for f in loaded_stages:
                        for s in loaded_stages[f]:
                            item = loaded_stages[f][s]
                            if 'MCP' in item and 'J1_Pitch' not in item:
                                if f == 'Thumb':
                                    cy = item.get('CMC_Yaw', -25.0)
                                    cp = item.get('CMC_Pitch', 5.4)
                                    mcp = item.get('MCP', 10.0)
                                    pip = item.get('PIP', 5.0)
                                    item['J1_Yaw'] = cy
                                    item['J1_Pitch'] = cp
                                    item['J2_Yaw'] = 0.0
                                    item['J2_Pitch'] = mcp
                                    item['J3_Yaw'] = 0.0
                                    item['J3_Pitch'] = pip
                                else:
                                    mcp = item.get('MCP', 5.0)
                                    pip = item.get('PIP', 5.0)
                                    def_y = {'Index': 5.0, 'Middle': 0.0, 'Ring': -5.0, 'Pinky': -15.0}
                                    item['J1_Yaw'] = def_y.get(f, 0.0)
                                    item['J1_Pitch'] = mcp
                                    item['J2_Yaw'] = 0.0
                                    item['J2_Pitch'] = pip
                                    item['J3_Yaw'] = 0.0
                                    item['J3_Pitch'] = pip
                    self.stages = loaded_stages
                
                self.rule_spread_constraint = saved.get("rule_spread_constraint", self.rule_spread_constraint)
                self.rule_tendon_pinky_ring = saved.get("rule_tendon_pinky_ring", self.rule_tendon_pinky_ring)
                
                # Sincronizar estados da simulação com estágio ativo inicial
                for f in self.finger_states:
                    self.finger_states[f] = int(self.active_stage)
                    
                print(f"[SISTEMA] Calibração carregada de: {CALIBRATION_FILE}")
            except Exception as e:
                print(f"[Aviso] Falha ao ler calibração: {e}")

    def save_calibration_file(self):
        data = {
            "stages": self.stages,
            "rule_spread_constraint": self.rule_spread_constraint,
            "rule_tendon_pinky_ring": self.rule_tendon_pinky_ring
        }
        try:
            os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
            with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"[SUCESSO] Configuração de calibração salva em: {CALIBRATION_FILE}")
            messagebox.showinfo("Sucesso", "Configurações de calibração salvas com sucesso!")
        except Exception as e:
            print(f"[Erro] Falha ao salvar calibração: {e}")
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

        # 3. REGRAS BIOMECÂNICAS (SEÇÃO ISOLADA)
        self.build_rules_card()

        # 4. AÇÕES DO SISTEMA
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
            self.yaw_frame, from_=-360, to=360, orient=tk.HORIZONTAL, bg=COLORS['bg_card'],
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
            self.pitch_frame, from_=-360, to=360, orient=tk.HORIZONTAL, bg=COLORS['bg_card'],
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

    def build_rules_card(self):
        # Seção de Regras Biomecânicas Ativas Isolada
        card = tk.Frame(self.scrollable_frame, bg=COLORS['bg_card'], padx=10, pady=10)
        card.pack(fill=tk.X, pady=(0, 10))

        lbl_r = tk.Label(card, text="Regras Biomecânicas Ativas", fg=COLORS['text_main'], bg=COLORS['bg_card'], font=("Segoe UI", 10, "bold"))
        lbl_r.pack(anchor=tk.W, pady=(0, 8))

        self.chk_spread = tk.BooleanVar(value=self.rule_spread_constraint)
        cb_spread = tk.Checkbutton(
            card, text="Spread restrito à medida que flexiona", variable=self.chk_spread,
            bg=COLORS['bg_card'], fg=COLORS['text_main'], activebackground=COLORS['bg_card'],
            activeforeground=COLORS['text_main'], selectcolor=COLORS['bg_sidebar'],
            font=("Segoe UI", 9), command=self.on_rules_toggle
        )
        cb_spread.pack(anchor=tk.W, pady=2)

        self.chk_tendon = tk.BooleanVar(value=self.rule_tendon_pinky_ring)
        cb_tendon = tk.Checkbutton(
            card, text="Tendon acoplado (Anelar segue o Mindinho)", variable=self.chk_tendon,
            bg=COLORS['bg_card'], fg=COLORS['text_main'], activebackground=COLORS['bg_card'],
            activeforeground=COLORS['text_main'], selectcolor=COLORS['bg_sidebar'],
            font=("Segoe UI", 9), command=self.on_rules_toggle
        )
        cb_tendon.pack(anchor=tk.W, pady=2)

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
        print(f"[AÇÃO] Seleção de Dedo - Dedo Ativo alterado para: {finger_code}")
        
        # Estilizar botões de dedo
        for code, btn in self.finger_buttons.items():
            if code == finger_code:
                btn.configure(bg=COLORS[finger_code], fg='#11111B')
            else:
                btn.configure(bg=COLORS['bg_sidebar'], fg=COLORS['text_muted'])

        # Atualizar a simulação para focar no estágio atual deste dedo
        self.finger_states[finger_code] = int(self.active_stage)
        if self.rule_tendon_pinky_ring and finger_code == 'Pinky':
            self.finger_states['Ring'] = int(self.active_stage)

        # Sincronizar o Combobox com a Junta 1 (MCP ou CMC) do novo dedo
        joint_name = "CMC" if finger_code == 'Thumb' else "MCP"
        joint_label = f"{self.translate_finger_name(finger_code)} - {joint_name} (Junta 1)"
        self.joint_selector.set(joint_label)

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

            # Locks anatômicos para dedos longos PIP/DIP
            if finger != 'Thumb' and joint in ['J2', 'J3']:
                self.yaw_slider.configure(state='disabled')
                self.yaw_entry.configure(state='disabled')
                self.lbl_yaw_title.configure(text="Yaw (Travado):")
            else:
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
        if finger != 'Thumb' and joint in ['J2', 'J3']:
            v = 0.0
            self.updating_gui = True
            self.yaw_slider.set(0)
            self.updating_gui = False

        self.stages[finger][state][f"{joint}_Yaw"] = v
        print(f"[SISTEMA YAW] {finger} {joint} Yaw no Estágio {state} alterado para: {v:.1f}°")

        if self.rule_tendon_pinky_ring and finger == 'Pinky' and joint in ['J2', 'J3']:
            self.stages['Ring'][state][f"{joint}_Yaw"] = v

        self.updating_gui = True
        self.yaw_entry.delete(0, tk.END)
        self.yaw_entry.insert(0, f"{v:.1f}")
        self.updating_gui = False

        self.redraw_hand()

    def on_pitch_slider_move(self, val):
        if self.updating_gui:
            return
        selected_text = self.joint_selector.get()
        res = JOINT_MAPPING.get(selected_text)
        if not res:
            return
        finger, joint = res
        state = self.active_stage

        v = float(val)
        
        if finger != 'Thumb' and joint in ['J2', 'J3']:
            self.stages[finger][state]['J2_Pitch'] = v
            self.stages[finger][state]['J3_Pitch'] = v
            print(f"[SISTEMA PITCH] {finger} PIP-DIP acoplados no Estágio {state} definidos para: {v:.1f}°")
            if self.rule_tendon_pinky_ring and finger == 'Pinky':
                self.stages['Ring'][state]['J2_Pitch'] = v
                self.stages['Ring'][state]['J3_Pitch'] = v
        else:
            self.stages[finger][state][f"{joint}_Pitch"] = v
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
            val = np.clip(val, -360.0, 360.0)
            if finger != 'Thumb' and joint in ['J2', 'J3']:
                val = 0.0

            self.stages[finger][state][f"{joint}_Yaw"] = val
            print(f"[SISTEMA INPUT] {finger} {joint} Yaw no Estágio {state} digitado: {val:.1f}°")

            if self.rule_tendon_pinky_ring and finger == 'Pinky' and joint in ['J2', 'J3']:
                self.stages['Ring'][state][f"{joint}_Yaw"] = val

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
            val = np.clip(val, -360.0, 360.0)

            if finger != 'Thumb' and joint in ['J2', 'J3']:
                self.stages[finger][state]['J2_Pitch'] = val
                self.stages[finger][state]['J3_Pitch'] = val
                print(f"[SISTEMA INPUT] {finger} J2 & J3 acoplados no Estágio {state} digitados: {val:.1f}°")
                if self.rule_tendon_pinky_ring and finger == 'Pinky':
                    self.stages['Ring'][state]['J2_Pitch'] = val
                    self.stages['Ring'][state]['J3_Pitch'] = val
            else:
                self.stages[finger][state][f"{joint}_Pitch"] = val
                print(f"[SISTEMA INPUT] {finger} {joint} Pitch no Estágio {state} digitado: {val:.1f}°")

            self.updating_gui = True
            self.pitch_slider.set(int(val))
            self.pitch_entry.delete(0, tk.END)
            self.pitch_entry.insert(0, f"{val:.1f}")
            self.updating_gui = False

            self.redraw_hand()
        except ValueError:
            self.update_gui_from_model()

    def on_rules_toggle(self):
        self.rule_spread_constraint = self.chk_spread.get()
        self.rule_tendon_pinky_ring = self.chk_tendon.get()
        print(f"[AÇÃO] Regras Biomecânicas - Alteradas: Spread constraint={self.rule_spread_constraint} | Acoplamento tendão={self.rule_tendon_pinky_ring}")
        
        if self.rule_tendon_pinky_ring:
            for s in range(4):
                state = str(s)
                for j in ['J2', 'J3']:
                    self.stages['Ring'][state][f"{j}_Pitch"] = self.stages['Pinky'][state][f"{j}_Pitch"]
            self.finger_states['Ring'] = self.finger_states['Pinky']
            
        self.redraw_hand()

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

        y_key = f"{joint}_Yaw"
        p_key = f"{joint}_Pitch"

        if finger != 'Thumb' and joint in ['J2', 'J3']:
            # Dedos longos PIP/DIP: yaw travado em 0.0, pitch acoplado
            self.stages[finger][state]['J2_Yaw'] = 0.0
            self.stages[finger][state]['J3_Yaw'] = 0.0
            curr_pitch = self.stages[finger][state]['J2_Pitch']
            new_pitch = np.clip(curr_pitch + delta_pitch, -360.0, 360.0)
            self.stages[finger][state]['J2_Pitch'] = new_pitch
            self.stages[finger][state]['J3_Pitch'] = new_pitch
            print(f"[ARRASADO] {finger} PIP-DIP acoplados: Pitch definido para {new_pitch:.1f}°")
            if self.rule_tendon_pinky_ring and finger == 'Pinky':
                self.stages['Ring'][state]['J2_Pitch'] = new_pitch
                self.stages['Ring'][state]['J3_Pitch'] = new_pitch
        else:
            # Isolado e independente
            curr_yaw = self.stages[finger][state][y_key]
            curr_pitch = self.stages[finger][state][p_key]

            new_yaw = np.clip(curr_yaw + delta_yaw, -360.0, 360.0)
            new_pitch = np.clip(curr_pitch + delta_pitch, -360.0, 360.0)

            self.stages[finger][state][y_key] = new_yaw
            self.stages[finger][state][p_key] = new_pitch
            print(f"[ARRASADO] {finger} {joint}: Yaw={new_yaw:.1f}°, Pitch={new_pitch:.1f}°")

    # ---------------------------------------------------------
    # CINEMÁTICA DIRETA RECURSIVA PURA YAW & PITCH
    # ---------------------------------------------------------
    def generate_simulated_hand_3d(self):
        palm_bases = {
            'Thumb':  np.array([-0.06, 0.04, 0.02]),
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
        if self.rule_tendon_pinky_ring:
            f_states['Ring'] = f_states['Pinky']

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
                opp_factor = float(state) / 3.0
                p_idx = 0 if state == '0' else (2 if state == '2' else 3)
                if state == '1': p_idx = 2

                lens_L0 = thumb_lengths[(0, p_idx)]
                lens_L1 = thumb_lengths[(1, p_idx)]
                lengths = [lerp(l0, l1, opp_factor) for l0, l1 in zip(lens_L0, lens_L1)]

            j1_y = self.stages[finger][state]['J1_Yaw']
            j1_p = self.stages[finger][state]['J1_Pitch']
            j2_y = self.stages[finger][state]['J2_Yaw']
            j2_p = self.stages[finger][state]['J2_Pitch']
            j3_y = self.stages[finger][state]['J3_Yaw']
            j3_p = self.stages[finger][state]['J3_Pitch']

            if finger != 'Thumb':
                j2_y = 0.0
                j3_y = 0.0
                j3_p = j2_p

            if self.rule_spread_constraint:
                sp_factor = max(0.0, 1.0 - (float(state) / 3.0))
                j1_y = j1_y * sp_factor

            # J1 rotates segment 0-base (Wrist to MCP/CMC)
            R1 = rot_z(j1_y).dot(rot_x(j1_p))
            p1 = R1.dot(palm_bases[finger])

            # J2 rotates segment base-mid1
            R2 = R1.dot(rot_z(j2_y).dot(rot_x(j2_p)))
            p2 = p1 + R2.dot(np.array([0.0, lengths[0], 0.0]))

            # J3 rotates segment mid1-mid2
            R3 = R2.dot(rot_z(j3_y).dot(rot_x(j3_p)))
            p3 = p2 + R3.dot(np.array([0.0, lengths[1], 0.0]))

            # Tip follows mid2 segment (relative rotation 0.0)
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
        self.cam_win.bind("<space>", lambda e: self.apply_live_camera_calibration())
        self.cam_win.bind("<Escape>", lambda e: self.close_camera())
        self.cam_win.bind("c", lambda e: self.close_camera())
        self.cam_win.bind("C", lambda e: self.close_camera())
        self.cam_win.protocol("WM_DELETE_WINDOW", self.close_camera)

        self.root.after(10, self.update_camera_frame)

    def close_camera(self, event=None):
        self.camera_active = False
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'cam_win') and self.cam_win is not None:
            self.cam_win.destroy()
            self.cam_win = None
        print("[CAMERA] Câmera fechada.")

    def reset_live_ranges(self):
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
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

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_camera_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)

        cv2.rectangle(frame, (0, 0), (280, h), (30, 30, 40), -1)
        cv2.putText(frame, "VALORES MEDIDOS:", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms.landmark])

            mcp_triplets = {
                'Thumb':  (0, 1, 2), 'Index':  (0, 5, 6), 'Middle': (0, 9, 10),
                'Ring':   (0, 13, 14), 'Pinky':  (0, 17, 18)
            }

            y_pos = 65
            for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
                a, b, c = mcp_triplets[f]
                mcp_f = joint_flexion(pts[a], pts[b], pts[c])

                if f == 'Thumb':
                    pip_f = joint_flexion(pts[2], pts[3], pts[4])
                else:
                    pip_f = (joint_flexion(pts[5 if f=='Index' else (9 if f=='Middle' else (13 if f=='Ring' else 17))], 
                                           pts[6 if f=='Index' else (10 if f=='Middle' else (14 if f=='Ring' else 18))], 
                                           pts[7 if f=='Index' else (11 if f=='Middle' else (15 if f=='Ring' else 19))]) + 
                              joint_flexion(pts[6 if f=='Index' else (10 if f=='Middle' else (14 if f=='Ring' else 18))], 
                                           pts[7 if f=='Index' else (11 if f=='Middle' else (15 if f=='Ring' else 19))], 
                                           pts[8 if f=='Index' else (12 if f=='Middle' else (16 if f=='Ring' else 20))])) / 2.0

                self.update_live_joint(f, 'MCP', mcp_f)
                self.update_live_joint(f, 'PIP', pip_f)

                color = (161, 227, 166) if f == self.active_finger else (200, 200, 200)
                cv2.putText(frame, f"{self.translate_finger_name(f)}:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
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

        self.root.after(10, self.update_camera_frame)

    def apply_live_camera_calibration(self):
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            l_mcp = self.live_ranges[f]['MCP']
            l_pip = self.live_ranges[f]['PIP']
            
            if l_mcp[0] < 180 and l_mcp[1] > -180:
                mcp_min, mcp_max = float(l_mcp[0]), float(l_mcp[1])
                mcp_key = 'J2_Pitch' if f == 'Thumb' else 'J1_Pitch'
                self.stages[f]['0'][mcp_key] = mcp_min
                self.stages[f]['1'][mcp_key] = lerp(mcp_min, mcp_max, 0.15)
                self.stages[f]['2'][mcp_key] = mcp_min
                self.stages[f]['3'][mcp_key] = mcp_max

            if l_pip[0] < 180 and l_pip[1] > -180:
                pip_min, pip_max = float(l_pip[0]), float(l_pip[1])
                pip_key = 'J3_Pitch' if f == 'Thumb' else 'J2_Pitch'
                self.stages[f]['0'][pip_key] = pip_min
                self.stages[f]['1'][pip_key] = lerp(pip_min, pip_max, 0.5)
                self.stages[f]['2'][pip_key] = pip_max
                self.stages[f]['3'][pip_key] = pip_max
                
                if f != 'Thumb':
                    for s in range(4):
                        self.stages[f][str(s)]['J3_Pitch'] = self.stages[f][str(s)]['J2_Pitch']

        if self.rule_tendon_pinky_ring:
            for s in range(4):
                state = str(s)
                for j in ['J2', 'J3']:
                    self.stages['Ring'][state][f"{j}_Pitch"] = self.stages['Pinky'][state][f"{j}_Pitch"]

        self.save_calibration_file()
        self.update_gui_from_model()
        self.redraw_hand()
        print("[SUCESSO] Limites da câmera foram adotados e persistidos com sucesso!")
        messagebox.showinfo("Sucesso", "Calibração instantânea via câmera aplicada e salva!")

# ---------------------------------------------------------
# EXECUÇÃO DA APLICAÇÃO
# ---------------------------------------------------------
def main():
    root = tk.Tk()
    app = HandCalibratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
