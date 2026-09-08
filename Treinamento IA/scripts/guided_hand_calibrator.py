#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided Hand Calibrator (guided_hand_calibrator.py)
=================================================
Assistente interativo em tempo real (OpenCV + MediaPipe Hands + Pillow) para guiar o usuário
na calibração anatômica individual de cada variável biomecânica da mão:
1. Baseline: Proporções e comprimentos ósseos das falanges (Palma e dedos abertos)
2. Dedos Longos (Indicador, Médio, Anelar, Mindinho):
   - Estágio 0: Estendido (Reto)
   - Estágio 1: Curvado / Concha (Arco suave contínuo)
   - Estágio 2: Gancho / Hook (Base reta, pontas dobradas)
   - Estágio 3: Plataforma / Tabletop (Base a 90°, pontas retas)
   - Estágio 4: Fechado / Punho (Dedo colado na palma)
3. Aberturas Laterais (Spreads entre pares de dedos):
   - Dedos em leque aberto máximo (A=0)
   - Dedos juntos em paralelo (A=1)
4. Movimentação do Polegar:
   - No plano da mão (F=0) com ponta reta (P=0) e dobrada (P=1)
   - Oposição transversal (F=1) com ponta reta (P=0) e dobrada (P=1)

Recursos:
- Renderização tipográfica em UTF-8 com suporte nativo a acentuação via Pillow.
- Modo de Revisão e Confirmação: exibe um cartão com todos os dados extraídos das juntas
  antes de avançar para o próximo passo.
- Instruções anatômicas cirúrgicas de posicionamento da mão e dos outros dedos.
- Estabilização temporal (rejeição de jitter e ruído da câmera).
"""

import os
import sys

# Configurar saída do console para UTF-8 no Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import time
import math
import textwrap
import argparse
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CAPTURES_DIR = os.path.join(DATA_DIR, 'calibration_captures')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'calibration_settings.json')
SEEDS_FILE = os.path.join(DATA_DIR, 'seeds', 'seeds.json')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SEEDS_FILE), exist_ok=True)

# ---------------------------------------------------------------------------
# RENDERIZADOR DE TEXTO UNICODE / UTF-8 COM PILLOW
# ---------------------------------------------------------------------------

class UnicodeHUD:
    """Renderizador tipográfico com suporte total a acentos e caracteres UTF-8."""
    def __init__(self):
        font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        bold_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        ]
        self.font_path = next((p for p in font_candidates if os.path.exists(p)), None)
        self.bold_path = next((p for p in bold_candidates if os.path.exists(p)), self.font_path)
        self.font_cache: Dict[Tuple[int, bool], Any] = {}

    def get_font(self, size: int = 16, bold: bool = False):
        key = (size, bold)
        if key not in self.font_cache:
            p = self.bold_path if bold and self.bold_path else self.font_path
            if p:
                try:
                    self.font_cache[key] = ImageFont.truetype(p, size)
                except Exception:
                    self.font_cache[key] = ImageFont.load_default()
            else:
                self.font_cache[key] = ImageFont.load_default()
        return self.font_cache[key]


# ---------------------------------------------------------------------------
# MATEMÁTICA VETORIAL E BIOMECÂNICA
# ---------------------------------------------------------------------------

def vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula ângulo em graus entre dois vetores 3D."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def joint_flexion(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Calcula a flexão articular do ponto p1 entre p0 e p2 (0 = reto, 90 = ângulo reto)."""
    return 180.0 - vec_angle(p0 - p1, p2 - p1)

def rot_x(deg: float) -> np.ndarray:
    """Matriz de rotação 3D em torno do eixo X (Pitch)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s],
        [0.0, s,    c]
    ], dtype=np.float64)

def rot_y(deg: float) -> np.ndarray:
    """Matriz de rotação 3D em torno do eixo Y (Yaw)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c]
    ], dtype=np.float64)

# Índices das juntas por dedo no padrão MediaPipe
FINGER_JOINTS = {
    'Thumb':  [0, 1, 2, 3, 4],     # Wrist, CMC, MCP, IP, TIP
    'Index':  [0, 5, 6, 7, 8],     # Wrist, MCP, PIP, DIP, TIP
    'Middle': [0, 9, 10, 11, 12],
    'Ring':   [0, 13, 14, 15, 16],
    'Pinky':  [0, 17, 18, 19, 20]
}

# Segmentos ósseos por dedo (pares de índices MediaPipe)
FINGER_SEGMENTS = {
    'Thumb':  [(0, 1), (1, 2), (2, 3), (3, 4)],
    'Index':  [(0, 5), (5, 6), (6, 7), (7, 8)],
    'Middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'Ring':   [(0, 13), (13, 14), (14, 15), (15, 16)],
    'Pinky':  [(0, 17), (17, 18), (18, 19), (19, 20)]
}

# Conexões ósseas da palma
PALM_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]

# Cores BGR anatômicas com alto contraste para o modelo 3D
FINGER_COLORS_BGR = {
    'Thumb':  (40, 140, 255),   # Laranja Âmbar
    'Index':  (70, 225, 255),   # Amarelo Ouro
    'Middle': (110, 230, 130),  # Verde Esmeralda
    'Ring':   (240, 210, 80),   # Ciano / Turquesa
    'Pinky':  (225, 120, 215)   # Magenta / Lavanda
}

# ---------------------------------------------------------------------------
# CATÁLOGO DE PASSOS COM DESCRIÇÕES DETALHADAS E PRECISAS
# ---------------------------------------------------------------------------

CALIBRATION_STEPS = [
    # 0. Baseline
    {
        'id': 'baseline_open',
        'category': 'baseline',
        'title': 'MÃO ESPALMADA - MEDIÇÃO DE PROPORÇÕES (BASELINE)',
        'posture': 'Distância: 40-50 cm da câmera. Palma voltada 100% de frente para a lente (ângulo frontal reto), mão na vertical alinhada ao antebraço.',
        'target_action': 'Abra os 5 dedos totalmente esticados e espaçados naturalmente, sem dobrar nenhuma articulação. A mão deve formar uma superfície plana voltada para a lente (calibra a escala real dos ossos da sua mão).',
        'other_fingers': 'Todos os 5 dedos (polegar, indicador, médio, anelar e mínimo) devem participar esticados.',
        'expected_summary': 'MCP ~ 0° | PIP ~ 0° | DIP ~ 0° | Falanges estendidas a 180°',
        'target_finger': None,
        'expected_stage': 0
    },

    # --- 1. OS 4 DEDOS LONGOS JUNTOS (INDICADOR, MÉDIO, ANELAR, MINDINHO) ---
    {
        'id': 'four_fingers_s0',
        'category': 'four_fingers_flexion',
        'stage': 0,
        'expected_stage': 0,
        'title': '4 DEDOS: Estágio 0 (ESTENDIDOS / RETOS)',
        'posture': 'Palma virada de frente para a câmera na vertical, punho reto.',
        'target_action': 'Estique os 4 dedos longos (indicador, médio, anelar e mindinho) TOTALMENTE RETOS para cima em continuidade com a palma (180°). Nenhuma junta dobrada.',
        'other_fingers': 'Polegar relaxado ou aberto no plano da mão.',
        'expected_summary': 'MCP ~ 0° | PIP ~ 0° | DIP ~ 0° (Todos os 4 dedos retos para cima)',
        'target_finger': 'FourFingers'
    },
    {
        'id': 'four_fingers_s1',
        'category': 'four_fingers_flexion',
        'stage': 1,
        'expected_stage': 1,
        'title': '4 DEDOS: Estágio 1 (CURVADOS / CONCHA)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Curve suavemente os 4 dedos num arco contínuo e uniforme em formato de "C" (como se estivesse segurando uma maçã ou bola de tênis). Nenhum dedo encosta na palma.',
        'other_fingers': 'Polegar acompanha a curvatura suavemente ao lado.',
        'expected_summary': 'MCP ~ 25-30° | PIP ~ 40° | DIP ~ 35° (Arco suave contínuo em C)',
        'target_finger': 'FourFingers'
    },
    {
        'id': 'four_fingers_s2',
        'category': 'four_fingers_flexion',
        'stage': 2,
        'expected_stage': 2,
        'title': '4 DEDOS: Estágio 2 (GANCHOS / GARRAS)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Mantenha a base (MCP) RETA apontando para cima (0° a 15°), mas DOBRE as duas juntas da ponta (PIP e DIP) dos 4 dedos a ~90° para frente (formato de garras de gato).',
        'other_fingers': 'Polegar aberto ou relaxado ao lado.',
        'expected_summary': 'MCP ~ 0-15° (base reta) | PIP ~ 90° | DIP ~ 75° (pontas flexionadas)',
        'target_finger': 'FourFingers'
    },
    {
        'id': 'four_fingers_s3',
        'category': 'four_fingers_flexion',
        'stage': 3,
        'expected_stage': 3,
        'title': '4 DEDOS: Estágio 3 (PLATAFORMA / TABLETOP)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre a junta da base (MCP) dos 4 dedos a 90° para frente em direção à câmera, mantendo as falanges média e distal TOTALMENTE RETAS (formato de mesa horizontal em "L").',
        'other_fingers': 'Polegar relaxado apontando ao lado.',
        'expected_summary': 'MCP ~ 85-90° (base dobrada) | PIP ~ 0° (reto) | DIP ~ 0° (reto)',
        'target_finger': 'FourFingers'
    },
    {
        'id': 'four_fingers_s4',
        'category': 'four_fingers_flexion',
        'stage': 4,
        'expected_stage': 4,
        'title': '4 DEDOS: Estágio 4 (FECHADOS / PUNHO)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre completamente todas as juntas dos 4 dedos colando as polpas digitais firmemente contra a palma da mão (mão fechada em punho).',
        'other_fingers': 'Polegar repousa travando sobre os dedos ou ao lado.',
        'expected_summary': 'MCP ~ 85-90° | PIP ~ 105° | DIP ~ 80° (Dedos fechados na palma)',
        'target_finger': 'FourFingers'
    },

    # --- 5. ABERTURAS LATERAIS (SPREADS) ---
    {
        'id': 'spread_open',
        'category': 'spread',
        'spread_state': 0,
        'title': 'ABERTURA LATERAL: DEDOS EM LEQUE MÁXIMO (A=0)',
        'posture': 'Palma 100% de frente para a câmera na vertical, dedos retos.',
        'target_action': 'Afaste todos os dedos uns dos outros o máximo que conseguir para os lados (abdução máxima em formato de leque aberto). Dedos totalmente retos.',
        'other_fingers': 'Separe bem todos: Mindinho, Anelar, Médio, Indicador e Polegar.',
        'expected_summary': 'Spreads máximos entre todos os pares adjacentes de dedos',
        'target_finger': None
    },
    {
        'id': 'spread_closed',
        'category': 'spread',
        'spread_state': 1,
        'title': 'ABERTURA LATERAL: DEDOS JUNTOS EM PARALELO (A=1)',
        'posture': 'Palma 100% de frente para a câmera na vertical, punho reto.',
        'target_action': 'Junte e cole os 4 dedos longos (indicador, médio, anelar e mínimo) completamente retos e colados lado a lado, sem nenhuma fresta ou espaço entre eles (como no sinal da letra "B" de Libras ou continência militar).',
        'other_fingers': 'Os 4 dedos longos formam um bloco plano e uniforme.',
        'expected_summary': 'Spreads mínimos ~ 0° (Dedos longos paralelos colados)',
        'target_finger': None
    },

    # --- 6. POLEGAR (THUMB) ---
    {
        'id': 'thumb_f0_p0',
        'category': 'thumb',
        'f': 0, 'p': 0,
        'title': 'POLEGAR NO PLANO DA PALMA - PONTA RETA (F=0, P=0)',
        'posture': 'Palma virada de frente para a câmera na vertical.',
        'target_action': 'Abra o polegar para a lateral radial, mantendo-o RIGIDAMENTE NO MESMO PLANO da palma (sem cruzar a frente da mão), com a ponta totalmente reta (como no sinal da letra "L" em Libras).',
        'other_fingers': 'Demais 4 dedos retos para cima ou em posição neutra.',
        'expected_summary': 'Polegar radial aberto ao lado | Ponta IP reta (0°)',
        'target_finger': 'Thumb'
    },
    {
        'id': 'thumb_f0_p1',
        'category': 'thumb',
        'f': 0, 'p': 1,
        'title': 'POLEGAR NO PLANO DA PALMA - PONTA DOBRADA (F=0, P=1)',
        'posture': 'Palma virada de frente para a câmera na vertical.',
        'target_action': 'Mantenha o polegar aberto ao lado no plano da mão, mas DOBRE apenas a falange da ponta (junta IP) para dentro a ~70° ("quebrando" a ponta do polegar enquanto a base metacarpo continua aberta ao lado).',
        'other_fingers': 'Demais dedos estendidos para cima.',
        'expected_summary': 'Polegar ao lado no plano | Ponta IP flexionada (~65-80°)',
        'target_finger': 'Thumb'
    },
    {
        'id': 'thumb_f1',
        'category': 'thumb',
        'f': 1, 'p': 1,
        'title': 'POLEGAR EM OPOSIÇÃO TRANSVERSAL NA PALMA (F=1)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Traga o polegar cruzando transversalmente na FRENTE da palma da mão (em direção à base do anelar/mínimo), em oposição fechada contra a mão.',
        'other_fingers': 'Dedos longos estendidos para permitir visão clara do polegar.',
        'expected_summary': 'Polegar cruzando a palma (oposição transversal completa)',
        'target_finger': 'Thumb'
    }
]

# ---------------------------------------------------------------------------
# CLASSE PRINCIPAL DE CALIBRAÇÃO GUIADA
# ---------------------------------------------------------------------------

class GuidedHandCalibrator:
    def __init__(self, camera_idx: int = 0):
        self.camera_idx = camera_idx
        self.hud = UnicodeHUD()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.current_step_idx = 0
        self.state = "CAPTURING"  # "CAPTURING" ou "REVIEW"
        self.current_sub_angle: str = "FRONTAL"  # "FRONTAL" ou "LATERAL"
        self.step_subcaptures: Dict[str, Any] = {}
        self.captured_data: Dict[str, Any] = {}
        self.current_review_metrics: List[str] = []
        self.current_review_points: List[str] = []
        self.current_review_status: str = "Adequado"
        self.current_review_snapshot: Optional[np.ndarray] = None

        self.stable_frame_buffer: List[np.ndarray] = []
        self.stability_start_time: Optional[float] = None
        self.REQUIRED_STABLE_TIME = 1.2  # 1.2 segundos para disparar captura

        # Controles de rotação 3D para a tela de revisão
        self.review_yaw: float = 18.0
        self.review_pitch: float = -12.0

    def run_interactive(self) -> bool:
        """Loop interativo com janela OpenCV + renderização tipográfica Pillow."""
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir a câmera índice {self.camera_idx}.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Título em ASCII para evitar quebra de codificação no frame da janela do Windows
        window_name = "Calibrador Biomecanico Guiado - LIBRAS TCC"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*70)
        print("  CALIBRADOR BIOMECÂNICO GUIADO INICIADO COM SUCESSO")
        print("="*70)
        print("Fluxo de Operação:")
        print("  1. Posicione a mão conforme as instruções detalhadas na tela.")
        print("  2. Segure estável por 1.2s (ou aperte [ESPAÇO]) para capturar.")
        print("  3. Na tela de REVISÃO, confira os dados extraídos dos pontos.")
        print("  4. Pressione [ESPAÇO] para confirmar ou [R] para refazer o passo.")
        print("="*70 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Modo espelho
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            has_hand = False
            pts_norm = None
            pts_pixels = None

            if results.multi_hand_landmarks:
                has_hand = True
                lm_list = results.multi_hand_landmarks[0].landmark
                pts_pixels = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lm_list])

                wrist = pts_pixels[0]
                palm_len = np.linalg.norm(pts_pixels[9] - wrist)
                if palm_len > 1e-5:
                    pts_norm = (pts_pixels - wrist) / palm_len

                # Desenhar skeleton com MediaPipe no modo captura
                if self.state == "CAPTURING":
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[0],
                        self.mp_hands.HAND_CONNECTIONS
                    )

            step = CALIBRATION_STEPS[self.current_step_idx]

            # ---------------------------------------------------------------
            # MODO 1: CAPTURA EM TEMPO REAL (FRONTAL E LATERAL)
            # ---------------------------------------------------------------
            if self.state == "CAPTURING":
                # Lógica de estabilidade temporal
                if has_hand and pts_norm is not None:
                    if self.stability_start_time is None:
                        self.stability_start_time = time.time()
                        self.stable_frame_buffer = [pts_norm]
                    else:
                        self.stable_frame_buffer.append(pts_norm)
                        elapsed = time.time() - self.stability_start_time
                        if elapsed >= self.REQUIRED_STABLE_TIME and len(self.stable_frame_buffer) >= 20:
                            if self.current_sub_angle == "FRONTAL":
                                self._record_sub_capture(step, frame, pts_pixels, pts_norm, angle="FRONTAL")
                                self.current_sub_angle = "LATERAL"
                                self.stability_start_time = None
                                self.stable_frame_buffer = []
                                print(f"  ✓ Ângulo Frontal capturado com sucesso! Agora gire a mão de PERFIL (90° lateral)...")
                            else:
                                self._record_sub_capture(step, frame, pts_pixels, pts_norm, angle="LATERAL")
                                self._finalize_step_and_review(step)
                else:
                    self.stability_start_time = None
                    self.stable_frame_buffer = []

                # Renderizar HUD de Captura
                frame = self._render_capturing_hud(frame, step, has_hand, pts_norm)

            # ---------------------------------------------------------------
            # MODO 2: REVISÃO E CONFIRMAÇÃO DOS DADOS EXTRAÍDOS
            # ---------------------------------------------------------------
            elif self.state == "REVIEW":
                # Renderizar tela de revisão sobreposta ao frame congelado da captura
                bg_frame = self.current_review_snapshot.copy() if self.current_review_snapshot is not None else frame
                frame = self._render_review_hud(bg_frame, step)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # Tratar teclas de controle
            if key in [ord('q'), 27]:  # ESC ou Q
                print("\n[LOG] Calibração interrompida pelo usuário.")
                break

            elif key in [32, 13]:  # ESPAÇO ou ENTER
                if self.state == "CAPTURING":
                    if has_hand and pts_norm is not None:
                        self.stable_frame_buffer.append(pts_norm)
                        if self.current_sub_angle == "FRONTAL":
                            self._record_sub_capture(step, frame, pts_pixels, pts_norm, angle="FRONTAL")
                            self.current_sub_angle = "LATERAL"
                            self.stability_start_time = None
                            self.stable_frame_buffer = []
                            print(f"  ✓ Ângulo Frontal capturado manualmente! Agora gire a mão de PERFIL (90° lateral)...")
                        else:
                            self._record_sub_capture(step, frame, pts_pixels, pts_norm, angle="LATERAL")
                            self._finalize_step_and_review(step)
                elif self.state == "REVIEW":
                    # Usuário confirmou os dados extraídos -> Avança!
                    print(f"  ✓ Passo [{self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}] confirmado pelo usuário.")
                    self._advance_step()

            elif key == ord('r'):  # Recapturar / Repetir
                step_id = step['id']
                if step_id in self.captured_data:
                    del self.captured_data[step_id]
                self.step_subcaptures = {}
                self.current_sub_angle = "FRONTAL"
                self.state = "CAPTURING"
                self.stability_start_time = None
                self.stable_frame_buffer = []
                print(f"[REFAZER] Reiniciando captura completa (Frontal + Lateral) de: {step['title']}")

            elif key == ord('b'):  # Voltar passo anterior
                if self.current_step_idx > 0:
                    self.current_step_idx -= 1
                    self.step_subcaptures = {}
                    self.current_sub_angle = "FRONTAL"
                    self.state = "CAPTURING"
                    self.stability_start_time = None
                    self.stable_frame_buffer = []
                    prev_step = CALIBRATION_STEPS[self.current_step_idx]
                    print(f"[VOLTAR] Retornando ao passo anterior: {prev_step['title']}")

            elif key == ord('s'):  # Pular passo
                print(f"[PULAR] Passo '{step['title']}' pulado com valores canônicos.")
                self._advance_step()

            elif self.state == "REVIEW":
                # Controles de rotação 3D do modelo da mão na tela de revisão
                if key in [ord('a'), ord('A'), 81, 2424832]:  # Girar para a esquerda
                    self.review_yaw -= 8.0
                elif key in [ord('d'), ord('D'), 83, 2555904]:  # Girar para a direita
                    self.review_yaw += 8.0
                elif key in [ord('w'), ord('W'), 82, 2490368]:  # Inclinar para cima
                    self.review_pitch = min(85.0, self.review_pitch + 6.0)
                elif key in [ord('x'), ord('X'), 84, 2621440]:  # Inclinar para baixo
                    self.review_pitch = max(-85.0, self.review_pitch - 6.0)

            if self.current_step_idx >= len(CALIBRATION_STEPS):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Compilar e salvar configurações APENAS se todos os passos foram concluídos
        if self.current_step_idx >= len(CALIBRATION_STEPS):
            print(f"\n[SUCESSO] Todos os {len(CALIBRATION_STEPS)} passos da calibração foram concluídos com êxito!")
            print("[COMPILAÇÃO] Processando limites anatômicos e gerando catálogo de seeds...")
            self.compile_and_save_settings()
            return True
        else:
            print(f"\n[INFO] Calibração interrompida no passo {self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}.")
            print("[INFO] As configurações e seeds anteriores foram mantidas intactas sem alterações.")
            return False

    def _record_sub_capture(
        self,
        step: Dict[str, Any],
        frame: np.ndarray,
        pts_raw: Optional[np.ndarray],
        pts_norm: Optional[np.ndarray],
        angle: str
    ):
        """Salva a imagem do sub-ângulo (frontal ou lateral) e guarda os landmarks correspondentes."""
        step_id = step['id']
        angle_key = angle.lower()
        img_filename = os.path.join(CAPTURES_DIR, f"{self.current_step_idx+1:02d}_{step_id}_{angle_key}.png")

        if len(self.stable_frame_buffer) > 0:
            avg_pts = np.mean(np.array(self.stable_frame_buffer), axis=0)
        elif pts_norm is not None:
            avg_pts = pts_norm
        elif pts_raw is not None and len(pts_raw) == 21:
            wrist = pts_raw[0]
            palm_len = np.linalg.norm(pts_raw[9] - wrist)
            avg_pts = (pts_raw - wrist) / palm_len if palm_len > 1e-5 else pts_raw
        else:
            avg_pts = np.zeros((21, 3))

        # Criar snapshot anotado com marcação do ângulo
        annotated_snapshot = frame.copy()
        if pts_raw is not None and len(pts_raw) == 21:
            target_f = step.get('target_finger')
            target_fingers = ['Index', 'Middle', 'Ring', 'Pinky'] if target_f == 'FourFingers' else ([target_f] if target_f in FINGER_JOINTS else [])
            for tf in target_fingers:
                idxs = FINGER_JOINTS[tf]
                for i in range(1, len(idxs)):
                    p1 = (int(pts_raw[idxs[i-1]][0]), int(pts_raw[idxs[i-1]][1]))
                    p2 = (int(pts_raw[idxs[i]][0]), int(pts_raw[idxs[i]][1]))
                    cv2.line(annotated_snapshot, p1, p2, (255, 230, 80), 4, cv2.LINE_AA)
                for i, j_idx in enumerate(idxs):
                    pt = (int(pts_raw[j_idx][0]), int(pts_raw[j_idx][1]))
                    cv2.circle(annotated_snapshot, pt, 7, (50, 255, 120), -1, cv2.LINE_AA)
                    cv2.circle(annotated_snapshot, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)

        # Adicionar badge do ângulo no canto da foto salva
        badge_text = f"ANGULO: {angle.upper()}"
        cv2.rectangle(annotated_snapshot, (10, 10), (220, 38), (20, 20, 30), -1)
        cv2.rectangle(annotated_snapshot, (10, 10), (220, 38), (137, 180, 250), 1)
        cv2.putText(annotated_snapshot, badge_text, (18, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (137, 180, 250), 1, cv2.LINE_AA)

        cv2.imwrite(img_filename, annotated_snapshot)

        self.step_subcaptures[angle_key] = {
            'pts_norm': avg_pts,
            'pts_raw': pts_raw if pts_raw is not None else np.zeros((21, 3)),
            'image_path': img_filename,
            'snapshot': annotated_snapshot
        }

    def _finalize_step_and_review(self, step: Dict[str, Any]):
        """Consolida as capturas dos dois ângulos (frontal e lateral) e entra em modo de revisão."""
        step_id = step['id']
        frontal_data = self.step_subcaptures.get('frontal', {})
        lateral_data = self.step_subcaptures.get('lateral', {})

        # Para flexão dos dedos longos, prioriza dados laterais se disponíveis
        pts_lat = lateral_data.get('pts_norm')
        pts_front = frontal_data.get('pts_norm')

        if step.get('category') == 'four_fingers_flexion' and pts_lat is not None:
            pts_eval = pts_lat
        else:
            pts_eval = pts_front if pts_front is not None else (pts_lat if pts_lat is not None else np.zeros((21, 3)))

        self.captured_data[step_id] = {
            'step_meta': step,
            'frontal': frontal_data,
            'lateral': lateral_data,
            'pts_norm': pts_eval,
            'pts_raw': frontal_data.get('pts_raw'),
            'image_path': frontal_data.get('image_path')
        }

        # Formatar métricas de revisão
        self.current_review_metrics, self.current_review_points, self.current_review_status = self._format_metrics_for_step(
            step, pts_eval, frontal_pts=pts_front, lateral_pts=pts_lat
        )

        self.current_review_snapshot = frontal_data.get('snapshot')
        self.review_yaw = 18.0
        self.review_pitch = -12.0
        self.state = "REVIEW"

        print(f"\n[CAPTURA CONCLUÍDA (FRONTAL + LATERAL)] -> {step['title']}")
        print(f"Status: {self.current_review_status}")
        for line in self.current_review_metrics:
            print(f"   {line}")
        print("-> Pressione [ESPAÇO] ou [ENTER] para Confirmar e Avançar, ou [R] para Refazer...")

    def _advance_step(self):
        """Avança para o próximo passo da calibração."""
        self.current_step_idx += 1
        self.current_sub_angle = "FRONTAL"
        self.step_subcaptures = {}
        self.state = "CAPTURING"
        self.stability_start_time = None
        self.stable_frame_buffer = []
        if self.current_step_idx < len(CALIBRATION_STEPS):
            next_step = CALIBRATION_STEPS[self.current_step_idx]
            print(f"\n=======================================================")
            print(f"Iniciando Passo [{self.current_step_idx+1}/{len(CALIBRATION_STEPS)}]: {next_step['title']}")
            print(f"Etapa 1/2: ÂNGULO FRONTAL (Palma voltada de frente para a câmera)")
            print(f"Instrução: {next_step['target_action']}")
            print(f"=======================================================")

    # -----------------------------------------------------------------------
    # EXTRAÇÃO E FORMATAÇÃO DE MÉTRICAS ANATÔMICAS
    # -----------------------------------------------------------------------

    def _format_metrics_for_step(
        self,
        step: Dict[str, Any],
        pts_norm: np.ndarray,
        frontal_pts: Optional[np.ndarray] = None,
        lateral_pts: Optional[np.ndarray] = None
    ) -> Tuple[List[str], List[str], str]:
        """Calcula os ângulos, distâncias e coordenadas 3D dos pontos extraídos do frame capturado."""
        cat = step['category']
        metrics = []
        points_info = []
        status_conformity = "Adequado"

        p_front = frontal_pts if frontal_pts is not None else pts_norm
        p_lat = lateral_pts if lateral_pts is not None else pts_norm

        if cat == 'baseline':
            p_wrist = p_front[0]
            p_mcp9 = p_front[9]
            palm_len = np.linalg.norm(p_mcp9 - p_wrist)
            metrics.append("Escala da Palma (Pulso -> Metacarpo Médio): 1.000")
            status_conformity = "Excelente (Mão Espalmada Detectada)"

            for fname in ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']:
                idxs = FINGER_JOINTS[fname]
                l1 = float(np.linalg.norm(p_front[idxs[2]] - p_front[idxs[1]]))
                l2 = float(np.linalg.norm(p_front[idxs[3]] - p_front[idxs[2]]))
                l3 = float(np.linalg.norm(p_front[idxs[4]] - p_front[idxs[3]]))
                metrics.append(f"{fname:7s}: Falanges = [{l1:.2f}, {l2:.2f}, {l3:.2f}] | Total = {l1+l2+l3:.2f}")

            points_info.append(f"Pulso (ID 0):   [{p_front[0][0]:+.2f}, {p_front[0][1]:+.2f}, {p_front[0][2]:+.2f}]")
            points_info.append(f"MCP Indicador:  [{p_front[5][0]:+.2f}, {p_front[5][1]:+.2f}, {p_front[5][2]:+.2f}]")
            points_info.append(f"MCP Médio (9):  [{p_front[9][0]:+.2f}, {p_front[9][1]:+.2f}, {p_front[9][2]:+.2f}]")
            points_info.append(f"MCP Mindinho:   [{p_front[17][0]:+.2f}, {p_front[17][1]:+.2f}, {p_front[17][2]:+.2f}]")
            points_info.append(f"Largura Palma:  {float(np.linalg.norm(p_front[5] - p_front[17])):.2f} (unidades norm)")

        elif cat == 'four_fingers_flexion':
            st = step['expected_stage']
            stage_names = {
                0: "0 (Estendidos / Retos)", 1: "1 (Curvados / Concha)",
                2: "2 (Ganchos / Garras)", 3: "3 (Plataforma / Tabletop)", 4: "4 (Fechados / Punho)"
            }
            metrics.append(f"4 DEDOS LONGOS | Estágio: {stage_names.get(st, str(st))}")
            if lateral_pts is not None:
                metrics.append("• Fonte: Perfil Lateral (Sem distorção de profundidade Z)")

            finger_angles = {}
            pt_names = {'Index': 'Indicador', 'Middle': 'Médio', 'Ring': 'Anelar', 'Pinky': 'Mínimo'}
            for fname in ['Index', 'Middle', 'Ring', 'Pinky']:
                idxs = FINGER_JOINTS[fname]
                j2 = joint_flexion(p_lat[idxs[0]], p_lat[idxs[1]], p_lat[idxs[2]])
                j3 = joint_flexion(p_lat[idxs[1]], p_lat[idxs[2]], p_lat[idxs[3]])
                j4 = joint_flexion(p_lat[idxs[2]], p_lat[idxs[3]], p_lat[idxs[4]])
                finger_angles[fname] = (j2, j3, j4)
                metrics.append(f"• {pt_names[fname]:9s}: MCP {j2:4.1f}° | PIP {j3:4.1f}° | DIP {j4:4.1f}°")

            mean_j2 = float(np.mean([fa[0] for fa in finger_angles.values()]))
            mean_j3 = float(np.mean([fa[1] for fa in finger_angles.values()]))
            mean_j4 = float(np.mean([fa[2] for fa in finger_angles.values()]))

            if st == 0:
                mcp_exp, pip_exp, dip_exp = "0° - 20°", "0° - 20°", "0° - 20°"
                status_conformity = "Excelente (Dedos Retos)" if mean_j2 < 25 and mean_j3 < 25 else "Ajuste: Estique mais os 4 dedos"
            elif st == 1:
                mcp_exp, pip_exp, dip_exp = "20° - 40°", "30° - 55°", "25° - 45°"
                status_conformity = "Excelente (Arco Concha Suave)" if 15 < mean_j2 < 55 and 25 < mean_j3 < 65 else "Ajuste suave"
            elif st == 2:
                mcp_exp, pip_exp, dip_exp = "0° - 25°", "75° - 105°", "60° - 90°"
                status_conformity = "Excelente (Garras / Ganchos)" if mean_j2 < 30 and mean_j3 > 65 else "Aviso: Mantenha base reta e dobre as pontas"
            elif st == 3:
                mcp_exp, pip_exp, dip_exp = "75° - 100°", "0° - 25°", "0° - 25°"
                status_conformity = "Excelente (Plataforma / Mesa)" if mean_j2 > 60 and mean_j3 < 30 else "Aviso: Dobre base a 90° e pontas retas"
            else:  # st == 4
                mcp_exp, pip_exp, dip_exp = "75° - 105°", "90° - 120°", "65° - 95°"
                status_conformity = "Excelente (Punho Fechado)" if mean_j2 > 65 and mean_j3 > 75 else "Aviso: Feche bem os dedos contra a palma"

            metrics.append(f"• Média Geral: MCP {mean_j2:4.1f}° | PIP {mean_j3:4.1f}° | DIP {mean_j4:4.1f}°")
            metrics.append(f"• Avaliação: {status_conformity}")

            points_info.append(f"MCP Indicador: [{p_lat[5][0]:+.2f}, {p_lat[5][1]:+.2f}, {p_lat[5][2]:+.2f}]")
            points_info.append(f"MCP Médio:     [{p_lat[9][0]:+.2f}, {p_lat[9][1]:+.2f}, {p_lat[9][2]:+.2f}]")
            points_info.append(f"MCP Anelar:    [{p_lat[13][0]:+.2f}, {p_lat[13][1]:+.2f}, {p_lat[13][2]:+.2f}]")
            points_info.append(f"MCP Mindinho:  [{p_lat[17][0]:+.2f}, {p_lat[17][1]:+.2f}, {p_lat[17][2]:+.2f}]")

        elif cat == 'finger_flexion':
            f_name = step['finger']
            idxs = FINGER_JOINTS[f_name]
            j2_p = joint_flexion(p_lat[idxs[0]], p_lat[idxs[1]], p_lat[idxs[2]])
            j3_p = joint_flexion(p_lat[idxs[1]], p_lat[idxs[2]], p_lat[idxs[3]])
            j4_p = joint_flexion(p_lat[idxs[2]], p_lat[idxs[3]], p_lat[idxs[4]])

            st = step['expected_stage']
            stage_names = {
                0: "0 (Estendido / Reto)", 1: "1 (Curvado / Concha)",
                2: "2 (Gancho / Hook)", 3: "3 (Plataforma / Tabletop)", 4: "4 (Fechado / Punho)"
            }
            metrics.append(f"Dedo: {f_name.upper()} | Estágio: {stage_names.get(st, str(st))}")
            if lateral_pts is not None:
                metrics.append("• Fonte: Perfil Lateral")

            # Avaliação de faixas ideais
            if st == 0:
                mcp_exp, pip_exp, dip_exp = "0° - 20°", "0° - 20°", "0° - 20°"
                status_conformity = "Excelente (Reto)" if j2_p < 25 and j3_p < 25 else "Aviso: Dedo parece ligeiramente flexionado"
            elif st == 1:
                mcp_exp, pip_exp, dip_exp = "20° - 40°", "30° - 55°", "25° - 45°"
                status_conformity = "Excelente (Curvatura suave)" if 15 < j2_p < 55 and 25 < j3_p < 65 else "Ajuste suave"
            elif st == 2:
                mcp_exp, pip_exp, dip_exp = "0° - 25°", "75° - 105°", "60° - 90°"
                status_conformity = "Excelente (Gancho / Garra)" if j2_p < 30 and j3_p > 65 else "Aviso: Mantenha MCP reta e dobre pontas"
            elif st == 3:
                mcp_exp, pip_exp, dip_exp = "75° - 100°", "0° - 25°", "0° - 25°"
                status_conformity = "Excelente (Plataforma / Mesa)" if j2_p > 65 and j3_p < 30 else "Aviso: Mantenha base em 90° e ponta reta"
            else:  # st == 4
                mcp_exp, pip_exp, dip_exp = "75° - 105°", "90° - 120°", "65° - 95°"
                status_conformity = "Excelente (Cerrado na palma)" if j2_p > 65 and j3_p > 75 else "Aviso: Dobre mais o dedo contra a palma"

            metrics.append(f"• MCP (Base palma):  {j2_p:5.1f}°  [Esperado: {mcp_exp}]")
            metrics.append(f"• PIP (Junta média): {j3_p:5.1f}°  [Esperado: {pip_exp}]")
            metrics.append(f"• DIP (Ponta dedo):  {j4_p:5.1f}°  [Esperado: {dip_exp}]")
            metrics.append(f"• Avaliação: {status_conformity}")

            # Coordenadas dos pontos extraídos
            points_info.append(f"MCP (Ponto {idxs[1]}): [{p_lat[idxs[1]][0]:+.2f}, {p_lat[idxs[1]][1]:+.2f}, {p_lat[idxs[1]][2]:+.2f}]")
            points_info.append(f"PIP (Ponto {idxs[2]}): [{p_lat[idxs[2]][0]:+.2f}, {p_lat[idxs[2]][1]:+.2f}, {p_lat[idxs[2]][2]:+.2f}]")
            points_info.append(f"DIP (Ponto {idxs[3]}): [{p_lat[idxs[3]][0]:+.2f}, {p_lat[idxs[3]][1]:+.2f}, {p_lat[idxs[3]][2]:+.2f}]")
            points_info.append(f"TIP (Ponta {idxs[4]}): [{p_lat[idxs[4]][0]:+.2f}, {p_lat[idxs[4]][1]:+.2f}, {p_lat[idxs[4]][2]:+.2f}]")
            dist_tip_wrist = float(np.linalg.norm(p_lat[idxs[4]] - p_lat[0]))
            points_info.append(f"Distância Ponta -> Pulso: {dist_tip_wrist:.2f}")

        elif cat == 'spread':
            sp_pnk_rng = vec_angle(p_front[17] - p_front[0], p_front[13] - p_front[0])
            sp_rng_mid = vec_angle(p_front[13] - p_front[0], p_front[9] - p_front[0])
            sp_mid_idx = vec_angle(p_front[9] - p_front[0], p_front[5] - p_front[0])
            sp_idx_thm = vec_angle(p_front[5] - p_front[0], p_front[1] - p_front[0])
            is_open = step['spread_state'] == 0
            mode_str = "Leque Máximo (A=0 Aberto)" if is_open else "Dedos Paralelos (A=1 Fechado)"
            status_conformity = "Excelente (Abertura Capturada)" if (is_open and sp_rng_mid > 12) or (not is_open and sp_rng_mid < 10) else "Concluído"

            metrics.append(f"Configuração: {mode_str}")
            metrics.append(f"• Mindinho - Anelar:   {sp_pnk_rng:5.1f}°")
            metrics.append(f"• Anelar - Médio:      {sp_rng_mid:5.1f}°")
            metrics.append(f"• Médio - Indicador:   {sp_mid_idx:5.1f}°")
            metrics.append(f"• Indicador - Polegar: {sp_idx_thm:5.1f}°")

            points_info.append(f"Ponta Mindinho:  [{p_front[20][0]:+.2f}, {p_front[20][1]:+.2f}, {p_front[20][2]:+.2f}]")
            points_info.append(f"Ponta Anelar:    [{p_front[16][0]:+.2f}, {p_front[16][1]:+.2f}, {p_front[16][2]:+.2f}]")
            points_info.append(f"Ponta Médio:     [{p_front[12][0]:+.2f}, {p_front[12][1]:+.2f}, {p_front[12][2]:+.2f}]")
            points_info.append(f"Ponta Indicador: [{p_front[8][0]:+.2f}, {p_front[8][1]:+.2f}, {p_front[8][2]:+.2f}]")
            span = float(np.linalg.norm(p_front[20] - p_front[4]))
            points_info.append(f"Envergadura Total (Polegar-Mindinho): {span:.2f}")

        elif cat == 'thumb':
            dist_opp = float(np.linalg.norm(p_front[4] - p_front[9]))
            ip_flex = joint_flexion(p_lat[2], p_lat[3], p_lat[4])
            mcp_flex = joint_flexion(p_front[1], p_front[2], p_front[3])
            f_label = "No Plano da Palma (F=0)" if step['f'] == 0 else "Oposição Transversal (F=1)"
            p_label = "Ponta Reta (P=0)" if step['p'] == 0 else "Ponta Dobrada (P=1)"
            status_conformity = "Excelente (Polegar Capturado)"

            metrics.append(f"Modo: {f_label} | {p_label}")
            metrics.append(f"• Flexão da Ponta (IP):       {ip_flex:5.1f}°")
            metrics.append(f"• Flexão da Base (MCP):       {mcp_flex:5.1f}°")
            metrics.append(f"• Distância Ponta -> Palma:   {dist_opp:5.2f}")

            points_info.append(f"CMC (Ponto 1): [{p_front[1][0]:+.2f}, {p_front[1][1]:+.2f}, {p_front[1][2]:+.2f}]")
            points_info.append(f"MCP (Ponto 2): [{p_front[2][0]:+.2f}, {p_front[2][1]:+.2f}, {p_front[2][2]:+.2f}]")
            points_info.append(f"IP  (Ponto 3): [{p_lat[3][0]:+.2f}, {p_lat[3][1]:+.2f}, {p_lat[3][2]:+.2f}]")
            points_info.append(f"TIP (Ponto 4): [{p_lat[4][0]:+.2f}, {p_lat[4][1]:+.2f}, {p_lat[4][2]:+.2f}]")
            points_info.append(f"Profundidade Z da Ponta: {pts_norm[4][2]:+.2f}")

        return metrics, points_info, status_conformity

    # -----------------------------------------------------------------------
    # RENDERIZAÇÃO GRÁFICA (HUD COM SUPORTE TOTAL A UTF-8)
    # -----------------------------------------------------------------------

    def _render_capturing_hud(self, frame: np.ndarray, step: Dict[str, Any], has_hand: bool, pts_norm: Optional[np.ndarray]) -> np.ndarray:
        h, w, _ = frame.shape

        # Converter para PIL RGBA para desenhar texto anti-aliased e transparências
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Configurar textos e orientações baseados no sub-ângulo ativo (Frontal vs Lateral)
        if self.current_sub_angle == "FRONTAL":
            angle_badge = "[ ETAPA 1/2: ÂNGULO FRONTAL (PALMA DE FRENTE) ]"
            badge_color = (166, 227, 161, 255)  # Verde Esmeralda
            posture_txt = step['posture']
            action_txt = step['target_action']
            summary_txt = step.get('expected_summary', '')
        else:  # LATERAL
            angle_badge = "[ ETAPA 2/2: ÂNGULO LATERAL (PERFIL DA MÃO A 90°) ]"
            badge_color = (249, 226, 175, 255)  # Âmbar Ouro
            posture_txt = "Gire o punho 90° de lado (mão de perfil para a câmera, com o polegar voltado para a lente ou para cima)."
            action_txt = "Mantenha a MESMA pose dos dedos! A câmera registrará a curvatura real das juntas de perfil (sem encurtamento do eixo Z)."
            summary_txt = "Curvatura articular 100% visível no plano da imagem lateral"

        # Quebra inteligente de texto sem cortes
        wrap_width = 110 if w >= 1100 else 75
        action_lines = textwrap.wrap(action_txt, width=wrap_width)
        posture_lines = textwrap.wrap(posture_txt, width=wrap_width)
        others_lines = textwrap.wrap(step['other_fingers'], width=wrap_width) if self.current_sub_angle == "FRONTAL" else []

        # Altura dinâmica calculada com base no conteúdo para nunca cortar
        line_height = 20
        total_text_lines = len(posture_lines) + len(action_lines) + (1 if summary_txt else 0) + len(others_lines)
        top_h = max(135, 54 + total_text_lines * line_height + 12)

        # 1. Painel Superior Responsivo
        draw.rectangle([(0, 0), (w, top_h)], fill=(17, 17, 27, 240))
        draw.line([(0, top_h), (w, top_h)], fill=(137, 180, 250, 255), width=2)

        # Cabeçalho do passo
        f_title = self.hud.get_font(19 if w >= 1000 else 16, bold=True)
        f_body = self.hud.get_font(13 if w >= 1000 else 11, bold=False)
        f_body_bold = self.hud.get_font(13 if w >= 1000 else 11, bold=True)
        f_highlight = self.hud.get_font(13 if w >= 1000 else 11, bold=True)
        f_small = self.hud.get_font(12 if w >= 1000 else 10, bold=False)

        step_num_str = f"PASSO {self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}"
        draw.text((25, 8), f"{step_num_str}  —  {angle_badge}", font=self.hud.get_font(13, bold=True), fill=badge_color)
        draw.text((25, 26), step['title'], font=f_title, fill=(137, 180, 250, 255))

        curr_y = 52
        for i, pline in enumerate(posture_lines):
            prefix = "Posição: " if i == 0 else "         "
            draw.text((25, curr_y), prefix + pline, font=f_body, fill=(205, 214, 244, 255))
            curr_y += line_height

        for i, aline in enumerate(action_lines):
            prefix = "Ação:    " if i == 0 else "         "
            draw.text((25, curr_y), prefix + aline, font=f_body_bold, fill=(166, 227, 161, 255))
            curr_y += line_height

        if summary_txt:
            draw.text((25, curr_y), f"Alvo:    {summary_txt}", font=f_highlight, fill=(249, 226, 175, 255))
            curr_y += line_height

        for i, oline in enumerate(others_lines):
            prefix = "Outros:  " if i == 0 else "         "
            draw.text((25, curr_y), prefix + oline, font=f_small, fill=(186, 194, 222, 255))
            curr_y += line_height

        # 2. Painel Inferior (Barra de comandos)
        bot_h = 65
        draw.rectangle([(0, h - bot_h), (w, h)], fill=(17, 17, 27, 240))
        draw.line([(0, h - bot_h), (w, h - bot_h)], fill=(69, 71, 90, 255), width=1)

        f_ctrl = self.hud.get_font(14, bold=True)
        f_ctrl_sub = self.hud.get_font(12, bold=False)
        ctrl_str = "[ESPAÇO] Capturar Ângulo Atual  |  [R] Repetir Passo  |  [B] Voltar  |  [S] Pular  |  [Q / ESC] Sair sem Salvar"
        draw.text((25, h - 45), ctrl_str, font=f_ctrl, fill=(245, 194, 231, 255))
        draw.text((25, h - 24), "Segure a pose estável por 1.2 segundos para disparo automático com média temporal.", font=f_ctrl_sub, fill=(186, 194, 222, 255))

        # 3. Telemetria ao Vivo (Lado direito)
        if has_hand and pts_norm is not None:
            live_metrics, _, _ = self._format_metrics_for_step(step, pts_norm)
            card_w = 340 if w >= 1100 else 280
            card_h = 30 + len(live_metrics) * 22
            cx = w - card_w - 20
            cy = top_h + 15

            draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], fill=(24, 24, 37, 215))
            draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], outline=(137, 180, 250, 255), width=1)

            f_card_hdr = self.hud.get_font(13, bold=True)
            f_card_body = self.hud.get_font(12, bold=False)
            draw.text((cx + 12, cy + 8), "TELEMETRIA AO VIVO (ÂNGULOS)", font=f_card_hdr, fill=(249, 226, 175, 255))

            for i, line in enumerate(live_metrics):
                draw.text((cx + 12, cy + 30 + i * 22), line, font=f_card_body, fill=(205, 214, 244, 255))

            # Barra de progresso de estabilização
            if self.stability_start_time is not None:
                elapsed = time.time() - self.stability_start_time
                pct = min(1.0, elapsed / self.REQUIRED_STABLE_TIME)
                bar_w = 340
                bar_h = 16
                bx = (w - bar_w) // 2
                by = h - bot_h - 32

                draw.rectangle([(bx, by), (bx + bar_w, by + bar_h)], fill=(40, 40, 60, 220))
                draw.rectangle([(bx, by), (bx + int(bar_w * pct), by + bar_h)], fill=(166, 227, 161, 255))
                draw.rectangle([(bx, by), (bx + bar_w, by + bar_h)], outline=(205, 214, 244, 255), width=1)

                f_bar = self.hud.get_font(13, bold=True)
                bar_txt = f"ESTABILIZANDO POSE: {int(pct * 100)}%"
                draw.text((bx + 85, by - 20), bar_txt, font=f_bar, fill=(166, 227, 161, 255))
        else:
            f_warn = self.hud.get_font(18, bold=True)
            msg = "AGUARDANDO DETECÇÃO DA MÃO NA TELA..."
            draw.text(((w // 2) - 200, h // 2), msg, font=f_warn, fill=(243, 139, 168, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _get_hand_crop_roi(self, snapshot: np.ndarray, pts_raw: Optional[np.ndarray], target_w: int, target_h: int, title: str = "CAMERA REAL (RASTREAMENTO)") -> np.ndarray:
        """
        Recorta a região da mão do snapshot da câmera mantendo proporção e adicionando bordas/chrome.
        """
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = (20, 18, 28)

        # Se houver coordenadas brutas em pixels, calcula a bounding box com margem de respiro
        if pts_raw is not None and len(pts_raw) == 21:
            sh, sw, _ = snapshot.shape
            x_coords = pts_raw[:, 0]
            y_coords = pts_raw[:, 1]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

            bw = max(10, x_max - x_min)
            bh = max(10, y_max - y_min)
            side = int(max(bw, bh) * 1.35)  # 35% de margem ao redor da mão

            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(sw, x1 + side)
            y2 = min(sh, y1 + side)

            actual_w = x2 - x1
            actual_h = y2 - y1
            if actual_w > 10 and actual_h > 10:
                crop = snapshot[y1:y2, x1:x2]
                inner_h = target_h - 26
                inner_w = target_w - 4
                resized = cv2.resize(crop, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
                canvas[24:24 + inner_h, 2:2 + inner_w] = resized
        else:
            inner_h = target_h - 26
            inner_w = target_w - 4
            resized = cv2.resize(snapshot, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
            canvas[24:24 + inner_h, 2:2 + inner_w] = resized

        # Barra de título superior
        cv2.rectangle(canvas, (0, 0), (target_w, 24), (28, 26, 38), -1)
        cv2.line(canvas, (0, 24), (target_w, 24), (166, 227, 161), 1)
        cv2.putText(canvas, title, (10, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (166, 227, 161), 1, cv2.LINE_AA)

        # Borda externa do viewport
        cv2.rectangle(canvas, (0, 0), (target_w - 1, target_h - 1), (166, 227, 161), 1)
        return canvas

    def _render_hand_model_3d(
        self,
        pts_norm: Optional[np.ndarray],
        step: Dict[str, Any],
        vp_w: int,
        vp_h: int,
        yaw: float = 18.0,
        pitch: float = -12.0,
        title: str = "3. 3D DIGITAL (CODIGO)"
    ) -> np.ndarray:
        """
        Renderiza o modelo biomecânico 3D digital da mão a partir dos landmarks normalizados lidos pelo código.
        """
        canvas = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
        canvas[:] = (20, 18, 28)

        cx = vp_w // 2
        scale = min(vp_w, vp_h) * 0.48
        cy = int(vp_h * 0.52 + 0.95 * scale)

        R = rot_x(pitch).dot(rot_y(yaw))

        # 1. Grade 3D isométrica no plano de referência do pulso
        grid_y = 0.12
        grid_col = (35, 32, 45)
        for gx in np.linspace(-0.7, 0.7, 5):
            p1 = np.array([gx, grid_y, -0.6]).dot(R.T)
            p2 = np.array([gx, grid_y, +0.6]).dot(R.T)
            x1, y1 = int(cx + p1[0] * scale), int(cy + p1[1] * scale)
            x2, y2 = int(cx + p2[0] * scale), int(cy + p2[1] * scale)
            cv2.line(canvas, (x1, y1), (x2, y2), grid_col, 1, cv2.LINE_AA)
        for gz in np.linspace(-0.6, 0.6, 5):
            p1 = np.array([-0.7, grid_y, gz]).dot(R.T)
            p2 = np.array([+0.7, grid_y, gz]).dot(R.T)
            x1, y1 = int(cx + p1[0] * scale), int(cy + p1[1] * scale)
            x2, y2 = int(cx + p2[0] * scale), int(cy + p2[1] * scale)
            cv2.line(canvas, (x1, y1), (x2, y2), grid_col, 1, cv2.LINE_AA)

        if pts_norm is None or len(pts_norm) != 21:
            cv2.putText(canvas, "SEM DADOS 3D", (cx - 55, vp_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 160), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (0, 0), (vp_w - 1, vp_h - 1), (137, 180, 250), 1)
            return canvas

        pts_clean = np.nan_to_num(pts_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Rotação e projeção dos 21 landmarks para o espaço de tela
        pts_rot = pts_clean.dot(R.T)
        screen_pts = []
        depths = []
        for i in range(21):
            sx = int(cx + pts_rot[i, 0] * scale)
            sy = int(cy + pts_rot[i, 1] * scale)
            screen_pts.append((sx, sy))
            depths.append(pts_rot[i, 2])

        # 3. Desenho do polígono semi-transparente da palma (Mesh da mão)
        palm_poly_indices = [0, 1, 5, 9, 13, 17]
        palm_pts = np.array([screen_pts[i] for i in palm_poly_indices], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [palm_pts], (48, 42, 58))
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # 4. Desenho dos ossos da palma
        for i1, i2 in PALM_BONES:
            p1, p2 = screen_pts[i1], screen_pts[i2]
            avg_z = (depths[i1] + depths[i2]) / 2.0
            shade = np.clip(1.0 - (avg_z * 0.35), 0.6, 1.3)
            col = (int(130 * shade), int(125 * shade), int(145 * shade))
            cv2.line(canvas, p1, p2, col, 2, cv2.LINE_AA)

        # 5. Desenho dos ossos dos dedos com cores anatômicas
        target_f = step.get('target_finger')
        is_four_fingers = (target_f == 'FourFingers' or step.get('category') == 'four_fingers_flexion')
        for fname in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            is_target = (target_f == fname) or (is_four_fingers and fname in ['Index', 'Middle', 'Ring', 'Pinky'])
            base_col = FINGER_COLORS_BGR[fname]
            segments = FINGER_SEGMENTS[fname]

            for i1, i2 in segments:
                p1, p2 = screen_pts[i1], screen_pts[i2]
                avg_z = (depths[i1] + depths[i2]) / 2.0
                shade = np.clip(1.0 - (avg_z * 0.35), 0.65, 1.35)
                bone_col = (
                    int(np.clip(base_col[0] * shade, 0, 255)),
                    int(np.clip(base_col[1] * shade, 0, 255)),
                    int(np.clip(base_col[2] * shade, 0, 255))
                )

                if is_target:
                    cv2.line(canvas, p1, p2, (255, 230, 80), 5, cv2.LINE_AA)
                    cv2.line(canvas, p1, p2, bone_col, 3, cv2.LINE_AA)
                else:
                    thickness = 2 if target_f is not None else 3
                    cv2.line(canvas, p1, p2, bone_col, thickness, cv2.LINE_AA)

        # 6. Nós articulares
        for j in range(21):
            pt = screen_pts[j]
            is_target_joint = False
            if is_four_fingers:
                is_target_joint = (j in range(5, 21))
            elif target_f and target_f in FINGER_JOINTS:
                is_target_joint = (j in FINGER_JOINTS[target_f])

            if is_target_joint:
                cv2.circle(canvas, pt, 7, (50, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, 4, (230, 230, 240), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 5, (20, 20, 30), 1, cv2.LINE_AA)

        # 7. Anotar ângulos medidos nas articulações do dedo em calibração
        if is_four_fingers:
            idxs = FINGER_JOINTS['Index']
            j2 = joint_flexion(pts_clean[idxs[0]], pts_clean[idxs[1]], pts_clean[idxs[2]])
            j3 = joint_flexion(pts_clean[idxs[1]], pts_clean[idxs[2]], pts_clean[idxs[3]])
            j4 = joint_flexion(pts_clean[idxs[2]], pts_clean[idxs[3]], pts_clean[idxs[4]])
            cv2.putText(canvas, f"MCP {j2:4.1f} deg", (screen_pts[idxs[1]][0] - 88, screen_pts[idxs[1]][1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"PIP {j3:4.1f} deg", (screen_pts[idxs[2]][0] - 88, screen_pts[idxs[2]][1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"DIP {j4:4.1f} deg", (screen_pts[idxs[3]][0] - 88, screen_pts[idxs[3]][1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
            for fname, tip_idx in [('IND', 8), ('MED', 12), ('ANE', 16), ('MIN', 20)]:
                cv2.putText(canvas, fname, (screen_pts[tip_idx][0] - 10, screen_pts[tip_idx][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (166, 227, 161), 1, cv2.LINE_AA)
        elif target_f and target_f in FINGER_JOINTS:
            idxs = FINGER_JOINTS[target_f]
            if target_f != 'Thumb':
                j2 = joint_flexion(pts_clean[idxs[0]], pts_clean[idxs[1]], pts_clean[idxs[2]])
                j3 = joint_flexion(pts_clean[idxs[1]], pts_clean[idxs[2]], pts_clean[idxs[3]])
                j4 = joint_flexion(pts_clean[idxs[2]], pts_clean[idxs[3]], pts_clean[idxs[4]])

                mcp_pt = screen_pts[idxs[1]]
                pip_pt = screen_pts[idxs[2]]
                dip_pt = screen_pts[idxs[3]]
                tip_pt = screen_pts[idxs[4]]

                cv2.putText(canvas, f"MCP {j2:4.1f} deg", (mcp_pt[0] + 11, mcp_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"PIP {j3:4.1f} deg", (pip_pt[0] + 11, pip_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"DIP {j4:4.1f} deg", (dip_pt[0] + 11, dip_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "TIP", (tip_pt[0] + 11, tip_pt[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (166, 227, 161), 1, cv2.LINE_AA)
            else:
                ip_flex = joint_flexion(pts_clean[2], pts_clean[3], pts_clean[4])
                mcp_flex = joint_flexion(pts_clean[1], pts_clean[2], pts_clean[3])
                cv2.putText(canvas, f"MCP {mcp_flex:4.1f} deg", (screen_pts[2][0] + 11, screen_pts[2][1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"IP {ip_flex:4.1f} deg", (screen_pts[3][0] + 11, screen_pts[3][1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "TIP", (screen_pts[4][0] + 11, screen_pts[4][1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (166, 227, 161), 1, cv2.LINE_AA)

        # 8. Rótulos das pontas dos dedos se for baseline ou abertura
        if target_f is None:
            tip_names = [("POLEGAR", 4), ("INDICADOR", 8), ("MEDIO", 12), ("ANELAR", 16), ("MINIMO", 20)]
            for t_name, tip_idx in tip_names:
                t_pt = screen_pts[tip_idx]
                cv2.putText(canvas, t_name, (t_pt[0] - 22, t_pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (205, 214, 244), 1, cv2.LINE_AA)

        # 9. Barra de título superior
        cv2.rectangle(canvas, (0, 0), (vp_w, 24), (28, 26, 38), -1)
        cv2.line(canvas, (0, 24), (vp_w, 24), (137, 180, 250), 1)
        cv2.putText(canvas, title, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (137, 180, 250), 1, cv2.LINE_AA)

        # 10. Indicadores e dicas de navegação no rodapé
        ang_str = f"Y={yaw:+.0f} P={pitch:+.0f}"
        cv2.putText(canvas, ang_str, (8, vp_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 145, 165), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Girar:[A][D]", (max(10, vp_w - 85), vp_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (249, 226, 175), 1, cv2.LINE_AA)

        # Borda externa do viewport
        cv2.rectangle(canvas, (0, 0), (vp_w - 1, vp_h - 1), (137, 180, 250), 1)
        return canvas

    def _render_review_hud(self, frame: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        """Renderiza a tela de revisão e confirmação com representação visual completa da mão."""
        h, w, _ = frame.shape

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Escurecer fundo
        draw.rectangle([(0, 0), (w, h)], fill=(10, 10, 15, 150))

        # Cartão central responsivo e amplo
        card_w = min(int(w * 0.96), 1220)
        card_h = min(int(h * 0.92), 650)
        cx = (w - card_w) // 2
        cy = (h - card_h) // 2

        # Sombra e fundo do cartão
        draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], fill=(24, 24, 37, 245))
        draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], outline=(166, 227, 161, 255), width=2)

        # Cabeçalho do Cartão
        hdr_h = 64
        draw.rectangle([(cx, cy), (cx + card_w, cy + hdr_h)], fill=(30, 30, 46, 255))
        draw.line([(cx, cy + hdr_h), (cx + card_w, cy + hdr_h)], fill=(166, 227, 161, 255), width=2)

        f_hdr = self.hud.get_font(18, bold=True)
        f_sub = self.hud.get_font(13, bold=False)
        draw.text((cx + 22, cy + 10), "[CAPTURA REALIZADA] — REVISÃO E CONFIRMAÇÃO DOS DADOS", font=f_hdr, fill=(166, 227, 161, 255))
        draw.text((cx + 22, cy + 36), f"Passo {self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}: {step['title']}", font=f_sub, fill=(205, 214, 244, 255))

        # Rodapé de Decisão no Cartão
        btn_h = 66
        btn_y = cy + card_h - btn_h
        draw.rectangle([(cx, btn_y), (cx + card_w, cy + card_h)], fill=(17, 17, 27, 255))
        draw.line([(cx, btn_y), (cx + card_w, btn_y)], fill=(69, 71, 90, 255), width=1)

        f_btn_bold = self.hud.get_font(15, bold=True)
        f_btn_sub = self.hud.get_font(12, bold=False)
        draw.text((cx + 25, btn_y + 12), "[ESPAÇO] ou [ENTER] : CONFIRMAR E AVANÇAR PARA O PRÓXIMO PASSO", font=f_btn_bold, fill=(166, 227, 161, 255))
        draw.text((cx + 25, btn_y + 38), "[R] : REFAZER CAPTURA (ajustar pose)    |    [B] : VOLTAR AO ANTERIOR    |    [A] / [D] : GIRAR MODELO 3D", font=f_btn_sub, fill=(249, 226, 175, 255))

        # ÁREA DE CONTEÚDO
        content_y = cy + hdr_h + 12
        body_h = btn_y - content_y - 12

        col1_w = 340 if w >= 1100 else 280
        col1_x = cx + 20
        col2_x = cx + col1_w + 30
        col2_w = card_w - col1_w - 45

        f_sec = self.hud.get_font(15, bold=True)
        f_data = self.hud.get_font(12, bold=False)
        f_small = self.hud.get_font(10, bold=False)

        # Coluna 1: Métricas Biomecânicas
        draw.text((col1_x, content_y), "Métricas Biomecânicas (Ângulos e Flexões):", font=f_sec, fill=(249, 226, 175, 255))

        for i, line in enumerate(self.current_review_metrics):
            col_color = (166, 227, 161, 255) if "Excelente" in line else (205, 214, 244, 255)
            draw.text((col1_x + 6, content_y + 24 + i * 22), line, font=f_data, fill=col_color)

        # Legenda das Cores Anatômicas no final da Coluna 1
        leg_y = content_y + 24 + len(self.current_review_metrics) * 22 + 10
        draw.line([(col1_x, leg_y - 6), (col1_x + col1_w - 10, leg_y - 6)], fill=(69, 71, 90, 180), width=1)
        draw.text((col1_x, leg_y), "Cores Anatômicas no Modelo 3D:", font=self.hud.get_font(11, bold=True), fill=(186, 194, 222, 255))
        draw.text((col1_x + 4, leg_y + 16), "• Polegar: Âmbar   • Indicador: Amarelo", font=f_small, fill=(249, 226, 175, 255))
        draw.text((col1_x + 4, leg_y + 30), "• Médio: Verde     • Anelar: Ciano", font=f_small, fill=(166, 227, 161, 255))
        draw.text((col1_x + 4, leg_y + 44), "• Mínimo: Magenta  • Alvo: Realce Neon", font=f_small, fill=(245, 194, 231, 255))

        # Coluna 2: Título da Representação Visual da Mão
        draw.text((col2_x, content_y), "Representação Visual: Frontal, Lateral (Perfil) e 3D Digital:", font=f_sec, fill=(137, 180, 250, 255))

        # Linha divisória vertical suave entre colunas
        div_x = col2_x - 15
        draw.line([(div_x, content_y + 4), (div_x, btn_y - 12)], fill=(69, 71, 90, 180), width=1)

        # Converter de PIL para OpenCV BGR para blitar os viewports gráficos
        out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Dimensões dos 3 sub-viewports na Coluna 2
        vp_spacing = 10
        vp_w = (col2_w - 2 * vp_spacing) // 3
        vp_h = body_h - 26
        vp_y = content_y + 24
        vp1_x = col2_x
        vp2_x = col2_x + vp_w + vp_spacing
        vp3_x = col2_x + 2 * (vp_w + vp_spacing)

        # Recuperar dados do passo capturado
        step_id = step['id']
        step_data = self.captured_data.get(step_id, {})
        frontal_entry = step_data.get('frontal', {})
        lateral_entry = step_data.get('lateral', {})

        frontal_snapshot = frontal_entry.get('snapshot')
        if frontal_snapshot is None:
            frontal_snapshot = self.current_review_snapshot if self.current_review_snapshot is not None else frame

        lateral_snapshot = lateral_entry.get('snapshot')
        if lateral_snapshot is None:
            lateral_snapshot = frontal_snapshot

        frontal_pts_raw = frontal_entry.get('pts_raw')
        lateral_pts_raw = lateral_entry.get('pts_raw')
        pts_norm = step_data.get('pts_norm')

        # 1. Viewport 1: Câmera Real - Frontal
        vp1_crop = self._get_hand_crop_roi(frontal_snapshot, frontal_pts_raw, vp_w, vp_h, title="1. REAL (FRONTAL)")
        out_frame[vp_y:vp_y + vp_h, vp1_x:vp1_x + vp_w] = vp1_crop

        # 2. Viewport 2: Câmera Real - Lateral / Perfil
        vp2_crop = self._get_hand_crop_roi(lateral_snapshot, lateral_pts_raw, vp_w, vp_h, title="2. REAL (PERFIL)")
        out_frame[vp_y:vp_y + vp_h, vp2_x:vp2_x + vp_w] = vp2_crop

        # 3. Viewport 3: Modelo 3D Biomecânico Reconstruído
        hand_3d = self._render_hand_model_3d(pts_norm, step, vp_w, vp_h, yaw=self.review_yaw, pitch=self.review_pitch, title="3. 3D DIGITAL (CODIGO)")
        out_frame[vp_y:vp_y + vp_h, vp3_x:vp3_x + vp_w] = hand_3d

        return out_frame

    # -----------------------------------------------------------------------
    # COMPILAÇÃO E EXPORTAÇÃO DE CALIBRAÇÃO E SEEDS
    # -----------------------------------------------------------------------

    def compile_and_save_settings(self, output_path: str = CALIBRATION_FILE) -> Dict[str, Any]:
        """Compila todas as medições capturadas e gera o calibration_settings.json."""
        print("\n[COMPILAÇÃO] Processando limites anatômicos capturados...")

        # Baseline: comprimentos ósseos (prioriza visão frontal da mão espalmada)
        base_data = self.captured_data.get('baseline_open', None)
        if base_data is not None:
            if 'frontal' in base_data and base_data['frontal'].get('pts_norm') is not None:
                ref_pts = base_data['frontal']['pts_norm']
            else:
                ref_pts = base_data.get('pts_norm', np.zeros((21, 3)))
        else:
            ref_pts = np.zeros((21, 3))
            ref_pts[9] = np.array([0.0, 1.0, 0.0])

        phalanx_lengths = {
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

        # Estágios dos dedos longos (0 a 4) - Prioriza visão de perfil lateral (sem distorção Z)
        stages: Dict[str, Dict[str, Dict[str, float]]] = {}
        for f_name in ['Index', 'Middle', 'Ring', 'Pinky']:
            stages[f_name] = {}
            idxs = FINGER_JOINTS[f_name]

            # Fallbacks canônicos por estágio caso o passo não tenha sido capturado
            canon_stages = {
                0: {'J2_Pitch': 0.0,  'J3_Pitch': 0.0,   'J4_Pitch': 0.0},
                1: {'J2_Pitch': 25.0, 'J3_Pitch': 40.0,  'J4_Pitch': 35.0},
                2: {'J2_Pitch': 0.0,  'J3_Pitch': 90.0,  'J4_Pitch': 75.0},
                3: {'J2_Pitch': 85.0, 'J3_Pitch': 0.0,   'J4_Pitch': 0.0},
                4: {'J2_Pitch': 85.0, 'J3_Pitch': 105.0, 'J4_Pitch': 80.0}
            }

            for st in [0, 1, 2, 3, 4]:
                step_key = None
                for cand in [f"four_fingers_s{st}", f"fingers_s{st}", f"{f_name.lower()}_s{st}"]:
                    if cand in self.captured_data:
                        step_key = cand
                        break

                if step_key is not None:
                    step_entry = self.captured_data[step_key]
                    if 'lateral' in step_entry and step_entry['lateral'].get('pts_norm') is not None:
                        p = step_entry['lateral']['pts_norm']
                    else:
                        p = step_entry.get('pts_norm', np.zeros((21, 3)))

                    if np.linalg.norm(p[idxs[4]] - p[idxs[1]]) < 1e-3:
                        stages[f_name][str(st)] = canon_stages[st]
                    elif st == 0:
                        # Estágio 0 é por definição estendido reto: MCP=0, PIP=0, DIP=0 (elimina ruído de curvatura)
                        stages[f_name]['0'] = {'J2_Pitch': 0.0, 'J3_Pitch': 0.0, 'J4_Pitch': 0.0}
                    else:
                        j2 = joint_flexion(p[idxs[0]], p[idxs[1]], p[idxs[2]])
                        j3 = joint_flexion(p[idxs[1]], p[idxs[2]], p[idxs[3]])
                        j4 = joint_flexion(p[idxs[2]], p[idxs[3]], p[idxs[4]])
                        if st == 4:
                            # Estágio 4 é punho cerrado total: garante fechamento completo contra a palma
                            j2 = max(float(j2), 85.0)
                            j3 = max(float(j3), 105.0)
                            j4 = max(float(j4), 78.0)
                        stages[f_name][str(st)] = {
                            'J2_Pitch': float(j2),
                            'J3_Pitch': float(j3),
                            'J4_Pitch': float(j4)
                        }
                else:
                    stages[f_name][str(st)] = canon_stages[st]

        # Spreads (Aberturas laterais - avaliadas na visão frontal)
        spread_angles = {
            'Pinky_Ring':   {'0': +10.0, '1': -15.0},
            'Ring_Middle':  {'0': +8.0,  '1': -10.0},
            'Middle_Index': {'0': -8.0,  '1': +10.0},
            'Index_Thumb':  {'0': -48.0, '1': 0.0}
        }
        if 'spread_open' in self.captured_data:
            step_open = self.captured_data['spread_open']
            p_open = step_open['frontal']['pts_norm'] if ('frontal' in step_open and step_open['frontal'].get('pts_norm') is not None) else step_open['pts_norm']
            spread_angles['Pinky_Ring']['0']   = float(vec_angle(p_open[17] - p_open[0], p_open[13] - p_open[0]))
            spread_angles['Ring_Middle']['0']  = float(vec_angle(p_open[13] - p_open[0], p_open[9] - p_open[0]))
            spread_angles['Middle_Index']['0'] = float(-vec_angle(p_open[9] - p_open[0], p_open[5] - p_open[0]))
            spread_angles['Index_Thumb']['0']  = -48.0

        if 'spread_closed' in self.captured_data:
            step_cls = self.captured_data['spread_closed']
            p_cls = step_cls['frontal']['pts_norm'] if ('frontal' in step_cls and step_cls['frontal'].get('pts_norm') is not None) else step_cls['pts_norm']
            spread_angles['Pinky_Ring']['1']   = float(-vec_angle(p_cls[17] - p_cls[0], p_cls[13] - p_cls[0]))
            spread_angles['Ring_Middle']['1']  = float(-vec_angle(p_cls[13] - p_cls[0], p_cls[9] - p_cls[0]))
            spread_angles['Middle_Index']['1'] = float(+vec_angle(p_cls[9] - p_cls[0], p_cls[5] - p_cls[0]))
            spread_angles['Index_Thumb']['1']  = 0.0

        # Polegar (Thumb limits baseados nas capturas reais em 2 ângulos)
        thumb_limits = {
            "f0_p0": {
                "cmc": [-0.353, -0.152, 0.0],
                "u1": [-0.7203, -0.6395, -0.2688],
                "u2": [-0.7765, -0.5263, -0.3465],
                "u3": [-0.8190, -0.2349, -0.5235]
            },
            "f0_p1": {
                "cmc": [-0.367, -0.111, 0.0],
                "u1": [-0.7471, -0.5264, -0.4058],
                "u2": [-0.5991, -0.6919, -0.4029],
                "u3": [ 0.2671, -0.8361, -0.4791]
            },
            "f0_closed": {
                "cmc": [-0.272, -0.290, 0.0],
                "u1": [-0.276, -0.954, -0.123],
                "u2": [ 0.006, -0.998, -0.064],
                "u3": [ 0.177, -0.982, -0.038]
            },
            "f0_closed_p1": {
                "cmc": [-0.272, -0.290, 0.0],
                "u1": [-0.276, -0.954, -0.123],
                "u2": [ 0.006, -0.998, -0.064],
                "u3": [ 0.520, -0.750, -0.400]
            },
            "f1_p0": {
                "cmc": [-0.292, -0.154, 0.0],
                "u1": [-0.4456, -0.8647, -0.2318],
                "u2": [ 0.2685, -0.9615,  0.0585],
                "u3": [ 0.6728, -0.6437,  0.3647]
            },
            "f1_p1": {
                "cmc": [-0.359, -0.178, 0.0],
                "u1": [-0.5247, -0.7790, -0.3433],
                "u2": [ 0.2517, -0.9602, -0.1210],
                "u3": [ 0.9066, -0.2657,  0.3279]
            }
        }

        # Extrai dinamicamente de captured_data caso disponível
        for limit_key, step_candidates in [('f0_p0', ['thumb_f0_p0']), ('f0_p1', ['thumb_f0_p1']),
                                           ('f1_p0', ['thumb_f1', 'thumb_f1_p0']), ('f1_p1', ['thumb_f1', 'thumb_f1_p1'])]:
            step_id = next((s for s in step_candidates if s in self.captured_data), None)
            if step_id is not None:
                entry = self.captured_data[step_id]
                p_f = entry.get('frontal', {}).get('pts_norm')
                p_l = entry.get('lateral', {}).get('pts_norm')
                if p_f is not None and len(p_f) == 21 and p_l is not None and len(p_l) == 21:
                    # Direção frontal (X, Y) e lateral (Z)
                    cmc_pos = [float(p_f[1][0]), float(p_f[1][1]), 0.0]
                    v1 = np.array([p_f[2][0] - p_f[1][0], p_f[2][1] - p_f[1][1], p_l[2][0] - p_l[1][0]])
                    v2 = np.array([p_f[3][0] - p_f[2][0], p_f[3][1] - p_f[2][1], p_l[3][0] - p_l[2][0]])
                    v3 = np.array([p_f[4][0] - p_f[3][0], p_f[4][1] - p_f[3][1], p_l[4][0] - p_l[3][0]])
                    n1, n2, n3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
                    if n1 > 1e-4 and n2 > 1e-4 and n3 > 1e-4:
                        thumb_limits[limit_key] = {
                            "cmc": cmc_pos,
                            "u1": (v1 / n1).round(4).tolist(),
                            "u2": (v2 / n2).round(4).tolist(),
                            "u3": (v3 / n3).round(4).tolist()
                        }

        thumb_config = {
            'f0_pitch': 5.0,
            'f0_mcp_pitch': 5.0,
            'f0_ip_flex': 65.0,
            'f1_opp_yaw': 45.0,
            'f1_opp_roll': -40.0,
            'f1_opp_pitch': 40.0,
            'f1_mcp_pitch': 45.0,
            'f1_ip_flex': 65.0
        }
        if 'thumb_f0_p1' in self.captured_data:
            step_t0 = self.captured_data['thumb_f0_p1']
            p_t0 = step_t0['lateral']['pts_norm'] if ('lateral' in step_t0 and step_t0['lateral'].get('pts_norm') is not None) else step_t0['pts_norm']
            thumb_config['f0_ip_flex'] = float(joint_flexion(p_t0[2], p_t0[3], p_t0[4]))

        f1_cand = next((s for s in ['thumb_f1', 'thumb_f1_p1', 'thumb_f1_p0'] if s in self.captured_data), None)
        if f1_cand is not None:
            step_t1 = self.captured_data[f1_cand]
            p_t1 = step_t1['lateral']['pts_norm'] if ('lateral' in step_t1 and step_t1['lateral'].get('pts_norm') is not None) else step_t1['pts_norm']
            thumb_config['f1_ip_flex'] = float(joint_flexion(p_t1[2], p_t1[3], p_t1[4]))

        # Salvar marcações completas dos 21 landmarks de cada passo para montagem modular direta
        captured_landmarks = {}
        for step_id, step_entry in self.captured_data.items():
            entry_dict = {}
            if 'frontal' in step_entry and step_entry['frontal'].get('pts_norm') is not None:
                entry_dict['frontal'] = np.array(step_entry['frontal']['pts_norm']).round(5).tolist()
            if 'lateral' in step_entry and step_entry['lateral'].get('pts_norm') is not None:
                entry_dict['lateral'] = np.array(step_entry['lateral']['pts_norm']).round(5).tolist()
            if 'pts_norm' in step_entry and step_entry['pts_norm'] is not None:
                entry_dict['pts_norm'] = np.array(step_entry['pts_norm']).round(5).tolist()
            captured_landmarks[step_id] = entry_dict

        calib_dict = {
            'metadata': {
                'generated_by': 'GuidedHandCalibrator_v3_UTF8',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'steps_captured': len(self.captured_data),
                'total_steps': len(CALIBRATION_STEPS),
                'flexion_stages_count': 5
            },
            'stages': stages,
            'spread_angles': spread_angles,
            'thumb_config': thumb_config,
            'thumb_limits': thumb_limits,
            'phalanx_lengths': phalanx_lengths,
            'avg_palm': avg_palm,
            'captured_landmarks': captured_landmarks
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calib_dict, f, indent=2, ensure_ascii=False)

        print(f"[SUCESSO] Calibração salva com sucesso em: {output_path}")

        # Gerar Seeds atualizadas automaticamente
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from kinematic_seed_generator import HandKinematicsDirect
            kinematics = HandKinematicsDirect.from_calibration_file(output_path)
            kinematics.export_seeds_json(SEEDS_FILE)
            print(f"[SEEDS] Catálogo seeds.json gerado a partir da calibração!")
        except Exception as e:
            print(f"[AVISO] Não foi possível auto-gerar seeds.json: {e}")

        return calib_dict

    def run_mock_calibration(self, output_path: str = CALIBRATION_FILE) -> Dict[str, Any]:
        """Gera calibração simulada completa sem necessidade de câmera (para testes automatizados)."""
        print("[MOCK] Gerando calibração anatômica 5-estágios simulada...")
        self.captured_data = {}  # Vazio para acionar todos os padrões canônicos perfeitamente
        return self.compile_and_save_settings(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrador Biomecânico Guiado da Mão - LIBRAS TCC")
    parser.add_argument('--camera', type=int, default=0, help="Índice da câmera OpenCV (padrão 0)")
    parser.add_argument('--mock', action='store_true', help="Executa calibração simulada sem câmera (testes)")
    parser.add_argument('--output', type=str, default=CALIBRATION_FILE, help="Caminho do calibration_settings.json")
    args = parser.parse_args()

    calibrator = GuidedHandCalibrator(camera_idx=args.camera)
    if args.mock:
        calibrator.run_mock_calibration(output_path=args.output)
    else:
        calibrator.run_interactive()

if __name__ == '__main__':
    main()
