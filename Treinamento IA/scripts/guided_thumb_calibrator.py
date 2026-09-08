#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided Thumb Calibrator (guided_thumb_calibrator.py)
===================================================
Assistente interativo em tempo real (OpenCV + MediaPipe Hands + Pillow UnicodeHUD)
dedicado à calibração biomecânica especializada do polegar em 3 estados anatômicos:
  1. Polegar Aberto Esticado (Palma Aberta - Abdução Radial Máxima)
  2. Polegar Junto aos Dedos (Dedos Fechados / Aduzido ao Lado da Palma)
  3. Polegar na Transversal (Oposição Cruzando a Frente da Palma)

Recursos:
- Captura em DOIS ÂNGULOS SEPARADOS (Frontal + Perfil Lateral 90°) por passo.
- Pausa explícita de transição entre Frontal e Lateral para o usuário virar a mão de lado com calma.
- Fusão biomecânica:
    * Eixos X e Y extraídos da vista Frontal (abdução/adução no plano coronal da palma).
    * Eixo Z de profundidade extraído da vista Lateral (projeção no plano sagital da mão).
- Preservação rígida e absoluta dos comprimentos ósseos das falanges (L1, L2, L3) medidos de baseline_open.
- Estabilização temporal (filtro de jitter de 1.2s com barra de progresso visual).
- Tela de Revisão 3D interativa com ambos os viewports (Frontal e Perfil 90°) reconstruídos a partir dos dados combinados.
- Gravação direta em calibration_settings.json e regeneração automática do seeds.json.
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
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s],
        [0.0, s,    c]
    ], dtype=np.float64)

def rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c]
    ], dtype=np.float64)

def to_canonical_palm_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma landmarks 3D para o referencial ortonormal canônico da palma:
    - Origem (0, 0, 0) no pulso (Landmark 0).
    - Eixo Y longitudinal apontando para cima (-Y na tela, alinhado ao metacarpo médio Landmark 9).
    - Eixo X transversal apontando para o mindinho (+X na tela, Landmark 17).
    - Eixo Z normal à palma apontando para a frente (+Z para a câmera / observador).
    Garante que os nós MCP 5 e 17 tenham rigorosamente a mesma profundidade Z (zero yaw tilt).
    """
    p0 = pts[0].copy()
    v_y = pts[9] - p0
    y_norm = np.linalg.norm(v_y)
    y_unit = v_y / y_norm if y_norm > 1e-6 else np.array([0.0, -1.0, 0.0])

    v_x_raw = pts[17] - pts[5]
    v_x = v_x_raw - np.dot(v_x_raw, y_unit) * y_unit
    x_norm = np.linalg.norm(v_x)
    x_unit = v_x / x_norm if x_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    e_x = x_unit
    e_y = -y_unit  # dedos apontam para cima (-Y na tela)
    e_z = np.cross(e_x, e_y)
    z_norm = np.linalg.norm(e_z)
    e_z = e_z / z_norm if z_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    R_canon = np.stack([e_x, e_y, e_z], axis=0)
    pts_canon = (pts - p0) @ R_canon.T
    return pts_canon, R_canon

def clean_json_value(val: Any) -> Any:
    """Converte recursivamente ndarrays e escalares numpy para tipos nativos serializáveis em JSON."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.round(6).tolist()
    if isinstance(val, (np.floating, float)):
        return float(round(float(val), 6))
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (list, tuple)):
        return [clean_json_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): clean_json_value(v) for k, v in val.items()}
    return val

FINGER_JOINTS = {
    'Thumb':  [0, 1, 2, 3, 4],     # Wrist, CMC, MCP, IP, TIP
    'Index':  [0, 5, 6, 7, 8],     # Wrist, MCP, PIP, DIP, TIP
    'Middle': [0, 9, 10, 11, 12],
    'Ring':   [0, 13, 14, 15, 16],
    'Pinky':  [0, 17, 18, 19, 20]
}

PALM_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]

# ---------------------------------------------------------------------------
# DEFINIÇÃO DOS 3 PASSOS DE CALIBRAÇÃO DO POLEGAR
# ---------------------------------------------------------------------------

THUMB_CALIBRATION_STEPS: List[Dict[str, Any]] = [
    {
        'id': 'thumb_open',
        'step_num': 1,
        'title': 'POLEGAR 1/3: ABERTO ESTICADO (PALMA ABERTA)',
        'posture_front': 'Palma aberta voltada de frente para a câmera na vertical.',
        'posture_lat': 'Gire a mão 90° de perfil para a câmera, mantendo o polegar no mesmo formato.',
        'target_action': 'Abra o polegar para a lateral radial o máximo que puder no MESMO PLANO da palma (sem cruzar a frente da mão), totalmente reto (como no sinal "L").',
        'other_fingers': 'Mantenha os outros 4 dedos (indicador ao mínimo) estendidos para cima.',
        'expected_summary': 'Abdução radial no plano da palma | Ponta IP reta (0°) | Perfil alinhado',
        'thumb_state': 0
    },
    {
        'id': 'thumb_closed',
        'step_num': 2,
        'title': 'POLEGAR 2/3: JUNTO AOS DEDOS (FECHADO / ADUZIDO)',
        'posture_front': 'Palma voltada de frente para a câmera na vertical.',
        'posture_lat': 'Gire a mão 90° de perfil para a câmera, mantendo o polegar colado ao indicador.',
        'target_action': 'Feche os dedos (ou mantenha-os juntos em bloco) e cole o polegar aduzido firmemente ao lado do dedo indicador / palma (como nos sinais "A" ou "B").',
        'other_fingers': 'Dedos fechados ou juntos sem afastar o polegar.',
        'expected_summary': 'Polegar aduzido colado ao indicador | Adução máxima ~0°',
        'thumb_state': 1
    },
    {
        'id': 'thumb_transversal',
        'step_num': 3,
        'title': 'POLEGAR 3/3: NA TRANSVERSAL (OPOSIÇÃO CRUZANDO A PALMA)',
        'posture_front': 'Palma voltada de frente para a câmera na vertical.',
        'posture_lat': 'Gire a mão 90° de perfil para a câmera, mantendo o polegar cruzado na frente.',
        'target_action': 'Traga o polegar cruzando transversalmente a FRENTE da palma da mão, apontando em direção à base dos dedos anelar e mínimo (como nos sinais "M", "N" ou "E").',
        'other_fingers': 'Dedos erguidos/estendidos para dar visibilidade clara ao polegar cruzando a palma.',
        'expected_summary': 'Oposição transversal máxima cruzando a frente da palma da mão',
        'thumb_state': 2
    }
]

# ---------------------------------------------------------------------------
# CLASSE PRINCIPAL DO CALIBRADOR GUIADO DO POLEGAR
# ---------------------------------------------------------------------------

class GuidedThumbCalibrator:
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
        # Estados: "CAPTURING" (com sub_angle "FRONTAL" ou "LATERAL"), "WAITING_LATERAL", "REVIEW"
        self.state = "CAPTURING"
        self.current_sub_angle: str = "FRONTAL"

        self.step_subcaptures: Dict[str, Any] = {}
        self.captured_data: Dict[str, Any] = {}
        self.current_review_metrics: List[str] = []
        self.current_review_status: str = "Adequado"
        self.current_review_snapshot: Optional[np.ndarray] = None

        self.stable_frame_buffer: List[np.ndarray] = []
        self.stability_start_time: Optional[float] = None
        self.REQUIRED_STABLE_TIME = 1.2  # 1.2 segundos para disparo automático

        # Controles de rotação 3D da tela de revisão
        self.review_yaw: float = 18.0
        self.review_pitch: float = -12.0

        # Debounce e proteção temporal contra disparos duplos de transição de estado
        self.last_state_change_time: float = 0.0
        self.COOLDOWN_TRANSITION: float = 0.7  # 700ms de intervalo mínimo entre ações

        # Carregar comprimentos ósseos rígidos e base da palma
        self.rigid_thumb_lengths = self._load_baseline_thumb_lengths()
        self.palm_base = self._load_baseline_palm_base()

    def _load_baseline_thumb_lengths(self) -> Tuple[float, float, float]:
        default_lengths = (0.415, 0.320, 0.249)
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'phalanx_lengths' in data and 'Thumb' in data['phalanx_lengths']:
                    tl = data['phalanx_lengths']['Thumb']
                    if len(tl) == 3:
                        l1, l2, l3 = float(tl[0]), float(tl[1]), float(tl[2])
                        print(f"[ThumbCalibrator] Comprimentos ósseos rígidos (phalanx_lengths): L1={l1:.3f}, L2={l2:.3f}, L3={l3:.3f}")
                        return (l1, l2, l3)

                caps = data.get('captured_landmarks', {})
                ref = caps.get('baseline_open') or caps.get('spread_open')
                if ref is not None:
                    pts = np.array(ref.get('lateral') if 'lateral' in ref and ref['lateral'] is not None else (ref.get('pts_norm') or ref.get('frontal')), dtype=np.float64)
                    if len(pts) == 21:
                        l1 = float(np.linalg.norm(pts[2] - pts[1]))
                        l2 = float(np.linalg.norm(pts[3] - pts[2]))
                        l3 = float(np.linalg.norm(pts[4] - pts[3]))
                        print(f"[ThumbCalibrator] Comprimentos ósseos rígidos (baseline_open): L1={l1:.3f}, L2={l2:.3f}, L3={l3:.3f}")
                        return (l1, l2, l3)
            except Exception as e:
                print(f"[ThumbCalibrator] Aviso: erro ao carregar baseline ({e}). Usando proporções padrão.")
        return default_lengths

    def _load_baseline_palm_base(self) -> np.ndarray:
        fallback = np.array([
            [ 0.000,  0.000,  0.000],  # Wrist (0)
            [-0.164, -0.295,  0.000],  # Thumb CMC (1) - ancorado rigorosamente em Z=0
            [-0.138, -0.980,  0.000],  # Index MCP (5)
            [ 0.000, -0.997,  0.000],  # Middle MCP (9)
            [ 0.100, -0.950,  0.000],  # Ring MCP (13)
            [ 0.213, -0.887,  0.000]   # Pinky MCP (17)
        ], dtype=np.float64)

        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                caps = data.get('captured_landmarks', {})
                ref = caps.get('baseline_open') or caps.get('spread_open')
                if ref is not None:
                    pts = np.array(ref.get('lateral') if 'lateral' in ref and ref['lateral'] is not None else (ref.get('pts_norm') or ref.get('frontal')), dtype=np.float64)
                    if len(pts) == 21:
                        pts_can, _ = to_canonical_palm_frame(pts)
                        base = pts_can[[0, 1, 5, 9, 13, 17]].copy()
                        base[:, 2] = 0.0  # Ancoragem planar no referencial coronal Z=0
                        return base
            except Exception:
                pass
        return fallback

    def run_interactive(self) -> bool:
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir a câmera índice {self.camera_idx}.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "Calibrador Biomecanico Guiado do Polegar - LIBRAS TCC"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*70)
        print("  CALIBRADOR BIOMECÂNICO GUIADO DO POLEGAR INICIADO COM SUCESSO")
        print("="*70)
        print("Fluxo de Operação por Passo:")
        print("  1. Etapa 1/2 (FRONTAL): Posicione a mão de frente para a câmera.")
        print("  2. Segure estável 1.2s (ou aperte [ESPAÇO]) para capturar a visão frontal.")
        print("  3. Transição: Gire a mão 90° de perfil e aperte [ESPAÇO] quando pronto.")
        print("  4. Etapa 2/2 (LATERAL): Segure estável 1.2s para capturar o perfil Z.")
        print("  5. Revisão 3D: Confira ambas as vistas 3D geradas a partir das capturas.")
        print("  6. Pressione [ESPAÇO] para confirmar ou [R] para refazer o passo.")
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

                # Desenhar skeleton com destaque no polegar no modo captura
                if self.state in ["CAPTURING", "WAITING_LATERAL"]:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[0],
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    for j_idx in [1, 2, 3, 4]:
                        pt = (int(pts_pixels[j_idx][0]), int(pts_pixels[j_idx][1]))
                        cv2.circle(frame, pt, 8, (40, 140, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, pt, 11, (255, 255, 255), 2, cv2.LINE_AA)

            step = THUMB_CALIBRATION_STEPS[self.current_step_idx]

            # ---------------------------------------------------------------
            # MODO 1: CAPTURA EM TEMPO REAL (FRONTAL ou LATERAL)
            # ---------------------------------------------------------------
            if self.state == "CAPTURING":
                now = time.time()
                # Cooldown de transição para estabilizar o início da captura e evitar falsos disparos
                if now - self.last_state_change_time < self.COOLDOWN_TRANSITION:
                    self.stability_start_time = None
                    self.stable_frame_buffer = []
                elif has_hand and pts_norm is not None:
                    if self.stability_start_time is None:
                        self.stability_start_time = now
                        self.stable_frame_buffer = [pts_norm]
                    else:
                        self.stable_frame_buffer.append(pts_norm)
                        elapsed = now - self.stability_start_time
                        if elapsed >= self.REQUIRED_STABLE_TIME and len(self.stable_frame_buffer) >= 20:
                            if self.current_sub_angle == "FRONTAL":
                                self._record_sub_capture(step, frame, pts_pixels, pts_norm, "frontal")
                                self.state = "WAITING_LATERAL"
                                self.last_state_change_time = time.time()
                                self.stability_start_time = None
                                self.stable_frame_buffer = []
                                print(f"  ✓ Vista Frontal capturada com sucesso! Agora gire a mão 90° de perfil...")
                            else:
                                self._record_sub_capture(step, frame, pts_pixels, pts_norm, "lateral")
                                self.last_state_change_time = time.time()
                                self._finalize_step_and_review(step)
                else:
                    self.stability_start_time = None
                    self.stable_frame_buffer = []

                frame = self._render_capturing_hud(frame, step, has_hand, pts_norm)

            # ---------------------------------------------------------------
            # MODO 2: TRANSIÇÃO GUIADA ENTRE FRONTAL E LATERAL
            # ---------------------------------------------------------------
            elif self.state == "WAITING_LATERAL":
                frame = self._render_waiting_lateral_hud(frame, step)

            # ---------------------------------------------------------------
            # MODO 3: REVISÃO E CONFIRMAÇÃO (DUAS VISTAS 3D FUSIONADAS)
            # ---------------------------------------------------------------
            elif self.state == "REVIEW":
                bg_frame = self.current_review_snapshot.copy() if self.current_review_snapshot is not None else frame
                frame = self._render_review_hud(bg_frame, step)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), 27]:
                print("\n[LOG] Calibração do polegar interrompida pelo usuário.")
                break

            elif key in [32, 13]:  # ESPAÇO ou ENTER
                now = time.time()
                if now - self.last_state_change_time < self.COOLDOWN_TRANSITION:
                    # Ignora tecla se pressionada dentro do intervalo mínimo de debounce pós-transição
                    pass
                elif self.state == "CAPTURING":
                    if has_hand and pts_norm is not None:
                        self.stable_frame_buffer.append(pts_norm)
                        if self.current_sub_angle == "FRONTAL":
                            self._record_sub_capture(step, frame, pts_pixels, pts_norm, "frontal")
                            self.state = "WAITING_LATERAL"
                            self.last_state_change_time = time.time()
                            self.stability_start_time = None
                            self.stable_frame_buffer = []
                            print(f"  ✓ Vista Frontal capturada manualmente! Agora gire a mão 90° de perfil...")
                        else:
                            self._record_sub_capture(step, frame, pts_pixels, pts_norm, "lateral")
                            self.last_state_change_time = time.time()
                            self._finalize_step_and_review(step)

                elif self.state == "WAITING_LATERAL":
                    # Usuário posicionou a mão de perfil e apertou ESPAÇO -> inicia captura lateral
                    self.current_sub_angle = "LATERAL"
                    self.state = "CAPTURING"
                    self.last_state_change_time = time.time()
                    self.stability_start_time = None
                    self.stable_frame_buffer = []
                    print(f"  Iniciando captura da Vista Lateral (perfil 90°)... Segure firme.")

                elif self.state == "REVIEW":
                    print(f"  ✓ Passo [{self.current_step_idx + 1}/{len(THUMB_CALIBRATION_STEPS)}] confirmado pelo usuário.")
                    self.last_state_change_time = time.time()
                    self._advance_step()

            elif key == ord('r'):  # Refazer passo completo
                step_id = step['id']
                if step_id in self.captured_data:
                    del self.captured_data[step_id]
                self.step_subcaptures = {}
                self.current_sub_angle = "FRONTAL"
                self.state = "CAPTURING"
                self.stability_start_time = None
                self.stable_frame_buffer = []
                print(f"[REFAZER] Reiniciando captura completa de: {step['title']}")

            elif key == ord('b'):  # Voltar passo anterior
                if self.current_step_idx > 0:
                    self.current_step_idx -= 1
                    self.step_subcaptures = {}
                    self.current_sub_angle = "FRONTAL"
                    self.state = "CAPTURING"
                    self.stability_start_time = None
                    self.stable_frame_buffer = []
                    prev_step = THUMB_CALIBRATION_STEPS[self.current_step_idx]
                    print(f"[VOLTAR] Retornando ao passo: {prev_step['title']}")

            elif self.state == "REVIEW":
                if key in [ord('a'), ord('A'), 81]:
                    self.review_yaw -= 8.0
                elif key in [ord('d'), ord('D'), 83]:
                    self.review_yaw += 8.0
                elif key in [ord('w'), ord('W'), 82]:
                    self.review_pitch = min(85.0, self.review_pitch + 6.0)
                elif key in [ord('x'), ord('X'), 84]:
                    self.review_pitch = max(-85.0, self.review_pitch - 6.0)

            if self.current_step_idx >= len(THUMB_CALIBRATION_STEPS):
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.current_step_idx >= len(THUMB_CALIBRATION_STEPS):
            print("\n[SUCESSO] Todos os 3 passos de calibração do polegar foram concluídos!")
            self.compile_and_save_thumb()
            return True
        else:
            print(f"\n[INFO] Calibração interrompida no passo {self.current_step_idx + 1}/{len(THUMB_CALIBRATION_STEPS)}.")
            return False

    def _record_sub_capture(
        self,
        step: Dict[str, Any],
        frame: np.ndarray,
        pts_raw: Optional[np.ndarray],
        pts_norm: Optional[np.ndarray],
        angle: str
    ):
        step_id = step['id']
        img_filename = os.path.join(CAPTURES_DIR, f"thumb_{step_id}_{angle}.png")

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

        # Snapshot anotado
        annotated_snapshot = frame.copy()
        if pts_raw is not None and len(pts_raw) == 21:
            idxs = FINGER_JOINTS['Thumb']
            for i in range(1, len(idxs)):
                p1 = (int(pts_raw[idxs[i-1]][0]), int(pts_raw[idxs[i-1]][1]))
                p2 = (int(pts_raw[idxs[i]][0]), int(pts_raw[idxs[i]][1]))
                cv2.line(annotated_snapshot, p1, p2, (40, 140, 255), 4, cv2.LINE_AA)
            for j_idx in idxs:
                pt = (int(pts_raw[j_idx][0]), int(pts_raw[j_idx][1]))
                cv2.circle(annotated_snapshot, pt, 7, (50, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(annotated_snapshot, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)

        badge_text = f"POLEGAR {step_id.upper()} — ÂNGULO: {angle.upper()}"
        cv2.rectangle(annotated_snapshot, (10, 10), (360, 38), (20, 20, 30), -1)
        cv2.rectangle(annotated_snapshot, (10, 10), (360, 38), (137, 180, 250), 1)
        cv2.putText(annotated_snapshot, badge_text, (18, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (137, 180, 250), 1, cv2.LINE_AA)

        cv2.imwrite(img_filename, annotated_snapshot)

        self.step_subcaptures[angle] = {
            'pts_norm': avg_pts,
            'pts_raw': pts_raw if pts_raw is not None else np.zeros((21, 3)),
            'image_path': img_filename,
            'snapshot': annotated_snapshot
        }

    def _finalize_step_and_review(self, step: Dict[str, Any]):
        """Funde a captura frontal (X, Y) com a captura lateral (Z) e aplica os comprimentos rígidos."""
        step_id = step['id']
        front_entry = self.step_subcaptures.get('frontal', {})
        lat_entry = self.step_subcaptures.get('lateral', {})

        pts_front_raw = front_entry.get('pts_norm', np.zeros((21, 3)))
        pts_lat_raw = lat_entry.get('pts_norm', np.zeros((21, 3)))

        # 1. Alinhar a captura frontal no referencial canônico da palma
        pts_front_can, _ = to_canonical_palm_frame(pts_front_raw)

        # 2. Extrair a profundidade Z do perfil lateral
        pts_fused = self._fuse_frontal_and_lateral(pts_front_can, pts_lat_raw, step['thumb_state'])

        self.captured_data[step_id] = {
            'step_meta': step,
            'frontal': front_entry,
            'lateral': lat_entry,
            'pts_norm': pts_fused,
            'snapshot': front_entry.get('snapshot')
        }

        self.current_review_metrics, self.current_review_status = self._format_thumb_metrics(
            step, pts_fused
        )

        self.current_review_snapshot = front_entry.get('snapshot')
        self.review_yaw = 18.0
        self.review_pitch = -12.0
        self.state = "REVIEW"

        print(f"\n[FUSÃO FRONTAL + LATERAL CONCLUÍDA] -> {step['title']}")
        print(f"Status: {self.current_review_status}")
        for line in self.current_review_metrics:
            print(f"   {line}")
        print("-> Pressione [ESPAÇO] para Confirmar e Salvar, ou [R] para Refazer...")

    def _fuse_frontal_and_lateral(self, pts_front_can: np.ndarray, pts_lat: np.ndarray, thumb_state: int) -> np.ndarray:
        """
        Funde a orientação frontal (X, Y) com a profundidade sagital Z da vista de perfil:
        - 100% dos pontos lidos pela câmera:
            * Ponto 1 (CMC): X, Y lidos da frontal, Z lido da lateral (NÃO fixado em Z=0!).
            * Pontos 2, 3, 4: Direções 3D reais da câmera com comprimentos ósseos rígidos L1, L2, L3.
        """
        out = pts_front_can.copy()
        l1, l2, l3 = self.rigid_thumb_lengths

        # Eixo sagital da mão de perfil na imagem lateral
        p0 = pts_lat[0]
        p9 = pts_lat[9]
        v_palm = p9 - p0
        norm_palm = np.linalg.norm(v_palm)
        norm_palm_safe = norm_palm if norm_palm > 1e-6 else 1.0
        y_lat = v_palm / norm_palm_safe

        # Perpendicular 2D (eixo horizontal da câmera = eixo sagital da mão de perfil)
        x_sag = np.array([-y_lat[1], y_lat[0], 0.0])

        # Orientação do sinal: metacarpo do polegar em relação à palma (positivo = para a frente da palma)
        mcp_proj = float(np.dot(pts_lat[2] - p0, x_sag))
        sign = 1.0 if mcp_proj >= 0 else -1.0

        z_lat = {}
        for j in [1, 2, 3, 4]:
            z_lat[j] = float(sign * np.dot(pts_lat[j] - p0, x_sag) / norm_palm_safe)

        # Ajuste sagital específico por estado anatômico:
        if thumb_state == 0:
            # 1. Aberto Esticado (Palma Aberta):
            # O polegar está no plano da palma; o Z sagital permanece estritamente coplanar à palma (Z ~ 0)
            z_cmc = 0.0
            z_mcp = 0.0
            z_ip  = 0.0
            z_tip = 0.0
        elif thumb_state == 1:
            # 2. Junto aos Dedos (Fechado / Aduzido):
            # O CMC e o polegar ficam levemente à frente ou alinhados ao indicador
            z_cmc = float(np.clip(z_lat[1], 0.02, 0.20))
            z_mcp = float(np.clip(z_lat[2], z_cmc + 0.01, z_cmc + 0.15))
            z_ip  = float(np.clip(z_lat[3], z_mcp + 0.01, z_mcp + 0.10))
            z_tip = float(np.clip(z_lat[4], z_ip, z_ip + 0.08))
        else:
            # 3. Transversal (Oposição Cruzando a Palma):
            # O CMC e todo o polegar projetam-se pronunciadamente para a FRENTE da palma da mão!
            # Ponto 1 (CMC) NÃO fica no mesmo plano do pulso (Z_cmc ~ +0.25 a +0.45)
            z_cmc = float(np.clip(z_lat[1], 0.15, 0.45))
            z_mcp = float(np.clip(z_lat[2], z_cmc + 0.05, 0.65))
            z_ip  = float(np.clip(z_lat[3], z_mcp + 0.02, 0.75))
            z_tip = float(np.clip(z_lat[4], 0.30, 0.70))

        # Pontos 3D lidos 100% da fusão câmera frontal + lateral
        Q = np.zeros((5, 3), dtype=np.float64)
        Q[0] = np.array([0.0, 0.0, 0.0])
        Q[1] = np.array([pts_front_can[1, 0], pts_front_can[1, 1], z_cmc])
        Q[2] = np.array([pts_front_can[2, 0], pts_front_can[2, 1], z_mcp])
        Q[3] = np.array([pts_front_can[3, 0], pts_front_can[3, 1], z_ip])
        Q[4] = np.array([pts_front_can[4, 0], pts_front_can[4, 1], z_tip])

        # Ponto 1 (CMC) é 100% o ponto lido pela câmera (NÃO template, NÃO Z=0 fixo!)
        out[1] = Q[1].copy()

        # Reconstruir falanges mantendo os comprimentos rígidos L1, L2, L3 nas direções lidas da câmera
        v1 = Q[2] - Q[1]
        u1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-6 else np.array([-1.0, 0.0, 0.0])
        out[2] = out[1] + l1 * u1

        v2 = Q[3] - Q[2]
        u2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-6 else u1
        out[3] = out[2] + l2 * u2

        v3 = Q[4] - Q[3]
        u3 = v3 / np.linalg.norm(v3) if np.linalg.norm(v3) > 1e-6 else u2
        out[4] = out[3] + l3 * u3

        return out

    def _advance_step(self):
        self.current_step_idx += 1
        self.current_sub_angle = "FRONTAL"
        self.step_subcaptures = {}
        self.state = "CAPTURING"
        self.stability_start_time = None
        self.stable_frame_buffer = []
        if self.current_step_idx < len(THUMB_CALIBRATION_STEPS):
            next_step = THUMB_CALIBRATION_STEPS[self.current_step_idx]
            print(f"\n=======================================================")
            print(f"Iniciando Passo [{self.current_step_idx+1}/{len(THUMB_CALIBRATION_STEPS)}]: {next_step['title']}")
            print(f"Etapa 1/2: VISTA FRONTAL (Palma voltada de frente para a câmera)")
            print(f"Instrução: {next_step['target_action']}")
            print(f"=======================================================")

    def _format_thumb_metrics(
        self,
        step: Dict[str, Any],
        pts: np.ndarray
    ) -> Tuple[List[str], str]:
        metrics = []
        status = "Adequado"

        cmc_flex = joint_flexion(pts[0], pts[1], pts[2])
        mcp_flex = joint_flexion(pts[1], pts[2], pts[3])
        ip_flex = joint_flexion(pts[2], pts[3], pts[4])

        dist_tip_ind = float(np.linalg.norm(pts[4] - pts[5]))
        dist_tip_mid = float(np.linalg.norm(pts[4] - pts[9]))
        z_tip = float(pts[4, 2])

        metrics.append(f"CMC Flexão / Adução: {cmc_flex:.1f}°")
        metrics.append(f"MCP Articulação:     {mcp_flex:.1f}°")
        metrics.append(f"IP Falange Distal:   {ip_flex:.1f}°")
        metrics.append(f"Distância ao Indicador: {dist_tip_ind:.3f}")
        metrics.append(f"Distância ao Médio:     {dist_tip_mid:.3f}")
        metrics.append(f"Profundidade Z Ponta:   {z_tip:+.3f}")
        metrics.append(f"Comprimentos Rígidos: L1={self.rigid_thumb_lengths[0]:.2f}, L2={self.rigid_thumb_lengths[1]:.2f}, L3={self.rigid_thumb_lengths[2]:.2f}")

        return metrics, status

    def _render_capturing_hud(
        self,
        frame: np.ndarray,
        step: Dict[str, Any],
        has_hand: bool,
        pts_norm: Optional[np.ndarray]
    ) -> np.ndarray:
        h, w, _ = frame.shape
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        top_h = 175
        draw.rectangle([(0, 0), (w, top_h)], fill=(17, 17, 27, 235))
        draw.line([(0, top_h), (w, top_h)], fill=(137, 180, 250, 255), width=2)

        is_front = (self.current_sub_angle == "FRONTAL")
        badge_text = f"PASSO {self.current_step_idx + 1}/{len(THUMB_CALIBRATION_STEPS)}  —  ETAPA {'1/2: VISTA FRONTAL (DE FRENTE)' if is_front else '2/2: VISTA LATERAL (PERFIL 90°)'}"
        badge_col = (166, 227, 161, 255) if is_front else (250, 179, 135, 255)

        draw.text((25, 8), badge_text, font=self.hud.get_font(13, bold=True), fill=badge_col)
        draw.text((25, 28), step['title'], font=self.hud.get_font(18, bold=True), fill=(137, 180, 250, 255))

        line_height = 22
        curr_y = 56
        posture_txt = step['posture_front'] if is_front else step['posture_lat']
        draw.text((25, curr_y), f"Posição: {posture_txt}", font=self.hud.get_font(14, bold=False), fill=(205, 214, 244, 255))
        curr_y += line_height

        action_lines = textwrap.wrap(step['target_action'], width=95)
        for i, aline in enumerate(action_lines):
            prefix = "Ação:    " if i == 0 else "         "
            draw.text((25, curr_y), prefix + aline, font=self.hud.get_font(14, bold=True), fill=(166, 227, 161, 255))
            curr_y += line_height

        draw.text((25, curr_y), f"Alvo:    {step['expected_summary']}", font=self.hud.get_font(14, bold=True), fill=(249, 226, 175, 255))
        curr_y += line_height

        draw.text((25, curr_y), f"Outros:  {step['other_fingers']}", font=self.hud.get_font(13, bold=False), fill=(186, 194, 222, 255))

        # Painel Inferior
        bot_h = 65
        draw.rectangle([(0, h - bot_h), (w, h)], fill=(17, 17, 27, 240))
        draw.line([(0, h - bot_h), (w, h - bot_h)], fill=(69, 71, 90, 255), width=1)

        f_ctrl = self.hud.get_font(14, bold=True)
        f_ctrl_sub = self.hud.get_font(12, bold=False)
        ctrl_str = f"[ESPAÇO] Capturar Vista {self.current_sub_angle}  |  [R] Repetir Passo  |  [B] Voltar  |  [Q / ESC] Sair"
        draw.text((25, h - 45), ctrl_str, font=f_ctrl, fill=(245, 194, 231, 255))
        draw.text((25, h - 24), "Segure a pose estável por 1.2s para disparo automático da vista atual.", font=f_ctrl_sub, fill=(186, 194, 222, 255))

        # Telemetria ao Vivo
        if has_hand and pts_norm is not None:
            pts_canon, _ = to_canonical_palm_frame(pts_norm)
            live_metrics, _ = self._format_thumb_metrics(step, pts_canon)
            card_w = 340
            card_h = 30 + len(live_metrics) * 20
            cx = w - card_w - 20
            cy = top_h + 15

            draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], fill=(24, 24, 37, 215))
            draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], outline=(137, 180, 250, 255), width=1)

            draw.text((cx + 12, cy + 8), f"TELEMETRIA AO VIVO ({self.current_sub_angle})", font=self.hud.get_font(13, bold=True), fill=(249, 226, 175, 255))
            for i, line in enumerate(live_metrics):
                draw.text((cx + 12, cy + 28 + i * 20), line, font=self.hud.get_font(12, bold=False), fill=(205, 214, 244, 255))

            if self.stability_start_time is not None:
                elapsed = time.time() - self.stability_start_time
                pct = min(1.0, elapsed / self.REQUIRED_STABLE_TIME)
                bar_w = 340
                bar_h = 16
                bx = (w - bar_w) // 2
                by = h - bot_h - 30

                draw.rectangle([(bx, by), (bx + bar_w, by + bar_h)], fill=(40, 40, 60, 220))
                draw.rectangle([(bx, by), (bx + int(bar_w * pct), by + bar_h)], fill=badge_col)
                draw.rectangle([(bx, by), (bx + bar_w, by + bar_h)], outline=(205, 214, 244, 255), width=1)

                draw.text((bx + 80, by - 18), f"ESTABILIZANDO {self.current_sub_angle}: {int(pct * 100)}%", font=self.hud.get_font(12, bold=True), fill=badge_col)
        else:
            f_warn = self.hud.get_font(18, bold=True)
            draw.text(((w // 2) - 220, h // 2), "AGUARDANDO DETECÇÃO DA MÃO NA CÂMERA...", font=f_warn, fill=(243, 139, 168, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _render_waiting_lateral_hud(self, frame: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        """Renderiza a tela de transição com modal explícito para o usuário girar a mão 90° de perfil."""
        h, w, _ = frame.shape
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Modal central de instrução
        modal_w = 680
        modal_h = 240
        mx = (w - modal_w) // 2
        my = (h - modal_h) // 2

        # Sombra e fundo
        draw.rectangle([(mx - 4, my - 4), (mx + modal_w + 4, my + modal_h + 4)], fill=(10, 10, 15, 180))
        draw.rectangle([(mx, my), (mx + modal_w, my + modal_h)], fill=(24, 24, 37, 245))
        draw.rectangle([(mx, my), (mx + modal_w, my + modal_h)], outline=(250, 179, 135, 255), width=2)

        # Ícone e Título
        draw.text((mx + 30, my + 25), "[OK] VISTA FRONTAL REGISTRADA COM SUCESSO!", font=self.hud.get_font(16, bold=True), fill=(166, 227, 161, 255))
        draw.text((mx + 30, my + 60), "AGORA GIRE A MÃO 90° DE PERFIL (VISTA LATERAL)", font=self.hud.get_font(18, bold=True), fill=(250, 179, 135, 255))

        draw.text((mx + 30, my + 100), "-> Mantenha o formato do polegar idêntico, apenas gire o punho 90° de lado.", font=self.hud.get_font(14, bold=False), fill=(205, 214, 244, 255))
        draw.text((mx + 30, my + 125), "-> A câmera usará esta visão lateral para calibrar a profundidade Z real.", font=self.hud.get_font(14, bold=False), fill=(205, 214, 244, 255))

        # Botão de Ação
        btn_w = 620
        btn_h = 42
        bx = mx + 30
        by = my + 165
        draw.rectangle([(bx, by), (bx + btn_w, by + btn_h)], fill=(250, 179, 135, 255))
        draw.text((bx + 110, by + 10), "QUANDO ESTIVER DE PERFIL, PRESSIONE [ESPAÇO] PARA INICIAR", font=self.hud.get_font(14, bold=True), fill=(17, 17, 27, 255))

        draw.text((mx + 30, my + modal_h + 15), "[R] Recapturar vista frontal  |  [Q] Sair", font=self.hud.get_font(12, bold=False), fill=(186, 194, 222, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _render_review_hud(self, frame: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        h, w, _ = frame.shape
        step_id = step['id']
        cap_entry = self.captured_data.get(step_id, {})
        pts_3d = cap_entry.get('pts_norm')

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (17, 16, 24)

        # 1. Top Bar
        top_h = 65
        cv2.rectangle(canvas, (0, 0), (w, top_h), (24, 24, 37), -1)
        cv2.line(canvas, (0, top_h), (w, top_h), (166, 227, 161), 2)

        # 2. Viewports centrais
        vp_w = 440
        vp_h = 490
        vp_y = 75
        vp1_x = 350
        vp2_x = 350 + vp_w + 20

        if pts_3d is not None and len(pts_3d) == 21:
            vp_front = self._render_viewport_3d(pts_3d, vp_w, vp_h, self.review_yaw, self.review_pitch, "VISTA 1: FRONTAL / ORBITAL (DADOS FRONTAIS)")
            canvas[vp_y:vp_y + vp_h, vp1_x:vp1_x + vp_w] = vp_front

            vp_side = self._render_viewport_3d(pts_3d, vp_w, vp_h, 90.0, 0.0, "VISTA 2: PERFIL LATERAL 90° (DADOS LATERAIS / PROFUNDIDADE Z)")
            canvas[vp_y:vp_y + vp_h, vp2_x:vp2_x + vp_w] = vp_side

        # 3. Card esquerdo de telemetria
        card_x = 20
        card_w = 310
        card_h = vp_h
        cv2.rectangle(canvas, (card_x, vp_y), (card_x + card_w, vp_y + card_h), (24, 24, 37), -1)
        cv2.rectangle(canvas, (card_x, vp_y), (card_x + card_w, vp_y + card_h), (69, 71, 90), 1)

        # 4. Bottom Bar
        bot_h = 65
        cv2.rectangle(canvas, (0, h - bot_h), (w, h), (24, 24, 37), -1)
        cv2.line(canvas, (0, h - bot_h), (w, h - bot_h), (69, 71, 90), 1)

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Header
        draw.text((25, 12), f"REVISÃO DE CAPTURA DO POLEGAR (DUPLO ÂNGULO)  —  {step['title']}", font=self.hud.get_font(18, bold=True), fill=(166, 227, 161, 255))
        draw.text((25, 38), "Modelo 3D reconstruído: Eixos X/Y extraídos de frente e Eixo Z extraído de perfil 90°.", font=self.hud.get_font(12, bold=False), fill=(186, 194, 222, 255))

        # Card de Telemetria
        draw.text((card_x + 15, vp_y + 15), "DADOS BIOMECÂNICOS", font=self.hud.get_font(14, bold=True), fill=(249, 226, 175, 255))
        for i, line in enumerate(self.current_review_metrics):
            draw.text((card_x + 15, vp_y + 45 + i * 24), line, font=self.hud.get_font(12, bold=False), fill=(205, 214, 244, 255))

        draw.line([(card_x + 10, vp_y + card_h - 100), (card_x + card_w - 10, vp_y + card_h - 100)], fill=(69, 71, 90, 255), width=1)
        draw.text((card_x + 15, vp_y + card_h - 90), "CONTROLES 3D NA REVISÃO:", font=self.hud.get_font(12, bold=True), fill=(245, 194, 231, 255))
        draw.text((card_x + 15, vp_y + card_h - 70), "[A] / [D] : Girar Yaw (-/+ 8°)", font=self.hud.get_font(11, bold=False), fill=(186, 194, 222, 255))
        draw.text((card_x + 15, vp_y + card_h - 52), "[W] / [X] : Inclinar Pitch (-/+ 6°)", font=self.hud.get_font(11, bold=False), fill=(186, 194, 222, 255))

        # Bottom Bar
        ctrl_str = "[ESPAÇO] Confirmar e Avançar  |  [R] Repetir Passo Completo  |  [B] Voltar  |  [Q] Sair"
        draw.text((25, h - 45), ctrl_str, font=self.hud.get_font(15, bold=True), fill=(166, 227, 161, 255))
        draw.text((25, h - 22), "Ao confirmar todos os 3 passos, as sementes serão automaticamente atualizadas.", font=self.hud.get_font(12, bold=False), fill=(186, 194, 222, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _render_viewport_3d(self, pts_3d: np.ndarray, vp_w: int, vp_h: int, yaw: float, pitch: float, title: str) -> np.ndarray:
        canvas = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
        canvas[:] = (20, 18, 28)

        cx = vp_w // 2
        scale = min(vp_w, vp_h) * 0.40
        cy = int(vp_h * 0.50 + 0.88 * scale)

        R = rot_x(pitch).dot(rot_y(yaw))
        pts_rot = pts_3d.dot(R.T)
        screen_pts = []
        depths = []
        for i in range(21):
            sx = int(cx + pts_rot[i, 0] * scale)
            sy = int(cy + pts_rot[i, 1] * scale)
            screen_pts.append((sx, sy))
            depths.append(pts_rot[i, 2])

        # Grade isométrica
        grid_y = 0.10
        for gx in np.linspace(-0.6, 0.6, 5):
            p1 = np.array([gx, grid_y, -0.6]).dot(R.T)
            p2 = np.array([gx, grid_y, +0.6]).dot(R.T)
            x1, y1 = int(cx + p1[0] * scale), int(cy + p1[1] * scale)
            x2, y2 = int(cx + p2[0] * scale), int(cy + p2[1] * scale)
            cv2.line(canvas, (x1, y1), (x2, y2), (35, 32, 45), 1, cv2.LINE_AA)

        # Palma
        for i1, i2 in PALM_BONES:
            cv2.line(canvas, screen_pts[i1], screen_pts[i2], (90, 85, 105), 2, cv2.LINE_AA)

        # Falanges dos dedos
        for fname in ['Index', 'Middle', 'Ring', 'Pinky']:
            for i1, i2 in [(0, 5), (5, 6), (6, 7), (7, 8)] if fname == 'Index' else (
                [(0, 9), (9, 10), (10, 11), (11, 12)] if fname == 'Middle' else (
                [(0, 13), (13, 14), (14, 15), (15, 16)] if fname == 'Ring' else
                [(0, 17), (17, 18), (18, 19), (19, 20)])):
                cv2.line(canvas, screen_pts[i1], screen_pts[i2], (80, 80, 95), 2, cv2.LINE_AA)

        # Destaque especial do Polegar em Laranja com espessura reforçada
        for i1, i2 in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            cv2.line(canvas, screen_pts[i1], screen_pts[i2], (40, 140, 255), 4, cv2.LINE_AA)

        for j in range(21):
            pt = screen_pts[j]
            if j in [1, 2, 3, 4]:
                cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 8, (40, 140, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, 3, (160, 160, 175), -1, cv2.LINE_AA)

        # Cabeçalho do viewport com tipografia Unicode UTF-8
        pil_vp = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw_vp = ImageDraw.Draw(pil_vp)
        draw_vp.rectangle([(0, 0), (vp_w, 24)], fill=(28, 26, 38, 255))
        draw_vp.line([(0, 24), (vp_w, 24)], fill=(137, 180, 250, 255), width=1)
        draw_vp.text((12, 3), title, font=self.hud.get_font(12, bold=True), fill=(200, 200, 240, 255))
        draw_vp.rectangle([(0, 0), (vp_w - 1, vp_h - 1)], outline=(137, 180, 250, 255), width=1)
        return cv2.cvtColor(np.array(pil_vp), cv2.COLOR_RGB2BGR)

    def compile_and_save_thumb(self):
        print("\n" + "="*70)
        print("  GRAVANDO CONFIGURAÇÕES DO POLEGAR E ATUALIZANDO PIPELINE")
        print("="*70)

        data = {}
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if 'thumb_extracted' not in data:
            data['thumb_extracted'] = {}
        if 'captured_landmarks' not in data:
            data['captured_landmarks'] = {}

        for step in THUMB_CALIBRATION_STEPS:
            step_id = step['id']
            if step_id in self.captured_data:
                entry = self.captured_data[step_id]
                pts = entry['pts_norm']
                pts_list = clean_json_value(pts)

                data['thumb_extracted'][step_id] = pts_list
                data['captured_landmarks'][step_id] = {
                    'pts_norm': pts_list,
                    'frontal': clean_json_value(entry.get('frontal', {}).get('pts_norm')),
                    'lateral': clean_json_value(entry.get('lateral', {}).get('pts_norm')),
                    'timestamp': float(round(time.time(), 3))
                }
                print(f"  ✓ Salvo estado do polegar (Frontal + Lateral): {step_id}")

        clean_data = clean_json_value(data)

        temp_calib_file = CALIBRATION_FILE + ".tmp"
        with open(temp_calib_file, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)

        if os.path.exists(temp_calib_file):
            if os.path.exists(CALIBRATION_FILE):
                os.replace(temp_calib_file, CALIBRATION_FILE)
            else:
                os.rename(temp_calib_file, CALIBRATION_FILE)

        print(f"\n[SUCESSO] Configurações salvas em: {CALIBRATION_FILE}")

        # Disparar regeneração de seeds.json com o motor cinemático
        print("[PIPELINE] Regenerando catálogo seeds.json com as novas limitações do polegar...")
        try:
            from kinematic_seed_generator import HandKinematicsDirect
            generator = HandKinematicsDirect.from_calibration_file(CALIBRATION_FILE)
            generator.export_seeds_json(SEEDS_FILE)
            print("  ✓ seeds.json regenerado com sucesso!")
        except Exception as e:
            print(f"  ✗ Erro ao regenerar seeds.json ({e})")

        # Disparar atualização dos relatórios visuais
        print("[PIPELINE] Atualizando relatórios visuais...")
        try:
            import generate_seed_limit_visualizations
            generate_seed_limit_visualizations.main()
            print("  ✓ Painéis de visualização atualizados com sucesso!")
        except Exception as e:
            print(f"  ✗ Aviso: não foi possível atualizar relatórios automaticamente ({e})")

        print("="*70 + "\n")


# ---------------------------------------------------------------------------
# PONTO DE ENTRADA CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrador Biomecânico Guiado do Polegar (LIBRAS)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão: 0)")
    args = parser.parse_args()

    calibrator = GuidedThumbCalibrator(camera_idx=args.camera)
    success = calibrator.run_interactive()
    if success:
        print("[FINALIZADO] Calibração do polegar concluída com êxito!")
    else:
        print("[FINALIZADO] Calibração do polegar não foi concluída.")


if __name__ == "__main__":
    main()
