#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided Hand Calibrator (guided_hand_calibrator.py)
=================================================
Assistente interativo em tempo real (OpenCV + MediaPipe Hands) para guiar o usuário
na calibração cirúrgica de cada variável anatômica individual da mão:
1. Baseline: Proporções e comprimentos ósseos das falanges
2. Dedos Longos (Mindinho, Anelar, Médio, Indicador):
   - Estágio 0: Estendido (Reto)
   - Estágio 1: Curvado / Concha (Arco suave)
   - Estágio 2: Gancho / Hook (Base reta, pontas dobradas)
   - Estágio 3: Plataforma / Tabletop (Base a 90°, pontas retas)
   - Estágio 4: Fechado / Punho (Totalmente dobrado)
3. Aberturas Laterais (Spreads entre pares de dedos)
4. Movimentação do Polegar:
   - No plano da mão (F=0) com ponta reta (P=0) e dobrada (P=1)
   - Oposição transversal/perpendicular (F=1) com ponta reta (P=0) e dobrada (P=1)

Os dados capturados são filtrados temporalmente (rejeição de jitter) e exportados
para calibration_settings.json, com geração automática de seeds.json.
"""

import os
import sys
import json
import time
import math
import argparse
import cv2
import numpy as np
import mediapipe as mp
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

# Índices das juntas por dedo no padrão MediaPipe
FINGER_JOINTS = {
    'Thumb':  [0, 1, 2, 3, 4],     # Wrist, CMC, MCP, IP, TIP
    'Index':  [0, 5, 6, 7, 8],     # Wrist, MCP, PIP, DIP, TIP
    'Middle': [0, 9, 10, 11, 12],
    'Ring':   [0, 13, 14, 15, 16],
    'Pinky':  [0, 17, 18, 19, 20]
}

# ---------------------------------------------------------------------------
# CATÁLOGO DE PASSOS DE CALIBRAÇÃO (WIZARD)
# ---------------------------------------------------------------------------

CALIBRATION_STEPS = [
    # 0. Baseline
    {
        'id': 'baseline_open',
        'category': 'baseline',
        'title': 'MÃO ESPALMADA (BASELINE)',
        'instruction': 'Abra a mão completamente com dedos retos para medir as proporções ósseas.',
        'target_finger': None,
        'expected_stage': 0
    },
    # 1. Indicador (Index)
    {
        'id': 'index_s0',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 0,
        'title': 'INDICADOR: Estágio 0 (ESTENDIDO)',
        'instruction': 'Mantenha o INDICADOR totalmente RETO (esticado para cima).',
        'target_finger': 'Index',
        'expected_stage': 0
    },
    {
        'id': 'index_s1',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 1,
        'title': 'INDICADOR: Estágio 1 (CURVADO / CONCHA)',
        'instruction': 'Curve levemente o INDICADOR num arco suave contínuo.',
        'target_finger': 'Index',
        'expected_stage': 1
    },
    {
        'id': 'index_s2',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 2,
        'title': 'INDICADOR: Estágio 2 (GANCHO / HOOK)',
        'instruction': 'Base (MCP) RETA e pontas (PIP/DIP) DOBRADAS a ~90 graus (Gancho).',
        'target_finger': 'Index',
        'expected_stage': 2
    },
    {
        'id': 'index_s3',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 3,
        'title': 'INDICADOR: Estágio 3 (PLATAFORMA / TABLETOP)',
        'instruction': 'Base (MCP) DOBRADA a 90 graus, mas falanges pontas RETAS (Mesa).',
        'target_finger': 'Index',
        'expected_stage': 3
    },
    {
        'id': 'index_s4',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 4,
        'title': 'INDICADOR: Estágio 4 (FECHADO / PUNHO)',
        'instruction': 'Dobre totalmente o INDICADOR colado na palma (Punho).',
        'target_finger': 'Index',
        'expected_stage': 4
    },
    # 2. Médio (Middle)
    {
        'id': 'middle_s0',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 0,
        'title': 'MÉDIO: Estágio 0 (ESTENDIDO)',
        'instruction': 'Mantenha o dedo MÉDIO totalmente RETO (esticado).',
        'target_finger': 'Middle',
        'expected_stage': 0
    },
    {
        'id': 'middle_s1',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 1,
        'title': 'MÉDIO: Estágio 1 (CURVADO / CONCHA)',
        'instruction': 'Curve levemente o dedo MÉDIO em arco contínuo.',
        'target_finger': 'Middle',
        'expected_stage': 1
    },
    {
        'id': 'middle_s2',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 2,
        'title': 'MÉDIO: Estágio 2 (GANCHO / HOOK)',
        'instruction': 'Base (MCP) RETA e pontas do dedo MÉDIO DOBRADAS (Gancho).',
        'target_finger': 'Middle',
        'expected_stage': 2
    },
    {
        'id': 'middle_s3',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 3,
        'title': 'MÉDIO: Estágio 3 (PLATAFORMA / TABLETOP)',
        'instruction': 'Base (MCP) DOBRADA a 90 graus, mantendo o dedo MÉDIO reto (Mesa).',
        'target_finger': 'Middle',
        'expected_stage': 3
    },
    {
        'id': 'middle_s4',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 4,
        'title': 'MÉDIO: Estágio 4 (FECHADO / PUNHO)',
        'instruction': 'Dobre totalmente o dedo MÉDIO colado na palma (Fechado).',
        'target_finger': 'Middle',
        'expected_stage': 4
    },
    # 3. Anelar (Ring)
    {
        'id': 'ring_s0',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 0,
        'title': 'ANELAR: Estágio 0 (ESTENDIDO)',
        'instruction': 'Mantenha o dedo ANELAR totalmente RETO.',
        'target_finger': 'Ring',
        'expected_stage': 0
    },
    {
        'id': 'ring_s1',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 1,
        'title': 'ANELAR: Estágio 1 (CURVADO / CONCHA)',
        'instruction': 'Curve levemente o dedo ANELAR em arco contínuo.',
        'target_finger': 'Ring',
        'expected_stage': 1
    },
    {
        'id': 'ring_s2',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 2,
        'title': 'ANELAR: Estágio 2 (GANCHO / HOOK)',
        'instruction': 'Base (MCP) RETA e pontas do ANELAR dobradas (Gancho).',
        'target_finger': 'Ring',
        'expected_stage': 2
    },
    {
        'id': 'ring_s3',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 3,
        'title': 'ANELAR: Estágio 3 (PLATAFORMA / TABLETOP)',
        'instruction': 'Base (MCP) dobrada a 90 graus, com pontas do ANELAR retas.',
        'target_finger': 'Ring',
        'expected_stage': 3
    },
    {
        'id': 'ring_s4',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 4,
        'title': 'ANELAR: Estágio 4 (FECHADO / PUNHO)',
        'instruction': 'Dobre totalmente o dedo ANELAR colado na palma.',
        'target_finger': 'Ring',
        'expected_stage': 4
    },
    # 4. Mindinho (Pinky)
    {
        'id': 'pinky_s0',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 0,
        'title': 'MINDINHO: Estágio 0 (ESTENDIDO)',
        'instruction': 'Mantenha o MINDINHO totalmente RETO (esticado).',
        'target_finger': 'Pinky',
        'expected_stage': 0
    },
    {
        'id': 'pinky_s1',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 1,
        'title': 'MINDINHO: Estágio 1 (CURVADO / CONCHA)',
        'instruction': 'Curve suavemente o MINDINHO em arco.',
        'target_finger': 'Pinky',
        'expected_stage': 1
    },
    {
        'id': 'pinky_s2',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 2,
        'title': 'MINDINHO: Estágio 2 (GANCHO / HOOK)',
        'instruction': 'Base (MCP) reta e pontas do MINDINHO dobradas (Gancho).',
        'target_finger': 'Pinky',
        'expected_stage': 2
    },
    {
        'id': 'pinky_s3',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 3,
        'title': 'MINDINHO: Estágio 3 (PLATAFORMA / TABLETOP)',
        'instruction': 'Base (MCP) a 90 graus, mantendo o MINDINHO estendido.',
        'target_finger': 'Pinky',
        'expected_stage': 3
    },
    {
        'id': 'pinky_s4',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 4,
        'title': 'MINDINHO: Estágio 4 (FECHADO / PUNHO)',
        'instruction': 'Dobre totalmente o MINDINHO colado na palma.',
        'target_finger': 'Pinky',
        'expected_stage': 4
    },
    # 5. Aberturas (Spreads)
    {
        'id': 'spread_open',
        'category': 'spread',
        'spread_state': 0,
        'title': 'ABERTURA: Dedos ABERTOS em Leque (A=0)',
        'instruction': 'Separe todos os dedos o máximo possível para os lados (Leque).',
        'target_finger': None
    },
    {
        'id': 'spread_closed',
        'category': 'spread',
        'spread_state': 1,
        'title': 'ABERTURA: Dedos JUNTOS em Paralelo (A=1)',
        'instruction': 'Junte todos os 4 dedos retos colados lado a lado (Sem abertura).',
        'target_finger': None
    },
    # 6. Polegar (Thumb: Plano vs Oposição Transversal e IP)
    {
        'id': 'thumb_f0_p0',
        'category': 'thumb',
        'f': 0, 'p': 0,
        'title': 'POLEGAR: No Plano da Mão (F=0, P=0)',
        'instruction': 'Polegar aberto ao lado no mesmo plano da mão, ponta RETA.',
        'target_finger': 'Thumb'
    },
    {
        'id': 'thumb_f0_p1',
        'category': 'thumb',
        'f': 0, 'p': 1,
        'title': 'POLEGAR: Plano com Ponta Dobrada (F=0, P=1)',
        'instruction': 'Polegar ao lado no plano da mão, mas com a PONTA (IP) DOBRADA.',
        'target_finger': 'Thumb'
    },
    {
        'id': 'thumb_f1_p0',
        'category': 'thumb',
        'f': 1, 'p': 0,
        'title': 'POLEGAR: Oposição Transversal (F=1, P=0)',
        'instruction': 'Polegar cruzando na frente da palma (perpendicular), ponta RETA.',
        'target_finger': 'Thumb'
    },
    {
        'id': 'thumb_f1_p1',
        'category': 'thumb',
        'f': 1, 'p': 1,
        'title': 'POLEGAR: Oposição com Ponta Dobrada (F=1, P=1)',
        'instruction': 'Polegar cruzando a palma em oposição profunda, ponta (IP) DOBRADA.',
        'target_finger': 'Thumb'
    }
]

# ---------------------------------------------------------------------------
# CLASSE PRINCIPAL DE CALIBRAÇÃO GUIADA
# ---------------------------------------------------------------------------

class GuidedHandCalibrator:
    def __init__(self, camera_idx: int = 0):
        self.camera_idx = camera_idx
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
        self.captured_data: Dict[str, Any] = {}
        self.stable_frame_buffer: List[np.ndarray] = []
        self.stability_start_time: Optional[float] = None
        self.REQUIRED_STABLE_TIME = 1.2  # 1.2 segundos segurando a pose
        self.flash_timer = 0

    def run_interactive(self) -> bool:
        """Loop principal interativo via OpenCV."""
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir a câmera índice {self.camera_idx}.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "Calibrador Biomecanico Guiado - LIBRAS TCC"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*65)
        print("  CALIBRADOR BIOMECANICO GUIADO INICIADO")
        print("="*65)
        print("Controles:")
        print("  [ESPACO] : Forcar Captura do Passo Atual")
        print("  [R]      : Recapturar Passo Atual")
        print("  [B]      : Voltar ao Passo Anterior")
        print("  [S]      : Pular Passo (Fallback)")
        print("  [Q/ESC]  : Sair e Salvar Calibracao")
        print("="*65 + "\n")

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

                # Desenhar skeleton com MediaPipe
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

            # Processar passo atual
            step = CALIBRATION_STEPS[self.current_step_idx]
            is_captured = False

            # Lógica de auto-estabilização
            if has_hand and pts_norm is not None:
                if self.stability_start_time is None:
                    self.stability_start_time = time.time()
                    self.stable_frame_buffer = [pts_norm]
                else:
                    self.stable_frame_buffer.append(pts_norm)
                    elapsed = time.time() - self.stability_start_time
                    if elapsed >= self.REQUIRED_STABLE_TIME and len(self.stable_frame_buffer) >= 20:
                        self._save_step_capture(step, frame, pts_pixels)
                        is_captured = True
            else:
                self.stability_start_time = None
                self.stable_frame_buffer = []

            # Renderizar HUD Visual
            self._render_hud(frame, step, has_hand, pts_norm)

            if is_captured:
                self._advance_step()

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), 27]:  # ESC ou Q
                print("[LOG] Finalizado pelo usuario.")
                break
            elif key == 32:  # ESPAÇO
                if has_hand and pts_norm is not None:
                    self.stable_frame_buffer.append(pts_norm)
                    self._save_step_capture(step, frame, pts_pixels)
                    self._advance_step()
            elif key == ord('r'):  # Recapturar
                step_id = step['id']
                if step_id in self.captured_data:
                    del self.captured_data[step_id]
                self.stability_start_time = None
                self.stable_frame_buffer = []
                print(f"[REPETIR] Recapturando: {step['title']}")
            elif key == ord('b'):  # Voltar
                if self.current_step_idx > 0:
                    self.current_step_idx -= 1
                    self.stability_start_time = None
                    self.stable_frame_buffer = []
                    print(f"[VOLTAR] Retornando ao passo: {CALIBRATION_STEPS[self.current_step_idx]['title']}")
            elif key == ord('s'):  # Pular
                print(f"[PULAR] Passo {step['title']} pulado.")
                self._advance_step()

            if self.current_step_idx >= len(CALIBRATION_STEPS):
                print("\n[SUCESSO] Todos os passos concluidos com exito!")
                break

        cap.release()
        cv2.destroyAllWindows()

        # Compilar e exportar calibration_settings.json
        self.compile_and_save_settings()
        return True

    def _advance_step(self):
        self.current_step_idx += 1
        self.stability_start_time = None
        self.stable_frame_buffer = []
        self.flash_timer = 15  # frames de flash verde
        if self.current_step_idx < len(CALIBRATION_STEPS):
            next_step = CALIBRATION_STEPS[self.current_step_idx]
            print(f"-> Avancando para [{self.current_step_idx+1}/{len(CALIBRATION_STEPS)}]: {next_step['title']}")

    def _save_step_capture(self, step: Dict[str, Any], frame: np.ndarray, pts_raw: np.ndarray):
        """Salva a foto para auditoria e acumula o ponto normalizado filtrado."""
        step_id = step['id']

        # Salvar snapshot da câmera
        img_filename = os.path.join(CAPTURES_DIR, f"{self.current_step_idx+1:02d}_{step_id}.png")
        cv2.imwrite(img_filename, frame)

        # Média das coordenadas do buffer estável
        avg_pts = np.mean(np.array(self.stable_frame_buffer), axis=0)

        self.captured_data[step_id] = {
            'step_meta': step,
            'pts_norm': avg_pts,
            'pts_raw': pts_raw,
            'image_path': img_filename
        }
        print(f"  [OK] Capturado: {step['title']} -> Salvo em {os.path.basename(img_filename)}")

    def _render_hud(self, frame: np.ndarray, step: Dict[str, Any], has_hand: bool, pts_norm: Optional[np.ndarray]):
        h, w, _ = frame.shape

        # Flash de sucesso
        if self.flash_timer > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            self.flash_timer -= 1

        # Barra Superior de Instrução
        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 30), -1)
        step_text = f"PASSO {self.current_step_idx + 1} de {len(CALIBRATION_STEPS)}: {step['title']}"
        cv2.putText(frame, step_text, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (137, 180, 250), 2)
        cv2.putText(frame, step['instruction'], (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (205, 214, 244), 2)

        # Barra Inferior de Status e Controles
        cv2.rectangle(frame, (0, h - 70), (w, h), (17, 17, 27), -1)
        ctrl_text = "[ESPACO] Capturar | [R] Repetir | [B] Voltar | [S] Pular | [Q/ESC] Salvar e Sair"
        cv2.putText(frame, ctrl_text, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (186, 194, 222), 1)

        # Telemetria da Junta no canto direito
        if has_hand and pts_norm is not None:
            # Calcular telemetria dependendo da categoria
            cat = step['category']
            telemetry_lines = []

            if cat == 'finger_flexion':
                f_name = step['finger']
                idxs = FINGER_JOINTS[f_name]
                j2_p = joint_flexion(pts_norm[idxs[0]], pts_norm[idxs[1]], pts_norm[idxs[2]])
                j3_p = joint_flexion(pts_norm[idxs[1]], pts_norm[idxs[2]], pts_norm[idxs[3]])
                j4_p = joint_flexion(pts_norm[idxs[2]], pts_norm[idxs[3]], pts_norm[idxs[4]])
                telemetry_lines.append(f"Dedo: {f_name.upper()}")
                telemetry_lines.append(f"MCP (J2): {j2_p:5.1f} deg")
                telemetry_lines.append(f"PIP (J3): {j3_p:5.1f} deg")
                telemetry_lines.append(f"DIP (J4): {j4_p:5.1f} deg")

            elif cat == 'spread':
                sp_pnk_rng = vec_angle(pts_norm[17] - pts_norm[0], pts_norm[13] - pts_norm[0])
                sp_rng_mid = vec_angle(pts_norm[13] - pts_norm[0], pts_norm[9] - pts_norm[0])
                sp_mid_idx = vec_angle(pts_norm[9] - pts_norm[0], pts_norm[5] - pts_norm[0])
                telemetry_lines.append(f"Pnk-Rng: {sp_pnk_rng:4.1f} deg")
                telemetry_lines.append(f"Rng-Mid: {sp_rng_mid:4.1f} deg")
                telemetry_lines.append(f"Mid-Idx: {sp_mid_idx:4.1f} deg")

            elif cat == 'thumb':
                dist_opp = np.linalg.norm(pts_norm[4] - pts_norm[9])
                ip_flex = joint_flexion(pts_norm[2], pts_norm[3], pts_norm[4])
                telemetry_lines.append(f"Thumb DistOpp: {dist_opp:4.2f}")
                telemetry_lines.append(f"Thumb IP Flex: {ip_flex:5.1f} deg")

            # Desenhar box de telemetria
            if telemetry_lines:
                box_w = 230
                box_h = 30 + len(telemetry_lines) * 24
                cv2.rectangle(frame, (w - box_w - 20, 115), (w - 20, 115 + box_h), (25, 25, 40), -1)
                cv2.rectangle(frame, (w - box_w - 20, 115), (w - 20, 115 + box_h), (137, 180, 250), 1)
                cv2.putText(frame, "TELEMETRIA AO VIVO", (w - box_w - 10, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (249, 226, 175), 1)
                for i, line in enumerate(telemetry_lines):
                    cv2.putText(frame, line, (w - box_w - 10, 163 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)

            # Barra de progresso de estabilidade
            if self.stability_start_time is not None:
                elapsed = time.time() - self.stability_start_time
                pct = min(1.0, elapsed / self.REQUIRED_STABLE_TIME)
                bar_w = 320
                bar_h = 16
                bx = (w - bar_w) // 2
                by = h - 105
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 60), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(bar_w * pct), by + bar_h), (166, 227, 161), -1)
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 220), 1)
                cv2.putText(frame, f"ESTABILIZANDO POSE: {int(pct*100)}%", (bx + 55, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (166, 227, 161), 1)
        else:
            cv2.putText(frame, "AGUARDANDO DETECCAO DA MAO...", ((w // 2) - 170, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (243, 139, 168), 2)

    def compile_and_save_settings(self, output_path: str = CALIBRATION_FILE) -> Dict[str, Any]:
        """Compila todas as medições capturadas e gera o calibration_settings.json."""
        print("\n[COMPILACAO] Processando limites anatomicos capturados...")

        # Baseline: comprimentos ósseos
        base_data = self.captured_data.get('baseline_open', None)
        if base_data is not None:
            ref_pts = base_data['pts_norm']
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

        # Estágios dos dedos longos (0 a 4)
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
                step_key = f"{f_name.lower()}_s{st}"
                if step_key in self.captured_data:
                    p = self.captured_data[step_key]['pts_norm']
                    # Se frame for degenerado/mock vazio, usar padrão canônico do estágio
                    if np.linalg.norm(p[idxs[4]] - p[idxs[1]]) < 1e-3:
                        stages[f_name][str(st)] = canon_stages[st]
                    else:
                        j2 = joint_flexion(p[idxs[0]], p[idxs[1]], p[idxs[2]])
                        j3 = joint_flexion(p[idxs[1]], p[idxs[2]], p[idxs[3]])
                        j4 = joint_flexion(p[idxs[2]], p[idxs[3]], p[idxs[4]])
                        stages[f_name][str(st)] = {
                            'J2_Pitch': float(j2),
                            'J3_Pitch': float(j3),
                            'J4_Pitch': float(j4)
                        }
                else:
                    stages[f_name][str(st)] = canon_stages[st]

        # Spreads (Aberturas)
        spread_angles = {
            'Pinky_Ring':   {'0': +10.0, '1': -15.0},
            'Ring_Middle':  {'0': +8.0,  '1': -10.0},
            'Middle_Index': {'0': -8.0,  '1': +10.0},
            'Index_Thumb':  {'0': -15.0, '1': +20.0}
        }
        if 'spread_open' in self.captured_data:
            p_open = self.captured_data['spread_open']['pts_norm']
            spread_angles['Pinky_Ring']['0']   = float(vec_angle(p_open[17] - p_open[0], p_open[13] - p_open[0]))
            spread_angles['Ring_Middle']['0']  = float(vec_angle(p_open[13] - p_open[0], p_open[9] - p_open[0]))
            spread_angles['Middle_Index']['0'] = float(-vec_angle(p_open[9] - p_open[0], p_open[5] - p_open[0]))

        if 'spread_closed' in self.captured_data:
            p_cls = self.captured_data['spread_closed']['pts_norm']
            spread_angles['Pinky_Ring']['1']   = float(-vec_angle(p_cls[17] - p_cls[0], p_cls[13] - p_cls[0]))
            spread_angles['Ring_Middle']['1']  = float(-vec_angle(p_cls[13] - p_cls[0], p_cls[9] - p_cls[0]))
            spread_angles['Middle_Index']['1'] = float(+vec_angle(p_cls[9] - p_cls[0], p_cls[5] - p_cls[0]))

        # Polegar (Thumb configuration)
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
            p_t0 = self.captured_data['thumb_f0_p1']['pts_norm']
            thumb_config['f0_ip_flex'] = float(joint_flexion(p_t0[2], p_t0[3], p_t0[4]))

        if 'thumb_f1_p1' in self.captured_data:
            p_t1 = self.captured_data['thumb_f1_p1']['pts_norm']
            thumb_config['f1_ip_flex'] = float(joint_flexion(p_t1[2], p_t1[3], p_t1[4]))

        # Montar manifesto completo
        calib_dict = {
            'metadata': {
                'generated_by': 'GuidedHandCalibrator_v2',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'steps_captured': len(self.captured_data),
                'total_steps': len(CALIBRATION_STEPS),
                'flexion_stages_count': 5
            },
            'stages': stages,
            'spread_angles': spread_angles,
            'thumb_config': thumb_config,
            'phalanx_lengths': phalanx_lengths,
            'avg_palm': avg_palm
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calib_dict, f, indent=2)

        print(f"[SUCESSO] Calibracao salva com sucesso em: {output_path}")

        # Gerar Seeds atualizadas automaticamente
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from kinematic_seed_generator import HandKinematicsDirect
            kinematics = HandKinematicsDirect.from_calibration_file(output_path)
            kinematics.export_seeds_json(SEEDS_FILE)
            print(f"[SEEDS] Catalogo seeds.json gerado a partir da calibracao!")
        except Exception as e:
            print(f"[AVISO] Nao foi possivel auto-gerar seeds.json: {e}")

        return calib_dict

    def run_mock_calibration(self, output_path: str = CALIBRATION_FILE) -> Dict[str, Any]:
        """Gera calibração simulada completa sem necessidade de câmera (para testes automatizados)."""
        print("[MOCK] Gerando calibracao anatomica 5-estagios simulada...")
        self.captured_data = {}  # Vazio para acionar todos os padrões canônicos perfeitamente
        return self.compile_and_save_settings(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrador Biomecanico Guiado da Mao - LIBRAS TCC")
    parser.add_argument('--camera', type=int, default=0, help="Indice da camera OpenCV (padrao 0)")
    parser.add_argument('--mock', action='store_true', help="Executa calibracao simulada sem camera (testes)")
    parser.add_argument('--output', type=str, default=CALIBRATION_FILE, help="Caminho do calibration_settings.json")
    args = parser.parse_args()

    calibrator = GuidedHandCalibrator(camera_idx=args.camera)
    if args.mock:
        calibrator.run_mock_calibration(output_path=args.output)
    else:
        calibrator.run_interactive()

if __name__ == '__main__':
    main()
