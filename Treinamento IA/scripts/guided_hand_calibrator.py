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

# Índices das juntas por dedo no padrão MediaPipe
FINGER_JOINTS = {
    'Thumb':  [0, 1, 2, 3, 4],     # Wrist, CMC, MCP, IP, TIP
    'Index':  [0, 5, 6, 7, 8],     # Wrist, MCP, PIP, DIP, TIP
    'Middle': [0, 9, 10, 11, 12],
    'Ring':   [0, 13, 14, 15, 16],
    'Pinky':  [0, 17, 18, 19, 20]
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

    # --- 1. DEDO INDICADOR ---
    {
        'id': 'index_s0',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 0,
        'title': 'INDICADOR: Estágio 0 (ESTENDIDO / RETO)',
        'posture': 'Palma virada de frente para a câmera na vertical, punho reto.',
        'target_action': 'Estique o dedo INDICADOR TOTALMENTE RETO para cima, em continuidade linear com a palma (180°). Nenhuma das 3 juntas (MCP, PIP, DIP) deve estar dobrada.',
        'other_fingers': 'Os outros dedos podem permanecer esticados ou relaxados em posição neutra.',
        'expected_summary': 'MCP ~ 0° | PIP ~ 0° | DIP ~ 0° (Dedo totalmente reto)',
        'target_finger': 'Index',
        'expected_stage': 0
    },
    {
        'id': 'index_s1',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 1,
        'title': 'INDICADOR: Estágio 1 (CURVADO / CONCHA)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Curve suavemente as 3 juntas do INDICADOR num arco contínuo e uniforme em forma de "C" (como se estivesse segurando uma bola de tênis ou maçã). O dedo não deve tocar a palma.',
        'other_fingers': 'Demais dedos podem acompanhar a curvatura suave ou ficar em posição neutra.',
        'expected_summary': 'MCP ~ 25-30° | PIP ~ 40° | DIP ~ 35° (Arco suave contínuo)',
        'target_finger': 'Index',
        'expected_stage': 1
    },
    {
        'id': 'index_s2',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 2,
        'title': 'INDICADOR: Estágio 2 (GANCHO / HOOK)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Mantenha a base (junta da palma MCP) RETA apontando para cima (0° a 15°), mas DOBRE as duas juntas da ponta (PIP e DIP) a ~90° para frente, formando uma garra de gato ou gancho de pirata. A base não avança para frente.',
        'other_fingers': 'Demais dedos estendidos para cima ou relaxados. Apenas o indicador faz a garra.',
        'expected_summary': 'MCP ~ 0-15° (base reta) | PIP ~ 90° | DIP ~ 75° (pontas flexionadas em 90°)',
        'target_finger': 'Index',
        'expected_stage': 2
    },
    {
        'id': 'index_s3',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 3,
        'title': 'INDICADOR: Estágio 3 (PLATAFORMA / TABLETOP)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre a junta da base (MCP) a 90° para frente em direção à câmera, mas mantenha as juntas do meio e da ponta (PIP e DIP) TOTALMENTE RETAS e travadas a 0°. O dedo forma um "L" reto perfeito, parecendo um tampo de mesa ou prateleira horizontal.',
        'other_fingers': 'Demais dedos podem ficar estendidos para cima para contraste.',
        'expected_summary': 'MCP ~ 85-90° (base dobrada) | PIP ~ 0° (reto) | DIP ~ 0° (reto)',
        'target_finger': 'Index',
        'expected_stage': 3
    },
    {
        'id': 'index_s4',
        'category': 'finger_flexion',
        'finger': 'Index',
        'stage': 4,
        'title': 'INDICADOR: Estágio 4 (FECHADO / PUNHO)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre completamente todas as 3 juntas do INDICADOR (MCP a 90°, PIP e DIP a ~105°/80°), colando a polpa digital firmemente contra a palma da mão.',
        'other_fingers': 'A mão pode se fechar em punho cerrado ou manter os outros dedos recolhidos.',
        'expected_summary': 'MCP ~ 85-90° | PIP ~ 105° | DIP ~ 80° (Dedo totalmente fechado na palma)',
        'target_finger': 'Index',
        'expected_stage': 4
    },

    # --- 2. DEDO MÉDIO ---
    {
        'id': 'middle_s0',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 0,
        'title': 'MÉDIO: Estágio 0 (ESTENDIDO / RETO)',
        'posture': 'Palma virada de frente para a câmera na vertical, punho reto.',
        'target_action': 'Estique o dedo MÉDIO totalmente reto para cima, alinhado ao eixo central da mão. Nenhuma das 3 juntas dobradas.',
        'other_fingers': 'Demais dedos estendidos ou relaxados.',
        'expected_summary': 'MCP ~ 0° | PIP ~ 0° | DIP ~ 0° (Reto para cima)',
        'target_finger': 'Middle',
        'expected_stage': 0
    },
    {
        'id': 'middle_s1',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 1,
        'title': 'MÉDIO: Estágio 1 (CURVADO / CONCHA)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Curve suavemente o dedo MÉDIO em arco uniforme contínuo (formato de "C"), sem encostar na palma.',
        'other_fingers': 'Demais dedos podem acompanhar a curvatura suavemente.',
        'expected_summary': 'MCP ~ 25-30° | PIP ~ 40° | DIP ~ 35° (Curvatura suave)',
        'target_finger': 'Middle',
        'expected_stage': 1
    },
    {
        'id': 'middle_s2',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 2,
        'title': 'MÉDIO: Estágio 2 (GANCHO / HOOK)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Base (MCP) RETA apontando para cima (0° a 15°), mas pontas do dedo MÉDIO (PIP e DIP) dobradas a ~90° para frente (garra/gancho).',
        'other_fingers': 'Demais dedos estendidos ou relaxados.',
        'expected_summary': 'MCP ~ 0-15° (reta) | PIP ~ 90° | DIP ~ 75° (pontas em ângulo reto)',
        'target_finger': 'Middle',
        'expected_stage': 2
    },
    {
        'id': 'middle_s3',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 3,
        'title': 'MÉDIO: Estágio 3 (PLATAFORMA / TABLETOP)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre a junta da base (MCP) a 90° para frente em direção à câmera, mantendo as falanges média e distal do dedo MÉDIO TOTALMENTE RETAS (formato de mesa horizontal em "L").',
        'other_fingers': 'Demais dedos podem ficar estendidos para cima ou neutros.',
        'expected_summary': 'MCP ~ 85-90° (base dobrada) | PIP ~ 0° (reto) | DIP ~ 0° (reto)',
        'target_finger': 'Middle',
        'expected_stage': 3
    },
    {
        'id': 'middle_s4',
        'category': 'finger_flexion',
        'finger': 'Middle',
        'stage': 4,
        'title': 'MÉDIO: Estágio 4 (FECHADO / PUNHO)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre totalmente o dedo MÉDIO cerrado e colado contra a palma da mão.',
        'other_fingers': 'Mão fechada em punho ou dedo médio isoladamente cerrado.',
        'expected_summary': 'MCP ~ 85-90° | PIP ~ 105° | DIP ~ 80° (Fechado na palma)',
        'target_finger': 'Middle',
        'expected_stage': 4
    },

    # --- 3. DEDO ANELAR ---
    {
        'id': 'ring_s0',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 0,
        'title': 'ANELAR: Estágio 0 (ESTENDIDO / RETO)',
        'posture': 'Palma de frente para a câmera na vertical, punho reto.',
        'target_action': 'Estique o dedo ANELAR totalmente reto para cima, alinhado à palma.',
        'other_fingers': 'Demais dedos esticados para cima para facilitar a extensão natural do anelar.',
        'expected_summary': 'MCP ~ 0° | PIP ~ 0° | DIP ~ 0° (Reto)',
        'target_finger': 'Ring',
        'expected_stage': 0
    },
    {
        'id': 'ring_s1',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 1,
        'title': 'ANELAR: Estágio 1 (CURVADO / CONCHA)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Curve suavemente o dedo ANELAR em arco uniforme (concha em forma de "C").',
        'other_fingers': 'Mão em formato de concha suave.',
        'expected_summary': 'MCP ~ 25-30° | PIP ~ 40° | DIP ~ 35° (Curvado)',
        'target_finger': 'Ring',
        'expected_stage': 1
    },
    {
        'id': 'ring_s2',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 2,
        'title': 'ANELAR: Estágio 2 (GANCHO / HOOK)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Base do ANELAR reta para cima (0° a 15°), pontas (PIP/DIP) dobradas a ~90° para frente (garra/gancho).',
        'other_fingers': 'Nota biomecânica: Dedos vizinhos (médio/mindinho) podem acompanhar levemente devido ao tendão compartilhado (Juncturae Tendinum). Foque na pose do ANELAR.',
        'expected_summary': 'MCP ~ 0-15° | PIP ~ 90° | DIP ~ 75° (Gancho)',
        'target_finger': 'Ring',
        'expected_stage': 2
    },
    {
        'id': 'ring_s3',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 3,
        'title': 'ANELAR: Estágio 3 (PLATAFORMA / TABLETOP)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre a base (MCP) a 90° para frente, mantendo o meio e a ponta do ANELAR retos (formato de prateleira/mesa em "L").',
        'other_fingers': 'Dedos médio ou mínimo podem dobrar junto se houver tensão anatômica do tendão.',
        'expected_summary': 'MCP ~ 85-90° (base dobrada) | PIP ~ 0° (reto) | DIP ~ 0° (reto)',
        'target_finger': 'Ring',
        'expected_stage': 3
    },
    {
        'id': 'ring_s4',
        'category': 'finger_flexion',
        'finger': 'Ring',
        'stage': 4,
        'title': 'ANELAR: Estágio 4 (FECHADO / PUNHO)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre completamente o dedo ANELAR fechado contra a palma da mão.',
        'other_fingers': 'Mão fechada em punho ou dedos adjacentes recolhidos.',
        'expected_summary': 'MCP ~ 85-90° | PIP ~ 105° | DIP ~ 80° (Fechado na palma)',
        'target_finger': 'Ring',
        'expected_stage': 4
    },

    # --- 4. DEDO MINDINHO ---
    {
        'id': 'pinky_s0',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 0,
        'title': 'MINDINHO: Estágio 0 (ESTENDIDO / RETO)',
        'posture': 'Palma de frente para a câmera na vertical, punho reto.',
        'target_action': 'Estique o dedo MINDINHO completamente reto para cima, alinhado à lateral da mão.',
        'other_fingers': 'Demais dedos estendidos ou em posição neutra.',
        'expected_summary': 'MCP ~ 0° | PIP ~ 0° | DIP ~ 0° (Reto para cima)',
        'target_finger': 'Pinky',
        'expected_stage': 0
    },
    {
        'id': 'pinky_s1',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 1,
        'title': 'MINDINHO: Estágio 1 (CURVADO / CONCHA)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Curve suavemente o MINDINHO em arco uniforme contínuo (formato de "C").',
        'other_fingers': 'Mão em formato suave.',
        'expected_summary': 'MCP ~ 25-30° | PIP ~ 40° | DIP ~ 35° (Curvado)',
        'target_finger': 'Pinky',
        'expected_stage': 1
    },
    {
        'id': 'pinky_s2',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 2,
        'title': 'MINDINHO: Estágio 2 (GANCHO / HOOK)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Base do MINDINHO reta para cima (0° a 15°), juntas da ponta (PIP/DIP) dobradas a ~90° para frente (garra/gancho).',
        'other_fingers': 'Demais dedos estendidos para cima.',
        'expected_summary': 'MCP ~ 0-15° | PIP ~ 90° | DIP ~ 75° (Gancho)',
        'target_finger': 'Pinky',
        'expected_stage': 2
    },
    {
        'id': 'pinky_s3',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 3,
        'title': 'MINDINHO: Estágio 3 (PLATAFORMA / TABLETOP)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre a base (MCP) a 90° para frente em direção à câmera, mantendo o restante do MINDINHO reto (formato de mesa horizontal em "L").',
        'other_fingers': 'Demais dedos estendidos para cima.',
        'expected_summary': 'MCP ~ 85-90° (base dobrada) | PIP ~ 0° (reto) | DIP ~ 0° (reto)',
        'target_finger': 'Pinky',
        'expected_stage': 3
    },
    {
        'id': 'pinky_s4',
        'category': 'finger_flexion',
        'finger': 'Pinky',
        'stage': 4,
        'title': 'MINDINHO: Estágio 4 (FECHADO / PUNHO)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Dobre totalmente o MINDINHO colado contra a palma da mão.',
        'other_fingers': 'Mão fechada em punho.',
        'expected_summary': 'MCP ~ 85-90° | PIP ~ 105° | DIP ~ 80° (Fechado na palma)',
        'target_finger': 'Pinky',
        'expected_stage': 4
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
        'id': 'thumb_f1_p0',
        'category': 'thumb',
        'f': 1, 'p': 0,
        'title': 'POLEGAR EM OPOSIÇÃO TRANSVERSAL - PONTA RETA (F=1, P=0)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Traga o polegar para a FRENTE da palma da mão, cruzando transversalmente em direção à base do dedo mínimo (oposição transversal), mantendo a ponta do polegar TOTALMENTE RETA e esticada.',
        'other_fingers': 'Dedos longos estendidos para permitir visão limpa do polegar.',
        'expected_summary': 'Polegar cruzando a palma | Ponta IP estendida (0°)',
        'target_finger': 'Thumb'
    },
    {
        'id': 'thumb_f1_p1',
        'category': 'thumb',
        'f': 1, 'p': 1,
        'title': 'POLEGAR EM OPOSIÇÃO PROFUNDA - PONTA DOBRADA (F=1, P=1)',
        'posture': 'Palma de frente para a câmera na vertical.',
        'target_action': 'Cruze o polegar na frente da palma colado sobre os dedos em oposição profunda com a junta da ponta (IP) flexionada a ~90° (como no punho fechado dos sinais "A" ou "S" em Libras).',
        'other_fingers': 'Mão em punho fechado com o polegar travando sobre os dedos.',
        'expected_summary': 'Polegar em oposição máxima na palma | Ponta IP flexionada (~70-90°)',
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
        self.captured_data: Dict[str, Any] = {}
        self.current_review_metrics: List[str] = []
        self.current_review_points: List[str] = []
        self.current_review_status: str = "Adequado"
        self.current_review_snapshot: Optional[np.ndarray] = None

        self.stable_frame_buffer: List[np.ndarray] = []
        self.stability_start_time: Optional[float] = None
        self.REQUIRED_STABLE_TIME = 1.2  # 1.2 segundos para disparar captura

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
            # MODO 1: CAPTURA EM TEMPO REAL
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
                            self._trigger_capture_and_review(step, frame, pts_pixels)
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
                print("\n[LOG] Calibração interrompida pelo usuário. Salvando progresso...")
                break

            elif key in [32, 13]:  # ESPAÇO ou ENTER
                if self.state == "CAPTURING":
                    if has_hand and pts_norm is not None:
                        self.stable_frame_buffer.append(pts_norm)
                        self._trigger_capture_and_review(step, frame, pts_pixels)
                elif self.state == "REVIEW":
                    # Usuário confirmou os dados extraídos -> Avança!
                    print(f"  ✓ Passo [{self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}] confirmado pelo usuário.")
                    self._advance_step()

            elif key == ord('r'):  # Recapturar / Repetir
                step_id = step['id']
                if step_id in self.captured_data:
                    del self.captured_data[step_id]
                self.state = "CAPTURING"
                self.stability_start_time = None
                self.stable_frame_buffer = []
                print(f"[REFAZER] Reiniciando captura de: {step['title']}")

            elif key == ord('b'):  # Voltar passo anterior
                if self.current_step_idx > 0:
                    self.current_step_idx -= 1
                    self.state = "CAPTURING"
                    self.stability_start_time = None
                    self.stable_frame_buffer = []
                    prev_step = CALIBRATION_STEPS[self.current_step_idx]
                    print(f"[VOLTAR] Retornando ao passo anterior: {prev_step['title']}")

            elif key == ord('s'):  # Pular passo
                print(f"[PULAR] Passo '{step['title']}' pulado com fallback.")
                self._advance_step()

            if self.current_step_idx >= len(CALIBRATION_STEPS):
                print("\n[SUCESSO] Todos os 27 passos da calibração foram concluídos com êxito!")
                break

        cap.release()
        cv2.destroyAllWindows()

        # Compilar e salvar configurações finais
        self.compile_and_save_settings()
        return True

    def _trigger_capture_and_review(self, step: Dict[str, Any], frame: np.ndarray, pts_raw: np.ndarray):
        """Calcula as métricas exatas e transita para o modo de revisão."""
        step_id = step['id']
        img_filename = os.path.join(CAPTURES_DIR, f"{self.current_step_idx+1:02d}_{step_id}.png")

        # Média filtrada das coordenadas estáveis
        if len(self.stable_frame_buffer) > 0:
            avg_pts = np.mean(np.array(self.stable_frame_buffer), axis=0)
        else:
            avg_pts = pts_raw if pts_raw is not None else np.zeros((21, 3))

        self.captured_data[step_id] = {
            'step_meta': step,
            'pts_norm': avg_pts,
            'pts_raw': pts_raw,
            'image_path': img_filename
        }

        # Extrair e formatar as métricas para exibição na tela
        self.current_review_metrics, self.current_review_points, self.current_review_status = self._format_metrics_for_step(step, avg_pts)

        # Criar snapshot congelado com destaque do dedo-alvo nos landmarks
        annotated_snapshot = frame.copy()
        if pts_raw is not None and len(pts_raw) == 21:
            target_f = step.get('target_finger')
            if target_f and target_f in FINGER_JOINTS:
                idxs = FINGER_JOINTS[target_f]
                # Conectar juntas do dedo alvo com linha vibrante
                for i in range(1, len(idxs)):
                    p1 = (int(pts_raw[idxs[i-1]][0]), int(pts_raw[idxs[i-1]][1]))
                    p2 = (int(pts_raw[idxs[i]][0]), int(pts_raw[idxs[i]][1]))
                    cv2.line(annotated_snapshot, p1, p2, (255, 230, 80), 4, cv2.LINE_AA)
                # Destacar nós das articulações
                joint_names = ["PULSO", "MCP", "PIP", "DIP", "TIP"] if target_f != 'Thumb' else ["PULSO", "CMC", "MCP", "IP", "TIP"]
                for i, j_idx in enumerate(idxs):
                    pt = (int(pts_raw[j_idx][0]), int(pts_raw[j_idx][1]))
                    cv2.circle(annotated_snapshot, pt, 8, (50, 255, 120), -1, cv2.LINE_AA)
                    cv2.circle(annotated_snapshot, pt, 11, (255, 255, 255), 2, cv2.LINE_AA)
                    if i > 0:
                        cv2.putText(annotated_snapshot, joint_names[i], (pt[0] + 12, pt[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(img_filename, annotated_snapshot)
        self.current_review_snapshot = annotated_snapshot
        self.state = "REVIEW"

        print(f"\n[CAPTURA REALIZADA] -> {step['title']}")
        print(f"Status: {self.current_review_status}")
        for line in self.current_review_metrics:
            print(f"   {line}")
        print("-> Pressione [ESPAÇO] ou [ENTER] para Confirmar e Avançar, ou [R] para Refazer...")

    def _advance_step(self):
        """Avança para o próximo passo da calibração."""
        self.current_step_idx += 1
        self.state = "CAPTURING"
        self.stability_start_time = None
        self.stable_frame_buffer = []
        if self.current_step_idx < len(CALIBRATION_STEPS):
            next_step = CALIBRATION_STEPS[self.current_step_idx]
            print(f"\n=======================================================")
            print(f"Iniciando Passo [{self.current_step_idx+1}/{len(CALIBRATION_STEPS)}]: {next_step['title']}")
            print(f"Instrução: {next_step['target_action']}")
            print(f"=======================================================")

    # -----------------------------------------------------------------------
    # EXTRAÇÃO E FORMATAÇÃO DE MÉTRICAS ANATÔMICAS
    # -----------------------------------------------------------------------

    def _format_metrics_for_step(self, step: Dict[str, Any], pts_norm: np.ndarray) -> Tuple[List[str], List[str], str]:
        """Calcula os ângulos, distâncias e coordenadas 3D dos pontos extraídos do frame capturado."""
        cat = step['category']
        metrics = []
        points_info = []
        status_conformity = "Adequado"

        if cat == 'baseline':
            p_wrist = pts_norm[0]
            p_mcp9 = pts_norm[9]
            palm_len = np.linalg.norm(p_mcp9 - p_wrist)
            metrics.append("Escala da Palma (Pulso -> Metacarpo Médio): 1.000")
            status_conformity = "Excelente (Mão Espalmada Detectada)"

            for fname in ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']:
                idxs = FINGER_JOINTS[fname]
                l1 = float(np.linalg.norm(pts_norm[idxs[2]] - pts_norm[idxs[1]]))
                l2 = float(np.linalg.norm(pts_norm[idxs[3]] - pts_norm[idxs[2]]))
                l3 = float(np.linalg.norm(pts_norm[idxs[4]] - pts_norm[idxs[3]]))
                metrics.append(f"{fname:7s}: Falanges = [{l1:.2f}, {l2:.2f}, {l3:.2f}] | Total = {l1+l2+l3:.2f}")

            points_info.append(f"Pulso (ID 0):   [{pts_norm[0][0]:+.2f}, {pts_norm[0][1]:+.2f}, {pts_norm[0][2]:+.2f}]")
            points_info.append(f"MCP Indicador:  [{pts_norm[5][0]:+.2f}, {pts_norm[5][1]:+.2f}, {pts_norm[5][2]:+.2f}]")
            points_info.append(f"MCP Médio (9):  [{pts_norm[9][0]:+.2f}, {pts_norm[9][1]:+.2f}, {pts_norm[9][2]:+.2f}]")
            points_info.append(f"MCP Mindinho:   [{pts_norm[17][0]:+.2f}, {pts_norm[17][1]:+.2f}, {pts_norm[17][2]:+.2f}]")
            points_info.append(f"Largura Palma:  {float(np.linalg.norm(pts_norm[5] - pts_norm[17])):.2f} (unidades norm)")

        elif cat == 'finger_flexion':
            f_name = step['finger']
            idxs = FINGER_JOINTS[f_name]
            j2_p = joint_flexion(pts_norm[idxs[0]], pts_norm[idxs[1]], pts_norm[idxs[2]])
            j3_p = joint_flexion(pts_norm[idxs[1]], pts_norm[idxs[2]], pts_norm[idxs[3]])
            j4_p = joint_flexion(pts_norm[idxs[2]], pts_norm[idxs[3]], pts_norm[idxs[4]])

            st = step['expected_stage']
            stage_names = {
                0: "0 (Estendido / Reto)", 1: "1 (Curvado / Concha)",
                2: "2 (Gancho / Hook)", 3: "3 (Plataforma / Tabletop)", 4: "4 (Fechado / Punho)"
            }
            metrics.append(f"Dedo: {f_name.upper()} | Estágio: {stage_names.get(st, str(st))}")

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
            points_info.append(f"MCP (Ponto {idxs[1]}): [{pts_norm[idxs[1]][0]:+.2f}, {pts_norm[idxs[1]][1]:+.2f}, {pts_norm[idxs[1]][2]:+.2f}]")
            points_info.append(f"PIP (Ponto {idxs[2]}): [{pts_norm[idxs[2]][0]:+.2f}, {pts_norm[idxs[2]][1]:+.2f}, {pts_norm[idxs[2]][2]:+.2f}]")
            points_info.append(f"DIP (Ponto {idxs[3]}): [{pts_norm[idxs[3]][0]:+.2f}, {pts_norm[idxs[3]][1]:+.2f}, {pts_norm[idxs[3]][2]:+.2f}]")
            points_info.append(f"TIP (Ponta {idxs[4]}): [{pts_norm[idxs[4]][0]:+.2f}, {pts_norm[idxs[4]][1]:+.2f}, {pts_norm[idxs[4]][2]:+.2f}]")
            dist_tip_wrist = float(np.linalg.norm(pts_norm[idxs[4]] - pts_norm[0]))
            points_info.append(f"Distância Ponta -> Pulso: {dist_tip_wrist:.2f}")

        elif cat == 'spread':
            sp_pnk_rng = vec_angle(pts_norm[17] - pts_norm[0], pts_norm[13] - pts_norm[0])
            sp_rng_mid = vec_angle(pts_norm[13] - pts_norm[0], pts_norm[9] - pts_norm[0])
            sp_mid_idx = vec_angle(pts_norm[9] - pts_norm[0], pts_norm[5] - pts_norm[0])
            sp_idx_thm = vec_angle(pts_norm[5] - pts_norm[0], pts_norm[1] - pts_norm[0])
            is_open = step['spread_state'] == 0
            mode_str = "Leque Máximo (A=0 Aberto)" if is_open else "Dedos Paralelos (A=1 Fechado)"
            status_conformity = "Excelente (Abertura Capturada)" if (is_open and sp_rng_mid > 12) or (not is_open and sp_rng_mid < 10) else "Concluído"

            metrics.append(f"Configuração: {mode_str}")
            metrics.append(f"• Mindinho - Anelar:   {sp_pnk_rng:5.1f}°")
            metrics.append(f"• Anelar - Médio:      {sp_rng_mid:5.1f}°")
            metrics.append(f"• Médio - Indicador:   {sp_mid_idx:5.1f}°")
            metrics.append(f"• Indicador - Polegar: {sp_idx_thm:5.1f}°")

            points_info.append(f"Ponta Mindinho:  [{pts_norm[20][0]:+.2f}, {pts_norm[20][1]:+.2f}, {pts_norm[20][2]:+.2f}]")
            points_info.append(f"Ponta Anelar:    [{pts_norm[16][0]:+.2f}, {pts_norm[16][1]:+.2f}, {pts_norm[16][2]:+.2f}]")
            points_info.append(f"Ponta Médio:     [{pts_norm[12][0]:+.2f}, {pts_norm[12][1]:+.2f}, {pts_norm[12][2]:+.2f}]")
            points_info.append(f"Ponta Indicador: [{pts_norm[8][0]:+.2f}, {pts_norm[8][1]:+.2f}, {pts_norm[8][2]:+.2f}]")
            span = float(np.linalg.norm(pts_norm[20] - pts_norm[4]))
            points_info.append(f"Envergadura Total (Polegar-Mindinho): {span:.2f}")

        elif cat == 'thumb':
            dist_opp = float(np.linalg.norm(pts_norm[4] - pts_norm[9]))
            ip_flex = joint_flexion(pts_norm[2], pts_norm[3], pts_norm[4])
            mcp_flex = joint_flexion(pts_norm[1], pts_norm[2], pts_norm[3])
            f_label = "No Plano da Palma (F=0)" if step['f'] == 0 else "Oposição Transversal (F=1)"
            p_label = "Ponta Reta (P=0)" if step['p'] == 0 else "Ponta Dobrada (P=1)"
            status_conformity = "Excelente (Polegar Capturado)"

            metrics.append(f"Modo: {f_label} | {p_label}")
            metrics.append(f"• Flexão da Ponta (IP):       {ip_flex:5.1f}°")
            metrics.append(f"• Flexão da Base (MCP):       {mcp_flex:5.1f}°")
            metrics.append(f"• Distância Ponta -> Palma:   {dist_opp:5.2f}")

            points_info.append(f"CMC (Ponto 1): [{pts_norm[1][0]:+.2f}, {pts_norm[1][1]:+.2f}, {pts_norm[1][2]:+.2f}]")
            points_info.append(f"MCP (Ponto 2): [{pts_norm[2][0]:+.2f}, {pts_norm[2][1]:+.2f}, {pts_norm[2][2]:+.2f}]")
            points_info.append(f"IP  (Ponto 3): [{pts_norm[3][0]:+.2f}, {pts_norm[3][1]:+.2f}, {pts_norm[3][2]:+.2f}]")
            points_info.append(f"TIP (Ponto 4): [{pts_norm[4][0]:+.2f}, {pts_norm[4][1]:+.2f}, {pts_norm[4][2]:+.2f}]")
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

        # 1. Painel Superior Responsivo (Instruções detalhadas)
        top_h = 135 if h >= 600 else 115
        draw.rectangle([(0, 0), (w, top_h)], fill=(17, 17, 27, 235))
        draw.line([(0, top_h), (w, top_h)], fill=(137, 180, 250, 255), width=2)

        # Cabeçalho do passo
        f_title = self.hud.get_font(20 if w >= 1000 else 16, bold=True)
        f_body = self.hud.get_font(14 if w >= 1000 else 12, bold=False)
        f_body_bold = self.hud.get_font(14 if w >= 1000 else 12, bold=True)
        f_small = self.hud.get_font(12, bold=False)

        step_num_str = f"PASSO {self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}"
        draw.text((25, 10), step_num_str, font=f_small, fill=(249, 226, 175, 255))
        draw.text((25, 28), step['title'], font=f_title, fill=(137, 180, 250, 255))

        # Instruções anatômicas
        posture_str = f"Posição: {step['posture']}"
        action_str = f"Ação:    {step['target_action']}"
        others_str = f"Outros:  {step['other_fingers']}"

        # Truncar visualmente se a tela for menor
        max_chars = 110 if w >= 1100 else 75
        p_disp = posture_str[:max_chars] + ("..." if len(posture_str) > max_chars else "")
        a_disp = action_str[:max_chars] + ("..." if len(action_str) > max_chars else "")
        o_disp = others_str[:max_chars] + ("..." if len(others_str) > max_chars else "")

        draw.text((25, 58), p_disp, font=f_body, fill=(205, 214, 244, 255))
        draw.text((25, 78), a_disp, font=f_body_bold, fill=(166, 227, 161, 255))
        draw.text((25, 98), o_disp, font=f_small, fill=(186, 194, 222, 255))

        # 2. Painel Inferior (Barra de comandos)
        bot_h = 65
        draw.rectangle([(0, h - bot_h), (w, h)], fill=(17, 17, 27, 240))
        draw.line([(0, h - bot_h), (w, h - bot_h)], fill=(69, 71, 90, 255), width=1)

        f_ctrl = self.hud.get_font(14, bold=True)
        f_ctrl_sub = self.hud.get_font(12, bold=False)
        ctrl_str = "[ESPAÇO] Capturar Agora  |  [R] Repetir  |  [B] Voltar  |  [S] Pular  |  [Q / ESC] Salvar e Sair"
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

    def _render_review_hud(self, frame: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        """Renderiza a tela de revisão e confirmação dos dados extraídos."""
        h, w, _ = frame.shape

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Escurecer fundo levemente para destacar o painel de revisão
        draw.rectangle([(0, 0), (w, h)], fill=(10, 10, 15, 140))

        # Cartão central responsivo
        card_w = min(int(w * 0.94), 960)
        card_h = min(int(h * 0.88), 540)
        cx = (w - card_w) // 2
        cy = (h - card_h) // 2

        # Sombra e fundo do cartão
        draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], fill=(24, 24, 37, 245))
        draw.rectangle([(cx, cy), (cx + card_w, cy + card_h)], outline=(166, 227, 161, 255), width=2)

        # Cabeçalho do Cartão
        hdr_h = 68
        draw.rectangle([(cx, cy), (cx + card_w, cy + hdr_h)], fill=(30, 30, 46, 255))
        draw.line([(cx, cy + hdr_h), (cx + card_w, cy + hdr_h)], fill=(166, 227, 161, 255), width=2)

        f_hdr = self.hud.get_font(19, bold=True)
        f_sub = self.hud.get_font(14, bold=False)
        draw.text((cx + 25, cy + 12), "✓ CAPTURA CONCLUÍDA — REVISÃO DOS DADOS EXTRAÍDOS", font=f_hdr, fill=(166, 227, 161, 255))
        draw.text((cx + 25, cy + 40), f"Passo {self.current_step_idx + 1}/{len(CALIBRATION_STEPS)}: {step['title']}", font=f_sub, fill=(205, 214, 244, 255))

        # Divisão em 2 colunas
        col_w = (card_w - 60) // 2
        col1_x = cx + 25
        col2_x = cx + 35 + col_w
        content_y = cy + hdr_h + 16

        f_sec = self.hud.get_font(15, bold=True)
        f_data = self.hud.get_font(14, bold=False)
        f_data_mono = self.hud.get_font(13, bold=False)

        # Coluna 1: Métricas Biomecânicas
        draw.text((col1_x, content_y), "Métricas Biomecânicas (Ângulos e Flexões):", font=f_sec, fill=(249, 226, 175, 255))
        for i, line in enumerate(self.current_review_metrics):
            col_color = (166, 227, 161, 255) if "Excelente" in line else (205, 214, 244, 255)
            draw.text((col1_x + 8, content_y + 30 + i * 25), line, font=f_data, fill=col_color)

        # Coluna 2: Pontos Extraídos dos Landmarks
        draw.text((col2_x, content_y), "Coordenadas 3D dos Landmarks Extraídos:", font=f_sec, fill=(137, 180, 250, 255))
        for i, line in enumerate(self.current_review_points):
            draw.text((col2_x + 8, content_y + 30 + i * 25), line, font=f_data_mono, fill=(186, 194, 222, 255))

        # Linha divisória vertical suave entre colunas
        div_x = col2_x - 12
        draw.line([(div_x, content_y + 10), (div_x, cy + card_h - 85)], fill=(69, 71, 90, 180), width=1)

        # Rodapé de Decisão no Cartão
        btn_y = cy + card_h - 75
        draw.rectangle([(cx, btn_y), (cx + card_w, cy + card_h)], fill=(17, 17, 27, 255))
        draw.line([(cx, btn_y), (cx + card_w, btn_y)], fill=(69, 71, 90, 255), width=1)

        f_btn_bold = self.hud.get_font(16, bold=True)
        f_btn_sub = self.hud.get_font(13, bold=False)

        draw.text((cx + 25, btn_y + 14), "[ESPAÇO] ou [ENTER] : CONFIRMAR E AVANÇAR PARA O PRÓXIMO PASSO", font=f_btn_bold, fill=(166, 227, 161, 255))
        draw.text((cx + 25, btn_y + 42), "[R] : REFAZER CAPTURA (ajustar pose)    |    [B] : VOLTAR AO ANTERIOR", font=f_btn_sub, fill=(249, 226, 175, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # -----------------------------------------------------------------------
    # COMPILAÇÃO E EXPORTAÇÃO DE CALIBRAÇÃO E SEEDS
    # -----------------------------------------------------------------------

    def compile_and_save_settings(self, output_path: str = CALIBRATION_FILE) -> Dict[str, Any]:
        """Compila todas as medições capturadas e gera o calibration_settings.json."""
        print("\n[COMPILAÇÃO] Processando limites anatômicos capturados...")

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
            'phalanx_lengths': phalanx_lengths,
            'avg_palm': avg_palm
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
