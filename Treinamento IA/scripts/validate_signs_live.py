"""
Script de Validação Interativa de Sinais em LIBRAS (IA Biomecânica)
===================================================================
Executa a validação em tempo real utilizando a webcam do computador, o modelo TFLite
treinado com as 2.568 classes cinemáticas e o motor diferencial de biomecânica.

Permite selecionar a classe esperada (por letra ou código numérico de 10 dígitos),
detecta a classe executada pelo usuário e gera feedbacks anatômicos imediatos
mostrando exatamente o que deve ser corrigido (ex: afastar dedos, esticar indicador, etc).

Atalhos do Teclado na Janela:
  [A - Z] : Troca imediatamente o sinal alvo para a letra correspondente
  [TAB]   : Avança para o próximo sinal canônico da lista
  [SHIFT+TAB] / [SETA ESQ] : Volta para o sinal anterior
  [C]     : Permite digitar um código de 10 dígitos customizado no terminal
  [ESPAÇO]: Pausa / Congela a análise do frame
  [Q / ESC]: Encerra o script
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------------
# CONFIGURAÇÃO DE DIRETÓRIOS E MODELOS
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_PATH = os.path.join(BASE_DIR, "models", "modelo_gestos.tflite")
H5_PATH = os.path.join(BASE_DIR, "models", "modelo_gestos.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")
SEEDS_PATH = os.path.join(BASE_DIR, "data", "seeds", "seeds.json")

# -------------------------------------------------------------
# DICIONÁRIO CANÔNICO DE LETRAS DE LIBRAS
# -------------------------------------------------------------
LETTER_KINEMATICS = {
    'A': {'code': '4141414110', 'name': "Sinal 'A'", 'desc': "Punho fechado com polegar apoiado na lateral"},
    'B': {'code': '0101010110', 'name': "Sinal 'B'", 'desc': "4 dedos erguidos juntos, polegar dobrado na frente"},
    'C': {'code': '1010101000', 'name': "Sinal 'C'", 'desc': "Dedos curvados em formato de arco/concha"},
    'D': {'code': '4141410110', 'name': "Sinal 'D'", 'desc': "Indicador erguido, outros 3 dedos tocando o polegar"},
    'E': {'code': '2121212110', 'name': "Sinal 'E'", 'desc': "Dedos em garra recolhida com pontas sobre o polegar"},
    'I': {'code': '0141414110', 'name': "Sinal 'I'", 'desc': "Apenas o mindinho erguido, outros dedos fechados"},
    'L': {'code': '4141410000', 'name': "Sinal 'L'", 'desc': "Indicador reto e polegar aberto em 90°"},
    'M': {'code': '3131314110', 'name': "Sinal 'M'", 'desc': "Indicador, médio e anelar dobrados sobre o polegar"},
    'N': {'code': '4131314110', 'name': "Sinal 'N'", 'desc': "Indicador e médio dobrados sobre o polegar"},
    'O': {'code': '1010101010', 'name': "Sinal 'O'", 'desc': "Dedos curvados formando um círculo com o polegar"},
    'R': {'code': '4141010110', 'name': "Sinal 'R'", 'desc': "Indicador e médio cruzados, outros dedos fechados"},
    'S': {'code': '4141414110', 'name': "Sinal 'S'", 'desc': "Punho cerrado com polegar cruzando a frente dos dedos"},
    'U': {'code': '4141010110', 'name': "Sinal 'U'", 'desc': "Indicador e médio estendidos retos e bem juntos"},
    'V': {'code': '4141000110', 'name': "Sinal 'V'", 'desc': "Indicador e médio estendidos e abertos em 'V'"},
    'W': {'code': '1000000110', 'name': "Sinal 'W'", 'desc': "Indicador, médio e anelar estendidos em 'W', mindinho apoiado"},
    'X': {'code': '4141412110', 'name': "Sinal 'X'", 'desc': "Indicador em gancho (dobrado), outros fechados"},
    'Y': {'code': '0041414100', 'name': "Sinal 'Y'", 'desc': "Polegar e mindinho estendidos, dedos do meio fechados"}
}

STAGE_NAMES = {
    0: 'Estendido',
    1: 'Curvado (Concha)',
    2: 'Gancho (Hook)',
    3: 'Plataforma',
    4: 'Fechado'
}

CANONICAL_KEYS = list(LETTER_KINEMATICS.keys())

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# -------------------------------------------------------------
# FUNÇÕES DE RESOLUÇÃO CINEMÁTICA & FEEDBACK
# -------------------------------------------------------------
def resolve_kinematic_code(input_val: str) -> str:
    if not input_val:
        return '4141000110'
    clean = str(input_val).strip().upper()
    if clean in LETTER_KINEMATICS:
        return LETTER_KINEMATICS[clean]['code']
    if len(clean) == 10 and clean.isdigit():
        return clean
    return '4141000110'

def get_closest_letter(code: str):
    best_letter = None
    best_dist = float('inf')
    for letter, data in LETTER_KINEMATICS.items():
        dist = sum(abs(int(code[i]) - int(data['code'][i])) for i in range(min(len(code), 10)))
        if dist < best_dist:
            best_dist = dist
            best_letter = letter
    return {
        'letter': best_letter,
        'distance': best_dist,
        'is_exact': best_dist == 0,
        'info': LETTER_KINEMATICS[best_letter]
    }

def parse_hand_pose(code: str):
    code = (code + "0000000000")[:10]
    d4 = int(code[0])  # Mindinho
    a3 = int(code[1])  # Spread Min-Ane
    d3 = int(code[2])  # Anelar
    a2 = int(code[3])  # Spread Ane-Med
    d2 = int(code[4])  # Médio
    a1 = int(code[5])  # Spread Med-Ind
    d1 = int(code[6])  # Indicador
    a0 = int(code[7])  # Spread Ind-Pol
    f  = int(code[8])  # Oposição Polegar
    p  = int(code[9])  # Ponta Polegar

    return {
        'raw_code': code,
        'pinky':  {'stage': d4, 'is_extended': d4 == 0, 'is_closed': d4 >= 3},
        'ring':   {'stage': d3, 'is_extended': d3 == 0, 'is_closed': d3 >= 3},
        'middle': {'stage': d2, 'is_extended': d2 == 0, 'is_closed': d2 >= 3},
        'index':  {'stage': d1, 'is_extended': d1 == 0, 'is_closed': d1 >= 3},
        'thumb': {
            'is_opposed': f == 1,
            'is_spread': a0 == 0,
            'is_tip_folded': p == 1
        },
        'spreads': {
            'middle_index': 'Aberto (V)' if a1 == 0 else 'Junto (U)',
            'is_v_open': a1 == 0
        }
    }

def get_biomechanical_guidance(detected_code: str, expected_code: str):
    if not detected_code or not expected_code:
        return {'match': False, 'hints': ['Aguardando mão na câmera...'], 'finger_status': {}}

    detected = parse_hand_pose(detected_code)
    expected = parse_hand_pose(expected_code)

    hints = []
    finger_status = {
        'index': 'OK', 'middle': 'OK', 'ring': 'OK',
        'pinky': 'OK', 'thumb': 'OK', 'spread': 'OK'
    }

    # 1. INDICADOR
    if expected['index']['stage'] != detected['index']['stage']:
        if expected['index']['is_extended'] and not detected['index']['is_extended']:
            hints.append("Estique o dedo INDICADOR totalmente para cima!")
            finger_status['index'] = 'ERR'
        elif expected['index']['is_closed'] and not detected['index']['is_closed']:
            hints.append("Dobre o dedo INDICADOR para baixo (fechado na palma)!")
            finger_status['index'] = 'ERR'
        elif expected['index']['stage'] == 1:
            hints.append("Curve o INDICADOR suavemente em formato de arco (C/O)!")
            finger_status['index'] = 'ERR'
        elif expected['index']['stage'] == 2:
            hints.append("Dobre o INDICADOR em forma de gancho/anzol (sinal X)!")
            finger_status['index'] = 'ERR'

    # 2. MÉDIO
    if expected['middle']['stage'] != detected['middle']['stage']:
        if expected['middle']['is_extended'] and not detected['middle']['is_extended']:
            hints.append("Estique o dedo MÉDIO totalmente para cima!")
            finger_status['middle'] = 'ERR'
        elif expected['middle']['is_closed'] and not detected['middle']['is_closed']:
            hints.append("Dobre o dedo MÉDIO para a palma!")
            finger_status['middle'] = 'ERR'
        elif expected['middle']['stage'] == 1:
            hints.append("Curve o dedo MÉDIO em arco!")
            finger_status['middle'] = 'ERR'

    # 3. ANELAR
    if expected['ring']['stage'] != detected['ring']['stage']:
        if expected['ring']['is_extended'] and not detected['ring']['is_extended']:
            hints.append("Estique o dedo ANELAR para cima!")
            finger_status['ring'] = 'ERR'
        elif expected['ring']['is_closed'] and not detected['ring']['is_closed']:
            hints.append("Dobre o dedo ANELAR para baixo!")
            finger_status['ring'] = 'ERR'

    # 4. MINDINHO
    # Tolerância anatômica: Se o Anelar está estendido (como em 'W'),
    # o mindinho não consegue fechar a 100% (stage 4); estágios 1 e 2 são aceitos como válidos.
    is_w_posture = expected['ring']['is_extended'] and expected['middle']['is_extended'] and expected['index']['is_extended']
    if is_w_posture and detected['pinky']['stage'] in (1, 2) and expected['pinky']['stage'] in (1, 2):
        pass  # Tolerância anatômica confirmada (juncturae tendinum)
    elif expected['pinky']['stage'] != detected['pinky']['stage']:
        if expected['pinky']['is_extended'] and not detected['pinky']['is_extended']:
            hints.append("Estique o dedo MINDINHO para cima (sinal I ou Y)!")
            finger_status['pinky'] = 'ERR'
        elif expected['pinky']['is_closed'] and not detected['pinky']['is_closed']:
            hints.append("Dobre o dedo MINDINHO para a palma!")
            finger_status['pinky'] = 'ERR'

    # 5. ABERTURA ENTRE INDICADOR E MÉDIO (V vs U)
    if expected['index']['is_extended'] and expected['middle']['is_extended']:
        exp_v = expected['spreads']['is_v_open']
        det_v = detected['spreads']['is_v_open']
        if exp_v and not det_v:
            hints.append("AFASTE o Indicador do Médio! No sinal 'V' os dedos ficam abertos em V.")
            finger_status['spread'] = 'ERR'
        elif not exp_v and det_v:
            hints.append("JUNTE o Indicador e o Médio! No sinal 'U' os dedos ficam retos e colados.")
            finger_status['spread'] = 'ERR'

    # 6. POLEGAR
    exp_th_spread = expected['thumb']['is_spread']
    det_th_spread = detected['thumb']['is_spread']
    exp_th_opp = expected['thumb']['is_opposed']
    det_th_opp = detected['thumb']['is_opposed']

    if exp_th_spread and not det_th_spread:
        hints.append("Abra o POLEGAR para fora em 90° (sinal L ou Y)!")
        finger_status['thumb'] = 'ERR'
    elif not exp_th_spread and det_th_spread:
        hints.append("Recolha o POLEGAR apoiado contra a lateral dos dedos!")
        finger_status['thumb'] = 'ERR'
    elif exp_th_opp and not det_th_opp:
        hints.append("Posicione o POLEGAR cruzando a frente da palma (sinais B / E)!")
        finger_status['thumb'] = 'ERR'

    # Verifica se os dedos principais conferem
    exact_match = (detected_code == expected_code)
    w_match = (
        is_w_posture and 
        detected['index']['is_extended'] and 
        detected['middle']['is_extended'] and 
        detected['ring']['is_extended'] and
        detected['pinky']['stage'] in (1, 2) and
        detected['spreads']['is_v_open']
    )
    is_match = exact_match or w_match

    if is_match:
        hints = ["PERFEITO! A configuração dos dedos confere exatamente com o sinal esperado!"]

    return {
        'match': is_match,
        'hints': hints,
        'finger_status': finger_status
    }

# -------------------------------------------------------------
# CLASSE PRINCIPAL DO VALIDADOR
# -------------------------------------------------------------
class LibrasLiveValidator:
    def __init__(self, initial_class="4141000110"):
        print("\n" + "="*65)
        print("   VALIDADOR INTERATIVO DE SINAIS LIBRAS - TCC")
        print("="*65)

        self.expected_code = resolve_kinematic_code(initial_class)
        self.labels = []
        self.interpreter = None
        self.seeds = {}
        self.paused = False

        # 1. Carrega Labels
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]
            print(f"[*] Labels carregadas: {len(self.labels):,} classes cadastradas.")
        else:
            print(f"[!] Erro: Arquivo de labels não encontrado em {LABELS_PATH}")

        # 2. Carrega Modelo (TFLite ultrarrápido)
        if os.path.exists(TFLITE_PATH):
            self.interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[*] Modelo TFLite carregado com sucesso ({os.path.basename(TFLITE_PATH)}).")
        else:
            print(f"[!] Erro: Arquivo TFLite não encontrado em {TFLITE_PATH}")

        # 3. Carrega Seeds de Referência
        if os.path.exists(SEEDS_PATH):
            try:
                with open(SEEDS_PATH, "r", encoding="utf-8") as f:
                    self.seeds = json.load(f)
                print(f"[*] Seeds 3D de referência carregadas: {len(self.seeds):,} modelos.")
            except Exception as e:
                print(f"[!] Aviso ao carregar seeds: {e}")

        # 4. MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55
        )

        # 5. Fonte PIL para suporte a caracteres acentuados
        try:
            self.font_title = ImageFont.truetype("arialbd.ttf", 20)
            self.font_bold = ImageFont.truetype("arialbd.ttf", 15)
            self.font_normal = ImageFont.truetype("arial.ttf", 14)
            self.font_small = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            self.font_title = ImageFont.load_default()
            self.font_bold = ImageFont.load_default()
            self.font_normal = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def set_target_letter(self, letter: str):
        letter = letter.upper()
        if letter in LETTER_KINEMATICS:
            self.expected_code = LETTER_KINEMATICS[letter]['code']
            info = LETTER_KINEMATICS[letter]
            print(f"\n🎯 [ALVO ALTERADO] Letra '{letter}' -> Código: {self.expected_code}")
            print(f"   Descrição: {info['desc']}")

    def set_target_code(self, code: str):
        if len(code) == 10 and code.isdigit():
            self.expected_code = code
            closest = get_closest_letter(code)
            letter_info = f" (Mais próximo de '{closest['letter']}')" if closest['letter'] else ""
            print(f"\n🎯 [ALVO ALTERADO] Código: {code}{letter_info}")
        else:
            print(f"[!] Código inválido '{code}'. Deve conter exatamente 10 dígitos numéricos.")

    def cycle_canonical_sign(self, direction: int = 1):
        closest = get_closest_letter(self.expected_code)
        cur_letter = closest['letter'] if closest and closest['letter'] else 'A'
        try:
            cur_idx = CANONICAL_KEYS.index(cur_letter)
        except ValueError:
            cur_idx = 0
        new_idx = (cur_idx + direction) % len(CANONICAL_KEYS)
        new_letter = CANONICAL_KEYS[new_idx]
        self.set_target_letter(new_letter)

    def extract_features(self, landmarks):
        """
        Extrai e normaliza as 42 features da mão identicamente ao pipeline de treino:
        Normaliza pelo tamanho da bounding box e subtrai o pulso (ponto 0).
        """
        pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        size = max(max_x - min_x, max_y - min_y, 1e-6)

        nx = (pts[:, 0] - min_x) / size
        ny = (pts[:, 1] - min_y) / size
        w_x, w_y = nx[0], ny[0]

        features = np.empty((1, 42), dtype=np.float32)
        features[0, 0::2] = nx - w_x
        features[0, 1::2] = ny - w_y
        return features

    def predict(self, features):
        if self.interpreter is None:
            return "0000000000", 0.0
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        top_idx = int(np.argmax(output_data))
        conf = float(output_data[top_idx])
        label = self.labels[top_idx] if top_idx < len(self.labels) else "0000000000"
        return label, conf

    def draw_skeleton(self, frame, landmarks, w, h):
        """Desenha as conexões e articulações da mão em verde neon e magenta"""
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, coords[start], coords[end], (0, 255, 128), 3, cv2.LINE_AA)

        tip_indices = {4, 8, 12, 16, 20}
        for i, pt in enumerate(coords):
            if i in tip_indices:
                cv2.circle(frame, pt, 6, (255, 0, 128), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_target_pip(self, frame, target_code: str):
        """Desenha um card PiP com o esqueleto do modelo alvo de referência"""
        if target_code not in self.seeds:
            return

        h, w, _ = frame.shape
        pip_w, pip_h = 160, 170
        pip_x, pip_y = w - pip_w - 20, 20

        # Fundo do card PiP com transparência
        overlay = frame.copy()
        cv2.rectangle(overlay, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (20, 26, 30), -1)
        cv2.rectangle(overlay, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (255, 229, 0), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Projeção 2D dos pontos da seed
        pts_raw = self.seeds[target_code]
        x2d = np.array([p['x'] for p in pts_raw], dtype=np.float32)
        y2d = np.array([p['y'] for p in pts_raw], dtype=np.float32)

        min_x, max_x = np.min(x2d), np.max(x2d)
        min_y, max_y = np.min(y2d), np.max(y2d)
        span_x = max(max_x - min_x, 1e-5)
        span_y = max(max_y - min_y, 1e-5)
        span = max(span_x, span_y)

        skel_w = pip_w - 30
        skel_h = pip_h - 50
        skel_ox = pip_x + 15
        skel_oy = pip_y + 40

        skel_pts = []
        for i in range(21):
            px = int(skel_ox + ((x2d[i] - min_x) / span) * skel_w)
            py = int(skel_oy + ((y2d[i] - min_y) / span) * skel_h)
            skel_pts.append((px, py))

        for s, e in HAND_CONNECTIONS:
            cv2.line(frame, skel_pts[s], skel_pts[e], (255, 229, 0), 2, cv2.LINE_AA)

        for pt in skel_pts:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_hud(self, frame, detected_code: str, confidence: float, guidance: dict, fps: float):
        """Renderiza um HUD limpo, moderno e completo com textos em português"""
        h, w, _ = frame.shape
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Dados da classe esperada
        exp_closest = get_closest_letter(self.expected_code)
        exp_letter_name = f"Letra '{exp_closest['letter']}'" if exp_closest['letter'] else "Personalizado"
        exp_desc = exp_closest['info']['desc'] if exp_closest['info'] else "Classe livre"

        # Dados da classe recebida
        det_closest = get_closest_letter(detected_code) if detected_code else None
        det_letter_name = f"Letra '{det_closest['letter']}'" if (det_closest and det_closest['letter']) else "Indefinido"

        # -------------------------------------------------------------
        # 1. CABEÇALHO SUPERIOR (Cards de Comparação)
        # -------------------------------------------------------------
        # Card Esperado (Esquerda)
        card_w = int((w - 200) / 2)
        draw.rectangle([15, 15, 15 + card_w, 105], fill=(15, 23, 26, 235), outline=(0, 229, 255), width=2)
        draw.text((25, 22), "🎯 CLASSE ESPERADA (ALVO):", font=self.font_bold, fill=(0, 229, 255))
        draw.text((25, 45), f"{self.expected_code}  [{exp_letter_name}]", font=self.font_title, fill=(255, 255, 255))
        draw.text((25, 75), exp_desc[:46] + ("..." if len(exp_desc) > 46 else ""), font=self.font_small, fill=(160, 177, 182))

        # Card Recebido (Direita do centro)
        card2_x = 25 + card_w
        is_match = guidance.get('match', False)
        rec_color = (0, 255, 128) if is_match else ((255, 82, 82) if confidence > 0.4 else (255, 180, 0))
        border_color = (0, 255, 128) if is_match else (255, 150, 0)
        
        draw.rectangle([card2_x, 15, card2_x + card_w, 105], fill=(15, 23, 26, 235), outline=border_color, width=2)
        draw.text((card2_x + 10, 22), "🤖 CLASSE RECEBIDA (IA):", font=self.font_bold, fill=border_color)
        
        if detected_code and confidence > 0.1:
            draw.text((card2_x + 10, 45), f"{detected_code}  [{det_letter_name}]", font=self.font_title, fill=(255, 255, 255))
            draw.text((card2_x + 10, 75), f"Confiança: {confidence*100:.1f}%  |  FPS: {fps:.0f}", font=self.font_bold, fill=rec_color)
        else:
            draw.text((card2_x + 10, 50), "Aguardando posicionar a mão...", font=self.font_normal, fill=(160, 177, 182))

        # Título do Card PiP
        pip_w, pip_h = 160, 170
        pip_x, pip_y = w - pip_w - 20, 20
        draw.text((pip_x + 25, pip_y + 8), "MODELO ALVO", font=self.font_bold, fill=(255, 229, 0))
        draw.text((pip_x + 35, pip_y + 24), exp_letter_name, font=self.font_small, fill=(255, 255, 255))

        # -------------------------------------------------------------
        # 2. PAINEL INFERIOR DE FEEDBACK BIOMECÂNICO
        # -------------------------------------------------------------
        footer_h = 135
        footer_y = h - footer_h - 15
        
        if is_match:
            box_bg = (10, 35, 20, 240)
            box_outline = (0, 255, 128)
            status_title = "✅ SINAL CORRETO! CONFIGURAÇÃO COMPATÍVEL"
            title_color = (0, 255, 128)
        else:
            box_bg = (35, 20, 15, 240)
            box_outline = (255, 150, 0)
            status_title = "⚠️ DIVERGÊNCIA IDENTIFICADA - AJUSTE SUA MÃO:"
            title_color = (255, 180, 0)

        draw.rectangle([15, footer_y, w - 15, footer_y + footer_h], fill=box_bg, outline=box_outline, width=2)
        draw.text((30, footer_y + 12), status_title, font=self.font_bold, fill=title_color)

        # Dicas detalhadas
        hints = guidance.get('hints', [])
        for idx, hint in enumerate(hints[:3]):
            icon = "•" if not is_match else "✓"
            draw.text((35, footer_y + 40 + (idx * 24)), f"{icon} {hint}", font=self.font_normal, fill=(255, 255, 255))

        # -------------------------------------------------------------
        # 3. BARRA DE ATALHOS NO RODAPÉ
        # -------------------------------------------------------------
        shortcuts = "[A-Z] Escolher Letra  |  [TAB] Próxima Letra  |  [C] Digitar Código  |  [ESPAÇO] Pausar  |  [Q] Sair"
        draw.text((25, h - 22), shortcuts, font=self.font_small, fill=(120, 140, 150))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def run(self, camera_index=0):
        print("\n[*] Abrindo webcam (dispositivo índice {})...".format(camera_index))
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[!] Câmera 0 indisponível. Tentando câmera 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("[!] Erro: Nenhuma webcam disponível no sistema.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "Validador LIBRAS - Reconhecimento e Feedback Biomecanico"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        print("\n" + "-"*65)
        print("  SISTEMA PRONTO! POSICIONE SUA MÃO EM FRENTE À CÂMERA.")
        print("  Pressione as letras [A-Z] na janela para trocar o sinal alvo.")
        print("-"*65 + "\n")

        prev_time = time.time()
        fps = 30.0
        last_log_state = ""

        detected_code = ""
        confidence = 0.0
        guidance = {'match': False, 'hints': ['Aguardando mão na câmera...']}

        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("[!] Falha na leitura do frame da webcam.")
                        break

                    # Espelhar imagem para comportamento natural de espelho
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    # Processamento MediaPipe
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb)

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        self.draw_skeleton(frame, hand_landmarks.landmark, w, h)

                        # Inferência da IA
                        features = self.extract_features(hand_landmarks.landmark)
                        detected_code, confidence = self.predict(features)

                        # Motor de Feedback Biomecânico
                        guidance = get_biomechanical_guidance(detected_code, self.expected_code)

                        # Log no console quando há mudança significativa
                        state_signature = f"{self.expected_code}_{detected_code}_{guidance['match']}"
                        if state_signature != last_log_state and confidence > 0.4:
                            match_str = "✅ COMBINOU!" if guidance['match'] else "❌ DIVERGÊNCIA"
                            print(f"[IA] Esperado: {self.expected_code} | Lido: {detected_code} ({confidence*100:.1f}%) -> {match_str}")
                            for h_txt in guidance['hints'][:2]:
                                print(f"     -> {h_txt}")
                            last_log_state = state_signature
                    else:
                        detected_code = ""
                        confidence = 0.0
                        guidance = {'match': False, 'hints': ['Posicione a mão visível no centro da câmera.']}

                    # Desenha o PiP com esqueleto do modelo alvo
                    self.draw_target_pip(frame, self.expected_code)

                    # Cálculo de FPS
                    now = time.time()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 0.001))
                    prev_time = now

                    # Renderiza o HUD completo
                    display_frame = self.draw_hud(frame, detected_code, confidence, guidance, fps)
                else:
                    # Modo Pausado: exibe aviso na tela
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "PAUSADO - Pressione ESPACO para retomar", 
                                (w // 2 - 250, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.imshow(window_name, display_frame)

                # Controle de Teclado
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), ord('Q'), 27): # ESC ou Q
                    print("\n[*] Encerrando validador...")
                    break
                elif key == 32: # Espaço
                    self.paused = not self.paused
                    print("[*] Análise pausada." if self.paused else "[*] Análise retomada.")
                elif key == 9: # TAB
                    self.cycle_canonical_sign(1)
                elif key in (ord('c'), ord('C')):
                    # Entrada de código personalizada via terminal
                    print("\n" + "="*50)
                    custom = input("Digite o código de 10 dígitos (ex: 4141000110): ").strip()
                    self.set_target_code(custom)
                    print("="*50 + "\n")
                elif 65 <= key <= 90: # A - Z (maiúsculas)
                    self.set_target_letter(chr(key))
                elif 97 <= key <= 122: # a - z (minúsculas)
                    self.set_target_letter(chr(key).upper())

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("[*] Recursos de vídeo e IA liberados com sucesso.\n")


def main():
    parser = argparse.ArgumentParser(description="Validador de Sinais LIBRAS com Feedback Biomecânico em Tempo Real")
    parser.add_argument("--letra", type=str, default="V", help="Letra canônica inicial (ex: V, B, A, L)")
    parser.add_argument("--classe", type=str, default="", help="Código de 10 dígitos inicial (ex: 4141000110)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0)")
    args = parser.parse_args()

    initial = args.classe if args.classe else args.letra
    validator = LibrasLiveValidator(initial_class=initial)
    validator.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
