#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspetor e Player Sequencial de Sementes 3D (inspect_seeds.py)
=============================================================
Permite navegar, auditar e reproduzir em sequência todas as sementes biomecânicas
geradas no arquivo seeds.json diretamente a partir das capturas da calibração real,
com visão simultânea Frontal e Lateral (Perfil).

Controles:
  [ESPAÇO]        : Play / Pause (reprodução contínua sequencial das seeds)
  [D] / [SETA DIR]: Próxima seed (+1)
  [A] / [SETA ESQ]: Seed anterior (-1)
  [W] / [SETA CIMA]: Avançar 50 seeds (+50)
  [S] / [SETA BAIXO]: Voltar 50 seeds (-50)
  [1] .. [5]      : Pular para poses com Estágios de flexão 0, 1, 2, 3 ou 4
  [L]             : Alternar entre poses de exemplos clássicos de Libras (A, B, C, D, I, L, V, W...)
  [+] / [-]       : Aumentar / Diminuir velocidade da reprodução automática
  [R]             : Resetar ângulo de rotação 3D da câmera
  [Q] / [ESC]     : Sair
"""

import os
import sys
import json
import math
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# RENDERIZADOR TIPOGRÁFICO UNICODE / UTF-8 COM PILLOW
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

    def render_batch(self, img_bgr: np.ndarray, text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int], bool]]) -> np.ndarray:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for (text, pos, font_size, color_bgr, bold) in text_items:
            font = self.get_font(font_size, bold)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(pos, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Configuração de encoding UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, "data", "seeds", "seeds.json")
CALIBRATION_FILE = os.path.join(BASE_DIR, "data", "calibration_settings.json")

# Paleta de cores BGR anatômica de alto contraste
FINGER_COLORS_BGR = {
    'Thumb':  (40, 140, 255),   # Laranja Âmbar
    'Index':  (70, 225, 255),   # Amarelo Ouro
    'Middle': (110, 230, 130),  # Verde Esmeralda
    'Ring':   (240, 210, 80),   # Ciano Turquesa
    'Pinky':  (225, 120, 215)   # Magenta Lavanda
}

FINGER_SEGMENTS = {
    'Thumb':  [(0, 1), (1, 2), (2, 3), (3, 4)],
    'Index':  [(0, 5), (5, 6), (6, 7), (7, 8)],
    'Middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'Ring':   [(0, 13), (13, 14), (14, 15), (15, 16)],
    'Pinky':  [(0, 17), (17, 18), (18, 19), (19, 20)]
}

PALM_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]

LIBRAS_PRESETS = [
    ("Sinal 'A' (Fist/Thumb side)", "4141414110"),
    ("Sinal 'B' (Flat/Thumb folded)", "0101010110"),
    ("Sinal 'C' (Curved/Arco)", "1010101000"),
    ("Sinal 'D' (Index pointing)", "4141410110"),
    ("Sinal 'I' (Pinky up)", "0141414110"),
    ("Sinal 'L' (Index + Thumb)", "4141410000"),
    ("Sinal 'V' (Index + Middle)", "4141000110"),
    ("Sinal 'W' (Index + Middle + Ring)", "4100000110"),
    ("Palma Aberta (Todos estendidos)", "0000000000"),
    ("Dedos Juntos Retos (Continência)", "0101010100"),
    ("Garras / Hook (Estágio 2)", "2121212100"),
    ("Mesa / Tabletop (Estágio 3)", "3131313100"),
    ("Punho Cerrado (Estágio 4)", "4141414110")
]

def rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]], dtype=np.float64)

def rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[math.cos(r), 0.0, math.sin(r)], [0.0, 1.0, 0.0], [-math.sin(r), 0.0, math.cos(r)]], dtype=np.float64)


def render_skeleton_viewport(pts_3d: np.ndarray, vp_w: int, vp_h: int, yaw: float, pitch: float, title: str) -> np.ndarray:
    """Renderiza a pose 3D em um canvas estilizado com o ângulo de visão fornecido."""
    canvas = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
    canvas[:] = (20, 18, 28)

    cx = vp_w // 2
    scale = min(vp_w, vp_h) * 0.40
    cy = int(vp_h * 0.50 + 0.88 * scale)

    R = rot_x(pitch).dot(rot_y(yaw))

    # 1. Grade 3D isométrica no pulso
    grid_y = 0.10
    grid_col = (35, 32, 45)
    for gx in np.linspace(-0.6, 0.6, 5):
        p1 = np.array([gx, grid_y, -0.6]).dot(R.T)
        p2 = np.array([gx, grid_y, +0.6]).dot(R.T)
        x1, y1 = int(cx + p1[0] * scale), int(cy + p1[1] * scale)
        x2, y2 = int(cx + p2[0] * scale), int(cy + p2[1] * scale)
        cv2.line(canvas, (x1, y1), (x2, y2), grid_col, 1, cv2.LINE_AA)
    for gz in np.linspace(-0.6, 0.6, 5):
        p1 = np.array([-0.6, grid_y, gz]).dot(R.T)
        p2 = np.array([+0.6, grid_y, gz]).dot(R.T)
        x1, y1 = int(cx + p1[0] * scale), int(cy + p1[1] * scale)
        x2, y2 = int(cx + p2[0] * scale), int(cy + p2[1] * scale)
        cv2.line(canvas, (x1, y1), (x2, y2), grid_col, 1, cv2.LINE_AA)

    pts_rot = pts_3d.dot(R.T)
    screen_pts = []
    depths = []
    for i in range(21):
        sx = int(cx + pts_rot[i, 0] * scale)
        sy = int(cy + pts_rot[i, 1] * scale)
        screen_pts.append((sx, sy))
        depths.append(pts_rot[i, 2])

    # 2. Malha semi-transparente da palma
    palm_poly_indices = [0, 1, 5, 9, 13, 17]
    palm_pts = np.array([screen_pts[i] for i in palm_poly_indices], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [palm_pts], (45, 40, 55))
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # 3. Ossos da palma
    for i1, i2 in PALM_BONES:
        p1, p2 = screen_pts[i1], screen_pts[i2]
        avg_z = (depths[i1] + depths[i2]) / 2.0
        shade = np.clip(1.0 - (avg_z * 0.35), 0.6, 1.3)
        col = (int(120 * shade), int(115 * shade), int(135 * shade))
        cv2.line(canvas, p1, p2, col, 2, cv2.LINE_AA)

    # 4. Falanges dos dedos
    for fname in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
        base_col = FINGER_COLORS_BGR[fname]
        for i1, i2 in FINGER_SEGMENTS[fname]:
            p1, p2 = screen_pts[i1], screen_pts[i2]
            avg_z = (depths[i1] + depths[i2]) / 2.0
            shade = np.clip(1.0 - (avg_z * 0.35), 0.65, 1.35)
            bone_col = (
                int(np.clip(base_col[0] * shade, 0, 255)),
                int(np.clip(base_col[1] * shade, 0, 255)),
                int(np.clip(base_col[2] * shade, 0, 255))
            )
            cv2.line(canvas, p1, p2, bone_col, 3, cv2.LINE_AA)

    # 5. Articulações (Knots)
    for j in range(21):
        pt = screen_pts[j]
        if j in [4, 8, 12, 16, 20]:  # Pontas dos dedos
            cv2.circle(canvas, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (80, 220, 100), 1, cv2.LINE_AA)
        elif j == 0:  # Pulso
            cv2.circle(canvas, pt, 6, (200, 200, 240), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 4, (230, 230, 240), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (30, 30, 40), 1, cv2.LINE_AA)

    # 6. Moldura do Viewport
    cv2.rectangle(canvas, (0, 0), (vp_w, 24), (28, 26, 38), -1)
    cv2.line(canvas, (0, 24), (vp_w, 24), (137, 180, 250), 1)
    cv2.rectangle(canvas, (0, 0), (vp_w - 1, vp_h - 1), (137, 180, 250), 1)

    return canvas


def main():
    if not os.path.exists(SEEDS_FILE):
        print(f"[ERRO] Arquivo seeds.json não encontrado em: {SEEDS_FILE}")
        return

    print("[INICIANDO] Carregando banco de sementes seeds.json...")
    with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
        seeds_data = json.load(f)

    seed_keys = [k for k in seeds_data.keys() if not k.startswith('__')]
    total_seeds = len(seed_keys)
    print(f"[SUCESSO] {total_seeds:,} sementes carregadas com sucesso!")

    # Cache de arrays NumPy normalizados
    pts_cache: Dict[str, np.ndarray] = {}
    for k in seed_keys[:300]:  # Pré-carregar as primeiras 300
        raw = seeds_data[k]
        if isinstance(raw[0], dict):
            pts_cache[k] = np.array([[p['x'], p['y'], p['z']] for p in raw], dtype=np.float64)
        else:
            pts_cache[k] = np.array(raw, dtype=np.float64)

    def get_seed_pts(key: str) -> np.ndarray:
        if key not in pts_cache:
            raw = seeds_data[key]
            if isinstance(raw[0], dict):
                pts_cache[key] = np.array([[p['x'], p['y'], p['z']] for p in raw], dtype=np.float64)
            else:
                pts_cache[key] = np.array(raw, dtype=np.float64)
        return pts_cache[key]

    current_idx = 0
    is_playing = False
    play_delay = 0.08  # ~12 FPS por padrão
    last_play_time = 0.0
    libras_preset_idx = 0

    window_name = "Auditoria e Player Sequencial de Sementes 3D (LIBRAS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Rotação interativa
    frontal_yaw = 15.0
    frontal_pitch = -12.0

    stage_names = {
        '0': "Estendido (0°)",
        '1': "Curvado (Concha)",
        '2': "Gancho (Hook)",
        '3': "Plataforma (Tabletop)",
        '4': "Fechado (Punho)"
    }

    hud = UnicodeHUD()

    vp_w = 460
    vp_h = 490
    vp_y = 90
    vp1_x = 330
    vp2_x = 330 + vp_w + 20

    card_x = 22
    card_w = 290
    card_h = vp_h
    leg_y = vp_y + card_h - 105

    while True:
        key_code = seed_keys[current_idx]
        pts_3d = get_seed_pts(key_code)

        # Criar frame do player (1280 x 720)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (17, 16, 24)

        # -------------------------------------------------------------
        # CABEÇALHO SUPERIOR
        # -------------------------------------------------------------
        cv2.rectangle(frame, (0, 0), (1280, 72), (24, 24, 37), -1)
        cv2.line(frame, (0, 72), (1280, 72), (166, 227, 161), 2)

        status_play = "▶ REPRODUZINDO EM SEQUÊNCIA" if is_playing else "❚❚ PAUSADO"
        play_col = (166, 227, 161) if is_playing else (249, 226, 175)

        # -------------------------------------------------------------
        # ÁREA CENTRAL: DOIS VIEWPORTS 3D (FRONTAL + LATERAL PERFIL)
        # -------------------------------------------------------------
        vp_front = render_skeleton_viewport(pts_3d, vp_w, vp_h, frontal_yaw, frontal_pitch, "VISÃO 1: PERSPECTIVA FRONTAL / ISOMÉTRICA")
        frame[vp_y:vp_y + vp_h, vp1_x:vp1_x + vp_w] = vp_front

        vp_side = render_skeleton_viewport(pts_3d, vp_w, vp_h, yaw=90.0, pitch=-5.0, title="VISÃO 2: PERFIL LATERAL 90° (PROFUNDIDADE Z)")
        frame[vp_y:vp_y + vp_h, vp2_x:vp2_x + vp_w] = vp_side

        # -------------------------------------------------------------
        # COLUNA ESQUERDA: TELEMETRIA DA SEED ATUAL
        # -------------------------------------------------------------
        cv2.rectangle(frame, (card_x, vp_y), (card_x + card_w, vp_y + card_h), (24, 24, 37), -1)
        cv2.rectangle(frame, (card_x, vp_y), (card_x + card_w, vp_y + card_h), (69, 71, 90), 1)
        cv2.line(frame, (card_x + 10, leg_y), (card_x + card_w - 10, leg_y), (69, 71, 90), 1)

        # -------------------------------------------------------------
        # RODAPÉ DE COMANDOS E ATALHOS
        # -------------------------------------------------------------
        cv2.rectangle(frame, (0, 645), (1280, 720), (17, 17, 27), -1)
        cv2.line(frame, (0, 645), (1280, 645), (69, 71, 90), 1)

        # Decodificar os 10 dígitos: D4 A3 D3 A2 D2 A1 D1 A0 F P
        d4, a3, d3, a2, d2, a1, d1, a0, f, p = key_code

        # Textos completos renderizados com UnicodeHUD sem quebra de acentos (??)
        text_batch = [
            ("INSPETOR E PLAYER DE SEMENTES CINEMÁTICAS 3D (SEEDS.JSON)", (22, 14), 18, (205, 214, 244), True),
            (f"Seed [{current_idx + 1:,} / {total_seeds:,}]  |  Código DADADADAFP: {key_code}  |  Estado: {status_play}", (22, 42), 13, play_col, False),

            # Títulos dos Viewports 3D
            ("VISÃO 1: PERSPECTIVA FRONTAL / ISOMÉTRICA", (vp1_x + 12, vp_y + 4), 12, (137, 180, 250), True),
            (f"Yaw: {frontal_yaw:+.0f}° | Pitch: {frontal_pitch:+.0f}°", (vp1_x + 12, vp_y + vp_h - 20), 11, (140, 145, 165), False),
            ("VISÃO 2: PERFIL LATERAL 90° (PROFUNDIDADE Z)", (vp2_x + 12, vp_y + 4), 12, (137, 180, 250), True),
            ("Yaw: +90° | Pitch: -5°", (vp2_x + 12, vp_y + vp_h - 20), 11, (140, 145, 165), False),

            # Coluna Esquerda: Análise Taxonômica
            ("ANÁLISE TAXONÔMICA:", (card_x + 14, vp_y + 14), 14, (249, 226, 175), True),
            (f"• Mindinho (D4): {stage_names.get(d4, d4)}", (card_x + 14, vp_y + 46), 12, (225, 120, 215), True),
            (f"  Spread Min-Ane: {'Aberto' if a3=='0' else 'Fechado'}", (card_x + 14, vp_y + 68), 11, (180, 180, 190), False),
            (f"• Anelar (D3):   {stage_names.get(d3, d3)}", (card_x + 14, vp_y + 92), 12, (240, 210, 80), True),
            (f"  Spread Ane-Med: {'Aberto' if a2=='0' else 'Fechado'}", (card_x + 14, vp_y + 114), 11, (180, 180, 190), False),
            (f"• Médio (D2):    {stage_names.get(d2, d2)}", (card_x + 14, vp_y + 138), 12, (110, 230, 130), True),
            (f"  Spread Med-Ind: {'Aberto' if a1=='0' else 'Fechado'}", (card_x + 14, vp_y + 160), 11, (180, 180, 190), False),
            (f"• Indicador (D1):{stage_names.get(d1, d1)}", (card_x + 14, vp_y + 184), 12, (70, 225, 255), True),
            (f"  Spread Ind-Pol: {'Aberto' if a0=='0' else 'Fechado'}", (card_x + 14, vp_y + 206), 11, (180, 180, 190), False),
            (f"• Polegar (F):   {'Oposição Transv.' if f=='1' else 'No Plano da Palma'}", (card_x + 14, vp_y + 230), 12, (40, 140, 255), True),
            (f"• Ponta Pol.(P): {'Flexionada (IP)' if p=='1' else 'Estendida'}", (card_x + 14, vp_y + 254), 12, (40, 140, 255), True),

            # Legenda
            ("Cores das Articulações:", (card_x + 14, leg_y + 8), 11, (205, 214, 244), True),
            ("• Polegar: Laranja  • Indicador: Amarelo", (card_x + 14, leg_y + 28), 10, (249, 226, 175), False),
            ("• Médio: Verde      • Anelar: Ciano", (card_x + 14, leg_y + 46), 10, (166, 227, 161), False),
            ("• Mínimo: Magenta   • Pontas: Branco/Verde", (card_x + 14, leg_y + 64), 10, (245, 194, 231), False),

            # Rodapé
            ("[ESPAÇO]: Play/Pause Sequencial  |  [D]/[->]: Próxima  |  [A]/[<-]: Anterior  |  [W]/[S]: +/- 50 Seeds", (25, 658), 12, (166, 227, 161), False),
            ("[L]: Exemplos Libras (A, B, C, V, W...)  |  [1]..[5]: Filtrar Estágio Flexão  |  [+/-]: Velocidade  |  [Q]: Sair", (25, 684), 11, (205, 214, 244), False)
        ]

        frame = hud.render_batch(frame, text_batch)
        cv2.imshow(window_name, frame)

        # Gerenciar reprodução contínua automática
        wait_ms = 1
        if is_playing:
            now = time.time()
            if now - last_play_time >= play_delay:
                current_idx = (current_idx + 1) % total_seeds
                last_play_time = now

        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord('q'), 27]:  # Q ou ESC
            break

        elif key == 32:  # ESPAÇO -> Play / Pause
            is_playing = not is_playing
            last_play_time = time.time()

        elif key in [ord('d'), 83, 2555904]:  # D ou Seta Direita -> Próximo
            is_playing = False
            current_idx = (current_idx + 1) % total_seeds

        elif key in [ord('a'), 81, 2424832]:  # A ou Seta Esquerda -> Anterior
            is_playing = False
            current_idx = (current_idx - 1) % total_seeds

        elif key in [ord('w'), 82, 2490368]:  # W ou Seta Cima -> +50
            is_playing = False
            current_idx = (current_idx + 50) % total_seeds

        elif key in [ord('s'), 84, 2621440]:  # S ou Seta Baixo -> -50
            is_playing = False
            current_idx = (current_idx - 50 + total_seeds) % total_seeds

        elif key in [ord('+'), ord('=')]:  # Acelerar
            play_delay = max(0.02, play_delay - 0.02)

        elif key in [ord('-'), ord('_')]:  # Desacelerar
            play_delay = min(0.50, play_delay + 0.02)

        elif key in [ord('l'), ord('L')]:  # Pular para exemplo Libras
            is_playing = False
            libras_preset_idx = (libras_preset_idx + 1) % len(LIBRAS_PRESETS)
            target_desc, target_code = LIBRAS_PRESETS[libras_preset_idx]
            if target_code in seed_keys:
                current_idx = seed_keys.index(target_code)
                print(f"[PRESET LIBRAS] -> {target_desc} ({target_code})")

        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:  # Filtrar por estágio
            is_playing = False
            target_stage = str(int(chr(key)) - 1)  # '1'->'0', '2'->'1', ..., '5'->'4'
            # Procurar próxima seed que tenha esse estágio nos 4 dedos
            found = False
            for step_search in range(1, total_seeds):
                cand_idx = (current_idx + step_search) % total_seeds
                cand_code = seed_keys[cand_idx]
                if cand_code[0] == target_stage and cand_code[2] == target_stage and cand_code[4] == target_stage and cand_code[6] == target_stage:
                    current_idx = cand_idx
                    found = True
                    break
            if found:
                print(f"[FILTRO] Pousou na seed com 4 dedos no Estágio {target_stage}: {seed_keys[current_idx]}")

        elif key in [ord('r'), ord('R')]:  # Reset câmera
            frontal_yaw = 15.0
            frontal_pitch = -12.0

    cv2.destroyAllWindows()
    print("[FINALIZADO] Inspetor de sementes encerrado com sucesso.")

if __name__ == "__main__":
    main()
