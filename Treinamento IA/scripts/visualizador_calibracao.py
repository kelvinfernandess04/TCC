#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Studio de Visualização 3D de Calibração LIBRAS (visualizador_calibracao.py)
========================================================================
Ferramenta interativa de alta performance para inspeção e auditoria visual 3D
das sementes calibradas, matrizes de tolerância articular, pesos punitivos e
comparação direta com o modelo cinemático teórico e os dados reais gravados.

Recursos:
- Rotação 3D orbital livre via clique e arrasto do mouse.
- Zoom suave (Scroll do mouse ou teclas +/-) e Pan (Botão direito).
- Alternância instantânea entre as 9 classes e suas sub-sementes (Frontal/Perfil).
- [T] Camada de Tolerâncias Articulares 3D (Halos de desvio padrão por junta).
- [P] Mapa de Calor de Pesos Punitivos (Destaque nas juntas mais discriminativas).
- [G] Modo Fantasma (Sobreposição da pose teórica DADADADAFP do HandKinematicsDirect).
- [O] Nuvem de Pontos Empírica (Amostras reais do dataset gravado sobre a semente).
- [Espaço] Modo Turntable (Rotação orbital automática contínua 360°).
- [1, 2, 3, 4] Presets de Câmera: Frontal, Lateral, Superior e Isométrica.
"""

import os
import sys
import math
import time
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

# Configuração de encoding para Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
DATA_DIR = os.path.join(BASE_DIR, "Treinamento IA", "data")
SEEDS_FILE = os.path.join(BASE_DIR, "seeds_calibradas.json")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_maos")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from kinematic_seed_generator import HandKinematicsDirect, rot_x, rot_y, rot_z
from pipeline_calibracao_multiagente import Agent2_SpatialNormalizer

# Mapeamento oficial das 9 classes gravadas para a taxonomia DADADADAFP
CLASSES_CONFIG = {
    "classe_PALMA_ABERTA": {
        "code": "0000000000",
        "name": "Palma Aberta",
        "desc": "Mão Aberta Total com Dedos Separados (Estágio 0, A=0)"
    },
    "classe_B": {
        "code": "0101010100",
        "name": "Sinal B",
        "desc": "Mão Espalmada com Dedos Unidos (Estágio 0, A=1)"
    },
    "classe_C": {
        "code": "1111111100",
        "name": "Sinal C",
        "desc": "Mão em C Curvada (Estágio 1, A=1)"
    },
    "classe_CONCHA": {
        "code": "2121212100",
        "name": "Mão Concha",
        "desc": "Mão Concha / Plataforma Semi-fletida (Estágio 2, A=1)"
    },
    "classe_A": {
        "code": "3131313111",
        "name": "Sinal A",
        "desc": "Punho Fechado com Polegar Oposto (Estágio 3, F=1, P=1)"
    },
    "classe_I": {
        "code": "0131313100",
        "name": "Sinal I",
        "desc": "Mindinho Levantado (D4=0, D1..D3=3)"
    },
    "classe_L": {
        "code": "3131310000",
        "name": "Sinal L",
        "desc": "Indicador e Polegar a 90 Graus (D1=0, D2..D4=3, A0=0)"
    },
    "classe_V": {
        "code": "3131000000",
        "name": "Sinal V",
        "desc": "Indicador e Médio em V (D1=0, D2=0, A1=0)"
    },
    "classe_W": {
        "code": "3100000000",
        "name": "Sinal W",
        "desc": "Indicador, Médio e Anelar em W (D1..D3=0, D4=3, A=0)"
    }
}

from seed_extractor import generate_anatomical_hand_3d

CALIBRATION_SETTINGS_FILE = os.path.join(DATA_DIR, "calibration_settings.json")

CLASSES_PARAMS = {
    'classe_PALMA_ABERTA': ({'Thumb': 0, 'Index': 0, 'Middle': 0, 'Ring': 0, 'Pinky': 0}, {'Index_Thumb': 0, 'Middle_Index': 1, 'Ring_Middle': 1, 'Pinky_Ring': 1}, 0.0, 0.0),
    'classe_B': ({'Thumb': 0, 'Index': 0, 'Middle': 0, 'Ring': 0, 'Pinky': 0}, {'Index_Thumb': 1, 'Middle_Index': 0, 'Ring_Middle': 0, 'Pinky_Ring': 0}, 0.0, 0.0),
    'classe_C': ({'Thumb': 1, 'Index': 1, 'Middle': 1, 'Ring': 1, 'Pinky': 1}, {'Index_Thumb': 0.5}, 0.4, 0.0),
    'classe_CONCHA': ({'Thumb': 2, 'Index': 2, 'Middle': 2, 'Ring': 2, 'Pinky': 2}, {'Index_Thumb': 0.5}, 0.2, 0.0),
    'classe_A': ({'Thumb': 3, 'Index': 3, 'Middle': 3, 'Ring': 3, 'Pinky': 3}, {'Index_Thumb': 1}, 1.0, 1.0),
    'classe_I': ({'Thumb': 3, 'Index': 3, 'Middle': 3, 'Ring': 3, 'Pinky': 0}, {'Index_Thumb': 1}, 1.0, 1.0),
    'classe_L': ({'Thumb': 0, 'Index': 0, 'Middle': 3, 'Ring': 3, 'Pinky': 3}, {'Index_Thumb': 0}, 0.0, 0.0),
    'classe_V': ({'Thumb': 3, 'Index': 0, 'Middle': 0, 'Ring': 3, 'Pinky': 3}, {'Middle_Index': 1, 'Index_Thumb': 1}, 1.0, 1.0),
    'classe_W': ({'Thumb': 3, 'Index': 0, 'Middle': 0, 'Ring': 0, 'Pinky': 3}, {'Middle_Index': 1, 'Ring_Middle': 1, 'Index_Thumb': 1}, 1.0, 1.0)
}

# Paleta de cores BGR com alto contraste visual
FINGER_COLORS_BGR = {
    'Thumb':  (42, 110, 255),   # Laranja-Vibrante
    'Index':  (60, 235, 255),   # Amarelo Ouro
    'Middle': (90, 215, 80),    # Verde Esmeralda
    'Ring':   (240, 200, 30),   # Ciano / Turquesa
    'Pinky':  (210, 70, 190)    # Roxo Magenta
}

FINGER_SEGMENTS = {
    'Thumb':  [(0, 1), (1, 2), (2, 3), (3, 4)],
    'Index':  [(0, 5), (5, 6), (6, 7), (7, 8)],
    'Middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'Ring':   [(0, 13), (13, 14), (14, 15), (15, 16)],
    'Pinky':  [(0, 17), (17, 18), (18, 19), (19, 20)]
}

PALM_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]


class Calibration3DVisualizer:
    def __init__(self, seeds_json_path: str = SEEDS_FILE):
        self.seeds_file = seeds_json_path
        self.catalog = self._load_seeds_catalog(seeds_json_path)
        self.class_names = sorted(list(self.catalog.get("classes", {}).keys()))

        if not self.class_names:
            raise ValueError(f"Nenhuma classe encontrada no arquivo: {seeds_json_path}")

        self.normalizer = Agent2_SpatialNormalizer()
        self.kinematics = HandKinematicsDirect()

        # Carrega poses reais extraídas do vídeo se disponíveis
        self.real_captured_poses = None
        if os.path.exists(CALIBRATION_SETTINGS_FILE):
            try:
                with open(CALIBRATION_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    self.real_captured_poses = cdata.get("captured_poses")
            except Exception:
                pass

        # Cache de amostras reais do dataset para visualização da nuvem empírica
        self.dataset_cache = {}

        # Estado da Navegação
        self.current_class_idx = 0
        self.current_sub_idx = 0

        # Estado da Câmera 3D
        self.yaw = 25.0       # Graus
        self.pitch = -15.0    # Graus
        self.zoom = 380.0     # Pixels por unidade normalizada
        self.pan_x = 0        # Offset em X no viewport
        self.pan_y = 60       # Offset em Y no viewport (centraliza a mão)

        # Interação do Mouse
        self.is_dragging_rot = False
        self.is_dragging_pan = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Camadas Visuais (Toggles)
        self.show_tolerances = True
        self.show_weights = True
        self.show_ghost = True
        self.show_cloud = True
        self.auto_turntable = False
        self.show_labels = True

        # Configurações de Janela
        self.win_w = 1280
        self.win_h = 800
        self.window_name = "Studio de Visualizacao 3D de Calibracao LIBRAS"

    def _load_seeds_catalog(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            alt_path = os.path.join(DATA_DIR, "seeds", "seeds_calibradas.json")
            if os.path.exists(alt_path):
                path = alt_path
            else:
                raise FileNotFoundError(f"seeds_calibradas.json não encontrado em: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_dataset_cloud(self, class_name: str, max_samples: int = 25) -> List[np.ndarray]:
        """Carrega e normaliza uma amostra da nuvem de pontos real do dataset para a classe."""
        if class_name in self.dataset_cache:
            return self.dataset_cache[class_name]

        class_dir = os.path.join(DATASET_DIR, class_name)
        cloud_frames = []
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if fname.endswith(".json") and "perfil" not in fname:
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            frames = data.get("frames", [])
                            # Pega amostras espaçadas
                            step = max(1, len(frames) // max_samples)
                            for fr in frames[::step]:
                                lms = fr.get("landmarks", [])
                                if len(lms) == 21:
                                    pts = np.array([[p["x"], p["y"], p.get("z", 0.0)] for p in lms])
                                    norm_res = self.normalizer.normalize_frame(pts)
                                    cloud_frames.append(norm_res["landmarks_local"])
                                if len(cloud_frames) >= max_samples:
                                    break
                    except Exception:
                        pass
        self.dataset_cache[class_name] = cloud_frames
        return cloud_frames

    # -----------------------------------------------------------------------
    # PROJEÇÃO 3D E TRANSFORMAÇÃO DE CÂMERA
    # -----------------------------------------------------------------------

    def _project_3d_to_screen(self, pts_3d: np.ndarray, vp_w: int, vp_h: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projeta coordenadas 3D da mão para o espaço de tela 2D usando a matriz de órbita.
        Retorna:
          screen_pts: (N, 2) inteiros (X, Y na tela)
          depths: (N,) float (Profundidade Z para ordenação e efeitos de iluminação)
        """
        # Matriz de rotação da câmera orbital: R = R_x(pitch) * R_y(yaw)
        R = rot_x(self.pitch).dot(rot_y(self.yaw))
        pts_rot = pts_3d.dot(R.T)

        cx = vp_w // 2 + self.pan_x
        cy = vp_h // 2 + self.pan_y

        # Projeção ortogonal com escala/zoom e profundidade z
        screen_x = cx + (pts_rot[:, 0] * self.zoom)
        screen_y = cy - (pts_rot[:, 1] * self.zoom) # Inverte Y para subir na tela

        screen_pts = np.column_stack((screen_x, screen_y)).astype(int)
        depths = pts_rot[:, 2]
        return screen_pts, depths

    # -----------------------------------------------------------------------
    # CONTROLES DE MOUSE E JANELA
    # -----------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        # 1. Clique Esquerdo: Rotação Orbital
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging_rot = True
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.auto_turntable = False

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging_rot = False

        # 2. Clique Direito: Panorâmica (Pan)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_dragging_pan = True
            self.last_mouse_x = x
            self.last_mouse_y = y

        elif event == cv2.EVENT_RBUTTONUP:
            self.is_dragging_pan = False

        # 3. Movimento do Mouse
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y

            if self.is_dragging_rot:
                self.yaw += dx * 0.55
                self.pitch = np.clip(self.pitch + dy * 0.55, -89.0, 89.0)
                self.last_mouse_x = x
                self.last_mouse_y = y

            elif self.is_dragging_pan:
                self.pan_x += dx
                self.pan_y += dy
                self.last_mouse_x = x
                self.last_mouse_y = y

        # 4. Roda do Mouse (Zoom)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = flags
            if delta > 0:
                self.zoom = min(self.zoom * 1.08, 1200.0)
            else:
                self.zoom = max(self.zoom / 1.08, 120.0)

    # -----------------------------------------------------------------------
    # RENDERIZAÇÃO DO VIEWPORT 3D
    # -----------------------------------------------------------------------

    def _render_scene(self, vp_w: int, vp_h: int) -> np.ndarray:
        # Fundo elegante Dark Space (Vinho-Escuro / Grafite)
        viewport = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
        viewport[:] = (20, 18, 24)

        # Grade isométrica de chão sutil
        self._draw_ground_grid(viewport, vp_w, vp_h)

        c_name = self.class_names[self.current_class_idx]
        cls_data = self.catalog["classes"][c_name]
        sub_seeds = list(cls_data["sub_seeds"].values())

        if not sub_seeds:
            return viewport

        self.current_sub_idx = min(self.current_sub_idx, len(sub_seeds) - 1)
        active_seed = sub_seeds[self.current_sub_idx]

        seed_lms_3d = np.array([[p["x"], p["y"], p["z"]] for p in active_seed["landmarks_3d"]])
        weights = np.array(cls_data.get("discriminative_joint_weights", np.ones(21)))
        thresholds = np.array(active_seed["tolerance_matrix"]["joint_thresholds"])

        # 1. Camada Opcional: Nuvem de Amostras Reais (Dataset Cloud)
        if self.show_cloud:
            cloud_samples = self._load_dataset_cloud(c_name, max_samples=18)
            for cloud_pts in cloud_samples:
                c_scr, _ = self._project_3d_to_screen(cloud_pts, vp_w, vp_h)
                for pt in c_scr:
                    if 0 <= pt[0] < vp_w and 0 <= pt[1] < vp_h:
                        cv2.circle(viewport, (pt[0], pt[1]), 2, (75, 75, 110), -1, cv2.LINE_AA)

        # 2. Camada Opcional: Modo Fantasma (Pose Anatômica de Referência do Vídeo / DADADADAFP)
        if self.show_ghost and c_name in CLASSES_CONFIG:
            if self.real_captured_poses and c_name in CLASSES_PARAMS:
                f_st, sp_st, opp, ip = CLASSES_PARAMS[c_name]
                theo_raw = generate_anatomical_hand_3d(f_st, sp_st, opp, ip, captured_poses=self.real_captured_poses)
            else:
                code = CLASSES_CONFIG[c_name]["code"]
                theo_raw = self.kinematics.build_landmarks_from_code(code)
            theo_norm = self.normalizer.normalize_frame(theo_raw)["landmarks_local"]
            t_scr, _ = self._project_3d_to_screen(theo_norm, vp_w, vp_h)

            # Desenha ossos fantasmas em Ciano Elétrico Translúcido
            for finger, segments in FINGER_SEGMENTS.items():
                for i1, i2 in segments:
                    p1, p2 = (t_scr[i1][0], t_scr[i1][1]), (t_scr[i2][0], t_scr[i2][1])
                    cv2.line(viewport, p1, p2, (220, 180, 50), 1, cv2.LINE_AA)
            for i1, i2 in PALM_BONES:
                p1, p2 = (t_scr[i1][0], t_scr[i1][1]), (t_scr[i2][0], t_scr[i2][1])
                cv2.line(viewport, p1, p2, (180, 140, 40), 1, cv2.LINE_AA)

        # 3. Projeção da Semente Principal Calibrada
        s_scr, depths = self._project_3d_to_screen(seed_lms_3d, vp_w, vp_h)

        # 3.1 Desenho dos Ossos da Palma (Estrutura de Sustentação)
        for i1, i2 in PALM_BONES:
            p1 = (s_scr[i1][0], s_scr[i1][1])
            p2 = (s_scr[i2][0], s_scr[i2][1])
            # Profundidade média para shading
            avg_z = (depths[i1] + depths[i2]) / 2.0
            shade = np.clip(180 + int(avg_z * 70), 90, 255)
            bone_color = (int(110 * shade / 255), int(120 * shade / 255), int(135 * shade / 255))
            cv2.line(viewport, p1, p2, bone_color, 2, cv2.LINE_AA)

        # 3.2 Desenho dos Ossos dos Dedos (Cores Anatômicas Vibrantes)
        for finger, segments in FINGER_SEGMENTS.items():
            base_col = FINGER_COLORS_BGR[finger]
            for i1, i2 in segments:
                p1 = (s_scr[i1][0], s_scr[i1][1])
                p2 = (s_scr[i2][0], s_scr[i2][1])
                avg_z = (depths[i1] + depths[i2]) / 2.0
                shade = np.clip(1.0 + (avg_z * 0.4), 0.55, 1.35)

                bone_col = (
                    int(np.clip(base_col[0] * shade, 0, 255)),
                    int(np.clip(base_col[1] * shade, 0, 255)),
                    int(np.clip(base_col[2] * shade, 0, 255))
                )
                cv2.line(viewport, p1, p2, bone_col, 3, cv2.LINE_AA)

        # 3.3 Camada Opcional: Halos de Tolerância Articular 3D
        if self.show_tolerances:
            overlay_tol = viewport.copy()
            for j in range(21):
                pt = s_scr[j]
                rad = max(4, int(thresholds[j] * self.zoom * 0.95))
                # Halo esférico azulado / verde
                cv2.circle(overlay_tol, (pt[0], pt[1]), rad, (130, 220, 110), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay_tol, 0.45, viewport, 0.55, 0, viewport)

        # 3.4 Desenho dos Nós Articulares (Joint Nodes com Mapa de Calor de Pesos)
        for j in range(21):
            pt = s_scr[j]
            z_val = depths[j]
            w_val = weights[j]

            # Raio base do nó varia com o peso punitivo W_j
            base_rad = 4
            if self.show_weights:
                # Juntas mais importantes ganham diâmetro maior e brilho dourado
                base_rad = int(np.clip(3 + (w_val - 1.0) * 4.5, 4, 11))
                if w_val > 1.2:
                    node_col = (40, 220, 255) # Dourado brilhante
                elif w_val > 1.05:
                    node_col = (80, 240, 160) # Verde limão
                else:
                    node_col = (235, 235, 245) # Branco suave
            else:
                node_col = (240, 240, 250)

            # Brilho central
            cv2.circle(viewport, (pt[0], pt[1]), base_rad, node_col, -1, cv2.LINE_AA)
            cv2.circle(viewport, (pt[0], pt[1]), base_rad + 1, (20, 20, 30), 1, cv2.LINE_AA)

            # Rótulo de índice da junta
            if self.show_labels and j in [0, 4, 8, 12, 16, 20]:
                cv2.putText(viewport, str(j), (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 200, 215), 1, cv2.LINE_AA)

        return viewport

    def _draw_ground_grid(self, img: np.ndarray, w: int, h: int):
        """Desenha grade 3D no plano basal para dar forte percepção de profundidade espacial."""
        grid_color = (32, 28, 40)
        plane_y = -0.15 # Plano do pulso
        grid_size = 1.6
        step = 0.4

        # Linhas em X
        for gx in np.arange(-grid_size, grid_size + 0.01, step):
            p1_3d = np.array([[gx, plane_y, -grid_size]])
            p2_3d = np.array([[gx, plane_y, grid_size]])
            s1, _ = self._project_3d_to_screen(p1_3d, w, h)
            s2, _ = self._project_3d_to_screen(p2_3d, w, h)
            cv2.line(img, (s1[0][0], s1[0][1]), (s2[0][0], s2[0][1]), grid_color, 1, cv2.LINE_AA)

        # Linhas em Z
        for gz in np.arange(-grid_size, grid_size + 0.01, step):
            p1_3d = np.array([[-grid_size, plane_y, gz]])
            p2_3d = np.array([[grid_size, plane_y, gz]])
            s1, _ = self._project_3d_to_screen(p1_3d, w, h)
            s2, _ = self._project_3d_to_screen(p2_3d, w, h)
            cv2.line(img, (s1[0][0], s1[0][1]), (s2[0][0], s2[0][1]), grid_color, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # PAINEL LATERAL HUD ELEGANTE (GLASSMORPHIC TELEMETRY)
    # -----------------------------------------------------------------------

    def _render_hud_panel(self, canvas: np.ndarray, hud_w: int, h: int):
        # Painel lateral escuro com divisor vertical
        cv2.rectangle(canvas, (0, 0), (hud_w, h), (14, 13, 19), -1)
        cv2.line(canvas, (hud_w, 0), (hud_w, h), (42, 38, 55), 2)

        c_name = self.class_names[self.current_class_idx]
        cls_data = self.catalog["classes"][c_name]
        sub_seeds = list(cls_data["sub_seeds"].values())
        active_seed = sub_seeds[self.current_sub_idx]

        cfg = CLASSES_CONFIG.get(c_name, {"code": "----------", "name": c_name, "desc": "Classe Personalizada"})

        y = 35

        # 1. Cabeçalho do App
        cv2.putText(canvas, "LIBRAS 3D STUDIO", (20, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 205, 75), 2, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas, "Visualizador de Calibracao Biomecanica", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 145, 165), 1, cv2.LINE_AA)

        # 2. Card da Classe Selecionada
        y += 35
        cv2.rectangle(canvas, (18, y), (hud_w - 18, y + 105), (25, 23, 34), -1)
        cv2.rectangle(canvas, (18, y), (hud_w - 18, y + 105), (55, 50, 75), 1)

        cv2.putText(canvas, f"CLASSE ({self.current_class_idx + 1}/{len(self.class_names)})", (30, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{cfg['name']}", (30, y + 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Sub-Semente: {active_seed['seed_name']}", (30, y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Amostras Reais: {active_seed['sample_count']} frames", (30, y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 145, 170), 1, cv2.LINE_AA)

        # 3. Taxonomia DADADADAFP
        y += 125
        cv2.rectangle(canvas, (18, y), (hud_w - 18, y + 80), (25, 23, 34), -1)
        cv2.rectangle(canvas, (18, y), (hud_w - 18, y + 80), (55, 50, 75), 1)

        cv2.putText(canvas, "TAXONOMIA DADADADAFP:", (30, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"[{cfg['code']}]", (30, y + 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (100, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{cfg['desc']}", (30, y + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 175, 195), 1, cv2.LINE_AA)

        # 4. Telemetria e Pesos Punitivos
        y += 100
        cv2.rectangle(canvas, (18, y), (hud_w - 18, y + 115), (25, 23, 34), -1)
        cv2.rectangle(canvas, (18, y), (hud_w - 18, y + 115), (55, 50, 75), 1)

        weights = np.array(cls_data.get("discriminative_joint_weights", np.ones(21)))
        thresholds = np.array(active_seed["tolerance_matrix"]["joint_thresholds"])
        top_j = int(np.argmax(weights))

        cv2.putText(canvas, "TELEMETRIA DA CALIBRACAO:", (30, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 235, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"- Tolerancia Media: {np.mean(thresholds):.3f} rad", (30, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"- Peso Punitivo Medio: {np.mean(weights):.2f}x", (30, y + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"- Junta Mais Critica: J{top_j} (W={weights[top_j]:.2f})", (30, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 215, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"- Orientacao: Yaw {self.yaw:.1f}° | Pitch {self.pitch:.1f}°", (30, y + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 155, 180), 1, cv2.LINE_AA)

        # 5. Status das Camadas Ativas
        y += 135
        cv2.putText(canvas, "CAMADAS VISUAIS:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 205, 75), 1, cv2.LINE_AA)
        y += 18
        self._draw_toggle_badge(canvas, 20, y, "[T] Tolerancias 3D", self.show_tolerances)
        self._draw_toggle_badge(canvas, 180, y, "[P] Pesos Heatmap", self.show_weights)
        y += 26
        self._draw_toggle_badge(canvas, 20, y, "[G] Ghost Teorico", self.show_ghost)
        self._draw_toggle_badge(canvas, 180, y, "[O] Nuvem Dataset", self.show_cloud)
        y += 26
        self._draw_toggle_badge(canvas, 20, y, "[Espaco] Turntable 360", self.auto_turntable)
        self._draw_toggle_badge(canvas, 180, y, "[L] Labels Juntas", self.show_labels)

        # 6. Guia Rápido de Teclas
        y += 45
        cv2.putText(canvas, "CONTROLES:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 210, 255), 1, cv2.LINE_AA)
        y += 18
        controls = [
            ("[A] / [D] ou [<] / [>]", "Navegar entre Classes"),
            ("[W] / [S] ou [^] / [v]", "Alternar Sub-Semente"),
            ("[Mouse Drag]", "Girar Orbita 3D"),
            ("[Scroll] ou [+]/[-]", "Zoom In / Zoom Out"),
            ("[Botao Direito]", "Arrastar / Pan"),
            ("[1, 2, 3, 4]", "Câmera Front/Lat/Sup/Iso"),
            ("[R]", "Resetar Posicao da Câmera"),
            ("[ESC] / [Q]", "Sair do Visualizador")
        ]
        for k_txt, desc in controls:
            cv2.putText(canvas, f"{k_txt:22s}: {desc}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 165, 185), 1, cv2.LINE_AA)
            y += 18

    def _draw_toggle_badge(self, img: np.ndarray, x: int, y: int, label: str, is_active: bool):
        col = (90, 210, 110) if is_active else (80, 75, 95)
        status = "ON" if is_active else "OFF"
        cv2.putText(img, f"{label}: {status}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # LOOP PRINCIPAL DA APLICAÇÃO
    # -----------------------------------------------------------------------

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        hud_w = 360
        last_time = time.time()

        print("\n" + "="*70)
        print(" LIBRAS 3D STUDIO - VISUALIZADOR DE CALIBRAÇÃO ATIVADO ")
        print("="*70)
        print(f"[*] Carregadas {len(self.class_names)} classes calibradas.")
        print("[*] Clique e arraste na tela com o mouse para girar em 3D livre!")
        print("[*] Pressione [H] para alternar ajuda, [Espaço] para Turntable 360°.")

        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Rotação contínua no modo Turntable
            if self.auto_turntable:
                self.yaw = (self.yaw + dt * 35.0) % 360.0

            # Renderiza Viewport 3D da Mão
            vp_w = self.win_w - hud_w
            vp_h = self.win_h
            vp_img = self._render_scene(vp_w, vp_h)

            # Cria Canvas Geral unificando HUD + Viewport
            canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
            self._render_hud_panel(canvas, hud_w, self.win_h)
            canvas[:, hud_w:] = vp_img

            cv2.imshow(self.window_name, canvas)

            # Captura de Teclas com tratamento estendido (Setas e Atalhos)
            key = cv2.waitKeyEx(16) & 0xFFFFFF

            if key == 27 or key == ord('q') or key == ord('Q'):
                break

            # Navegação de Classes (A / D ou Setas Esquerda / Direita)
            elif key == ord('d') or key == ord('D') or key == 2555904: # Right arrow
                self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                self.current_sub_idx = 0

            elif key == ord('a') or key == ord('A') or key == 2424832: # Left arrow
                self.current_class_idx = (self.current_class_idx - 1) % len(self.class_names)
                self.current_sub_idx = 0

            # Navegação de Sub-Sementes (W / S ou Setas Cima / Baixo)
            elif key == ord('w') or key == ord('W') or key == 2490368: # Up arrow
                c_name = self.class_names[self.current_class_idx]
                n_subs = len(self.catalog["classes"][c_name]["sub_seeds"])
                self.current_sub_idx = (self.current_sub_idx + 1) % max(n_subs, 1)

            elif key == ord('s') or key == ord('S') or key == 2621440: # Down arrow
                c_name = self.class_names[self.current_class_idx]
                n_subs = len(self.catalog["classes"][c_name]["sub_seeds"])
                self.current_sub_idx = (self.current_sub_idx - 1) % max(n_subs, 1)

            # Toggles Visuais
            elif key == ord('t') or key == ord('T'):
                self.show_tolerances = not self.show_tolerances

            elif key == ord('p') or key == ord('P'):
                self.show_weights = not self.show_weights

            elif key == ord('g') or key == ord('G'):
                self.show_ghost = not self.show_ghost

            elif key == ord('o') or key == ord('O'):
                self.show_cloud = not self.show_cloud

            elif key == ord('l') or key == ord('L'):
                self.show_labels = not self.show_labels

            elif key == 32: # Barra de espaço: Turntable
                self.auto_turntable = not self.auto_turntable

            # Presets de Câmera
            elif key == ord('1'): # Frontal
                self.yaw = 0.0
                self.pitch = 0.0
                self.auto_turntable = False
            elif key == ord('2'): # Lateral (Perfil)
                self.yaw = 90.0
                self.pitch = 0.0
                self.auto_turntable = False
            elif key == ord('3'): # Superior (Top)
                self.yaw = 0.0
                self.pitch = 85.0
                self.auto_turntable = False
            elif key == ord('4'): # Isométrica 3D
                self.yaw = 35.0
                self.pitch = -20.0
                self.auto_turntable = False

            # Reset de Câmera
            elif key == ord('r') or key == ord('R'):
                self.yaw = 25.0
                self.pitch = -15.0
                self.zoom = 380.0
                self.pan_x = 0
                self.pan_y = 60
                self.auto_turntable = False

            # Zoom via teclado
            elif key == ord('+') or key == ord('='):
                self.zoom = min(self.zoom * 1.10, 1200.0)
            elif key == ord('-') or key == ord('_'):
                self.zoom = max(self.zoom / 1.10, 120.0)

        cv2.destroyAllWindows()


def main():
    app = Calibration3DVisualizer()
    app.run()

if __name__ == "__main__":
    main()
