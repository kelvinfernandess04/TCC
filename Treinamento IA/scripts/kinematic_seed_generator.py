"""
Kinematic Seed Generator (kinematic_seed_generator.py)
=====================================================
Specialist 3D Kinematics and Computational Biomechanics module for generating
anatomically valid 21-point 3D hand landmarks compatible with MediaPipe Hands
from the strict 10-digit DADADADAFP taxonomy.

Taxonomy Structure (DADADADAFP):
--------------------------------
1. [D] Mindinho (Pinky): Flexion (Stages 0 to 3)
2. [A] Abertura Mindinho-Anelar (Spread Pinky-Ring): (0 = Open / -15°, 1 = Closed / 0°)
3. [D] Anelar (Ring): Flexion (Stages 0 to 3)
4. [A] Abertura Anelar-Médio (Spread Ring-Middle): (0 = Open / -10°, 1 = Closed / 0°)
5. [D] Médio (Middle): Flexion (Stages 0 to 3)
6. [A] Abertura Médio-Indicador (Spread Middle-Index): (0 = Open / +10°, 1 = Closed / 0°)
7. [D] Indicador (Index): Flexion (Stages 0 to 3)
8. [A] Abertura Indicador-Polegar (Index-Thumb CMC Radial Abduction): (0 = Open / -55° total, 1 = Closed / -20° total)
9. [F] Transversal (Polegar) Opposition: (0 = In palm plane / Pitch 10°, 1 = Crossing palm / Pitch 35° towards Landmarks 9/13/17)
10. [P] Ponta do Polegar (IP) Distal Phalanx J4: (0 = Extended / 0°, 1 = Flexed / 65°)
"""

import os
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# 3D ROTATION UTILITIES (Eulerian / Direct SO(3))
# ---------------------------------------------------------------------------

def rot_x(deg: float) -> np.ndarray:
    """Rotation matrix around X-axis (Pitch / Flexion)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s],
        [0.0, s,    c]
    ], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    """Rotation matrix around Y-axis (Roll / Pronation / Torsion)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c]
    ], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    """Rotation matrix around Z-axis (Yaw / Abduction-Adduction Spread)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c,   -s,  0.0],
        [s,    c,  0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# DIRECT HAND KINEMATICS ENGINE
# ---------------------------------------------------------------------------

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


class HandKinematicsDirect:
    """
    Direct Geometric Forward Kinematics for MediaPipe Hands (21 3D Landmarks).
    - Coordinate Frame: Canonical Palm Space with Z=(0,0,1) strictly perpendicular to palm.
    - Fingers [D]: Pure sagittal forward flexion without any sideways lateral drift (Delta X = 0),
      preserving 100% rigid bone lengths from the extended open palm (baseline_open).
    - Spreads [A]: Clear lateral abduction in the palm plane.
    - Thumb: 3 simplified canonical states (Aberto, Junto, Transversal), with
      rigid bone lengths from extended palm.
    """

    STAGE_ANGLES = {
        0: (0.0, 0.0, 0.0),       # Reto / Estendido
        1: (25.0, 35.0, 25.0),    # Concha / Curvado suave
        2: (0.0, 85.0, 80.0),     # Gancho / Hook (MCP reto, pontas dobradas)
        3: (90.0, 0.0, 0.0),      # Mesa / Tabletop (MCP a 90° para a frente)
        4: (85.0, 95.0, 75.0)     # Punho / Fechado (cerrado na palma)
    }

    def __init__(self,
                 captured_landmarks: Optional[Dict[str, Any]] = None,
                 thumb_extracted: Optional[Dict[str, Any]] = None,
                 phalanx_lengths: Optional[Dict[str, Any]] = None):
        self.captured_landmarks = dict(captured_landmarks) if captured_landmarks else {}
        self.thumb_extracted = dict(thumb_extracted) if thumb_extracted else {}
        self.raw_phalanx_lengths = dict(phalanx_lengths) if phalanx_lengths else {}
        self._init_modular_bases()

    def _init_modular_bases(self):
        """Inicializa as bases da palma e pré-computa os comprimentos ósseos rígidos da mão aberta."""
        ref_pts_raw = None
        # 1. Tenta obter das extrações do polegar / mão estendida
        if self.thumb_extracted and 'thumb_open' in self.thumb_extracted:
            raw = self.thumb_extracted['thumb_open']
            if isinstance(raw, (list, np.ndarray)) and len(raw) == 21:
                ref_pts_raw = np.array(raw, dtype=np.float64)

        if ref_pts_raw is None:
            ref_pts_raw = self.get_landmark_array('thumb_open')
        if ref_pts_raw is None:
            ref_pts_raw = self.get_landmark_array('baseline_open')
        if ref_pts_raw is None:
            ref_pts_raw = self.get_landmark_array('spread_open')
        if ref_pts_raw is None:
            ref_pts_raw = self.get_landmark_array('spread_closed')

        if ref_pts_raw is not None and len(ref_pts_raw) == 21:
            self.ref_pts, self.R_canon = to_canonical_palm_frame(ref_pts_raw)
            self.palm_base = self.ref_pts[[0, 1, 5, 9, 13, 17]].copy()
        else:
            self.R_canon = np.eye(3)
            self.palm_base = np.array([
                [ 0.000,  0.000,  0.000],  # Wrist (0)
                [-0.280, -0.191,  0.000],  # Thumb CMC (1)
                [-0.236, -0.960, -0.041],  # Index MCP (5)
                [ 0.000, -0.997,  0.000],  # Middle MCP (9)
                [ 0.226, -0.943, -0.002],  # Ring MCP (13)
                [ 0.433, -0.819, -0.041]   # Pinky MCP (17)
            ], dtype=np.float64)
            self.ref_pts = np.zeros((21, 3), dtype=np.float64)
            self.ref_pts[0] = self.palm_base[0]
            self.ref_pts[1] = self.palm_base[1]
            self.ref_pts[5] = self.palm_base[2]
            self.ref_pts[9] = self.palm_base[3]
            self.ref_pts[13] = self.palm_base[4]
            self.ref_pts[17] = self.palm_base[5]

        # No referencial canônico:
        # rx = (1, 0, 0)
        # ry = (0, -1, 0)  [dedos para cima]
        # rz = (0, 0, 1)   [normal para a frente / observador]
        self.rx = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.ry = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        self.rz = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Definições das juntas e comprimentos ósseos rígidos dos 4 dedos
        self.finger_defs = {
            'Index':  {'mcp': 5,  'joints': [6, 7, 8]},
            'Middle': {'mcp': 9,  'joints': [10, 11, 12]},
            'Ring':   {'mcp': 13, 'joints': [14, 15, 16]},
            'Pinky':  {'mcp': 17, 'joints': [18, 19, 20]}
        }

        self.finger_lengths = {}
        self.finger_spread_dirs_open = {}
        self.finger_spread_dirs_closed = {}

        default_finger_lengths = {
            'Index':  (0.393407, 0.240129, 0.205338),
            'Middle': (0.422649, 0.271338, 0.222845),
            'Ring':   (0.396014, 0.258492, 0.216478),
            'Pinky':  (0.329636, 0.223851, 0.197460)
        }

        for fname, d in self.finger_defs.items():
            mcp_idx = d['mcp']
            j = d['joints']

            if fname in self.raw_phalanx_lengths and len(self.raw_phalanx_lengths[fname]) == 3:
                l1, l2, l3 = [float(x) for x in self.raw_phalanx_lengths[fname]]
            else:
                l1 = float(np.linalg.norm(self.ref_pts[j[0]] - self.ref_pts[mcp_idx]))
                l2 = float(np.linalg.norm(self.ref_pts[j[1]] - self.ref_pts[j[0]]))
                l3 = float(np.linalg.norm(self.ref_pts[j[2]] - self.ref_pts[j[1]]))

            def_l = default_finger_lengths.get(fname, (0.35, 0.23, 0.20))
            if l1 < 1e-4: l1 = def_l[0]
            if l2 < 1e-4: l2 = def_l[1]
            if l3 < 1e-4: l3 = def_l[2]
            self.finger_lengths[fname] = (l1, l2, l3)

            # Direção unitária de abertura em leque no plano da palma (Z = 0)
            v_open = (self.ref_pts[j[2]] - self.ref_pts[mcp_idx]).copy()
            v_open[2] = 0.0  # projeta no plano frontal da palma
            norm_open = np.linalg.norm(v_open)
            if norm_open > 1e-4:
                self.finger_spread_dirs_open[fname] = v_open / norm_open
            else:
                default_dirs = {
                    'Index':  np.array([-0.06, -0.998, 0.0]),
                    'Middle': np.array([0.00,  -1.000, 0.0]),
                    'Ring':   np.array([0.15,  -0.988, 0.0]),
                    'Pinky':  np.array([0.38,  -0.925, 0.0])
                }
                v_def = default_dirs.get(fname, self.ry.copy())
                self.finger_spread_dirs_open[fname] = v_def / np.linalg.norm(v_def)

            # Direção unitária de dedos juntos (estritamente paralela ao eixo longitudinal -Y)
            self.finger_spread_dirs_closed[fname] = self.ry.copy()

        # Comprimentos ósseos rígidos do polegar medidos da palma estendida
        if 'Thumb' in self.raw_phalanx_lengths and len(self.raw_phalanx_lengths['Thumb']) == 3:
            tl1, tl2, tl3 = [float(x) for x in self.raw_phalanx_lengths['Thumb']]
        else:
            tl1 = float(np.linalg.norm(self.ref_pts[2] - self.ref_pts[1]))
            tl2 = float(np.linalg.norm(self.ref_pts[3] - self.ref_pts[2]))
            tl3 = float(np.linalg.norm(self.ref_pts[4] - self.ref_pts[3]))

        if tl1 < 1e-4: tl1 = 0.415389
        if tl2 < 1e-4: tl2 = 0.319620
        if tl3 < 1e-4: tl3 = 0.248609
        self.thumb_lengths = (tl1, tl2, tl3)

    def get_landmark_array(self, step_key: str) -> Optional[np.ndarray]:
        """Obtém a matriz (21, 3) de landmarks normalizados para o passo solicitado."""
        entry = self.captured_landmarks.get(step_key)
        if entry is None:
            entry = self.thumb_extracted.get(step_key)
        if entry is None:
            return None
        if isinstance(entry, dict):
            pts = entry.get('pts_norm') or entry.get('frontal') or entry.get('lateral')
        elif isinstance(entry, (list, np.ndarray)):
            pts = entry
        else:
            pts = None

        if pts is not None and len(pts) == 21:
            return np.array(pts, dtype=np.float64)
        return None

    def get_source_for_thumb(self, thumb_state: int) -> np.ndarray:
        """
        Retorna as coordenadas 3D canônicas para o estado solicitado do polegar:
          0 = Aberto esticado (thumb_open)
          1 = Junto aos dedos com os dedos fechados (thumb_closed)
          2 = Na transversal (thumb_transversal)
        """
        arr_raw = None
        if self.thumb_extracted:
            keys = ['thumb_open', 'thumb_closed', 'thumb_transversal']
            if thumb_state < len(keys):
                arr_raw = self.thumb_extracted.get(keys[thumb_state])

        if arr_raw is None:
            if thumb_state == 0:
                arr_raw = self.get_landmark_array('thumb_open') or self.get_landmark_array('thumb_f0_p0') or self.get_landmark_array('spread_open')
            elif thumb_state == 1:
                arr_raw = self.get_landmark_array('thumb_closed') or self.get_landmark_array('spread_closed')
            else:
                arr_raw = self.get_landmark_array('thumb_transversal') or self.get_landmark_array('thumb_f1') or self.get_landmark_array('thumb_f1_p1') or self.get_landmark_array('thumb_f1_p0')

        if arr_raw is None:
            arr_raw = self.ref_pts

        if isinstance(arr_raw, list):
            arr_raw = np.array(arr_raw, dtype=np.float64)

        # Transforma o arr_raw para a mesma base canônica
        pts_can, _ = to_canonical_palm_frame(arr_raw)
        return pts_can

    @classmethod
    def from_calibration_file(cls, filepath: str) -> 'HandKinematicsDirect':
        """Instancia HandKinematicsDirect diretamente das marcações capturadas reais em calibration_settings.json."""
        if not os.path.exists(filepath):
            print(f"[HandKinematicsDirect] Arquivo de calibração não encontrado ({filepath}).")
            return cls()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            captured_landmarks = data.get('captured_landmarks', {})
            thumb_extracted = data.get('thumb_extracted', {})
            phalanx_lengths = data.get('phalanx_lengths', {})
            print(f"[HandKinematicsDirect] Carregadas {len(captured_landmarks)} capturas reais, {len(thumb_extracted)} dados de polegar e comprimentos calibrados ({len(phalanx_lengths)} dedos).")
            return cls(captured_landmarks=captured_landmarks,
                       thumb_extracted=thumb_extracted,
                       phalanx_lengths=phalanx_lengths)
        except Exception as e:
            print(f"[HandKinematicsDirect] Erro ao carregar calibração ({e}). Usando fallback padrão.")
            return cls()

    @staticmethod
    def is_valid_pose(dadadafafp_code: str) -> Tuple[bool, Optional[str]]:
        """
        Valida se o código de 10 dígitos DADADADAFP representa uma pose biomecanicamente possível
        com a simplificação anatômica (3 estados de polegar e IP desconsiderado com P=0).
        """
        if not isinstance(dadadafafp_code, str) or len(dadadafafp_code) != 10:
            return False, f"Code must be a 10-character string, got '{dadadafafp_code}'"

        d4_c, a3_c, d3_c, a2_c, d2_c, a1_c, d1_c, a0_c, f_c, p_c = dadadafafp_code

        if d4_c not in '01234' or d3_c not in '01234' or d2_c not in '01234' or d1_c not in '01234':
            return False, "Finger flexion stages [D] must be digits in '01234'"

        if a3_c not in '01' or a2_c not in '01' or a1_c not in '01' or a0_c not in '01':
            return False, "Spread states [A] must be '0' (Open) or '1' (Closed)"

        if f_c not in '01':
            return False, "Thumb opposition [F] must be '0' or '1'"

        # IP desconsiderado: aceita P=0 (ou P=1 normalizado)
        if p_c not in '01':
            return False, "Thumb IP [P] must be '0' or '1'"

        d4 = int(d4_c)  # Pinky
        a3 = int(a3_c)  # Pinky-Ring Spread
        d3 = int(d3_c)  # Ring
        a2 = int(a2_c)  # Ring-Middle Spread
        d2 = int(d2_c)  # Middle
        a1 = int(a1_c)  # Middle-Index Spread
        d1 = int(d1_c)  # Index
        a0 = int(a0_c)  # Index-Thumb CMC Abduction
        f  = int(f_c)   # Thumb Opposition
        p  = int(p_c)   # Thumb IP Flexion (desconsiderado, fixado em 0)

        # Regra do IP: sempre normalizado para P=0 (desconsiderado)
        if p != 0:
            return False, "Thumb IP flexion disregarded (P must be 0)"

        # 3 Estados do Polegar:
        # (A0=0, F=0) -> Aberto esticado
        # (A0=1, F=0) -> Junto aos dedos
        # (A0=1, F=1) -> Na transversal
        # (A0=0, F=1) -> Impossível (aberto e na transversal ao mesmo tempo)
        if a0 == 0 and f == 1:
            return False, "Thumb cannot be wide abducted (A0=0) and crossing palm (F=1) simultaneously"

        # Se o indicador está flexionado/fechado (D1 >= 2), o polegar não pode estar aberto esticado (A0=0)
        if d1 >= 2 and a0 == 0:
            return False, f"Thumb cannot be wide open (A0=0) when Index is flexed/closed (D1={d1})"

        # --- RESTRIÇÃO BIOMECÂNICA: Bloqueio de Abertura Lateral na Flexão ---
        if (d4 >= 2 or d3 >= 2) and a3 == 0:
            return False, f"Pinky-Ring spread (A3=0) impossible when Pinky (D4={d4}) or Ring (D3={d3}) is flexed/closed (D >= 2)"

        if (d3 >= 2 or d2 >= 2) and a2 == 0:
            return False, f"Ring-Middle spread (A2=0) impossible when Ring (D3={d3}) or Middle (D2={d2}) is flexed/closed (D >= 2)"

        if (d2 >= 2 or d1 >= 2) and a1 == 0:
            return False, f"Middle-Index spread (A1=0) impossible when Middle (D2={d2}) or Index (D1={d1}) is flexed/closed (D >= 2)"

        # --- RESTRIÇÃO BIOMECÂNICA: Juncturae Tendinum ---
        if d2 == 4 and d4 == 4 and d3 in (0, 3):
            return False, f"Ring (D3={d3}) cannot be fully extended/tabletop when Middle and Pinky are clenched (D2=4, D4=4)"

        return True, None

    def build_landmarks_from_code(self, dadadafafp_code: str) -> np.ndarray:
        """
        Montagem modular direta de cada dedo:
        - Flexão dos dedos 'D' movimentando estritamente PARA A FRENTE (plano sagital).
        - 3 estados simplificados para o polegar (Aberto, Junto, Na transversal).
        """
        is_valid, reason = self.is_valid_pose(dadadafafp_code)
        if not is_valid:
            raise ValueError(f"Invalid biomechanical pose code '{dadadafafp_code}': {reason}")

        d4 = int(dadadafafp_code[0])  # Pinky Flexion
        a3 = int(dadadafafp_code[1])  # Pinky-Ring Spread
        d3 = int(dadadafafp_code[2])  # Ring Flexion
        a2 = int(dadadafafp_code[3])  # Ring-Middle Spread
        d2 = int(dadadafafp_code[4])  # Middle Flexion
        a1 = int(dadadafafp_code[5])  # Middle-Index Spread
        d1 = int(dadadafafp_code[6])  # Index Flexion
        a0 = int(dadadafafp_code[7])  # Index-Thumb Abduction
        f  = int(dadadafafp_code[8])  # Thumb Opposition

        # Determina o estado do polegar: 0=Aberto, 1=Junto, 2=Na transversal
        if f == 1:
            thumb_state = 2
        elif a0 == 1:
            thumb_state = 1
        else:
            thumb_state = 0

        landmarks = np.zeros((21, 3), dtype=np.float64)

        # 1. Base da palma da mão
        landmarks[0]  = self.palm_base[0]  # Wrist (0)
        landmarks[1]  = self.palm_base[1]  # Thumb CMC (1)
        landmarks[5]  = self.palm_base[2]  # Index MCP (5)
        landmarks[9]  = self.palm_base[3]  # Middle MCP (9)
        landmarks[13] = self.palm_base[4]  # Ring MCP (13)
        landmarks[17] = self.palm_base[5]  # Pinky MCP (17)

        # 2. Dedos longos: Cinemática Direta pura com comprimentos rígidos da palma estendida
        # e flexão estritamente no plano sagital para a frente (+Z)
        fingers_data = [
            ('Index',  d1, a1, 5,  [6, 7, 8]),
            ('Middle', d2, 0,  9,  [10, 11, 12]),
            ('Ring',   d3, a2, 13, [14, 15, 16]),
            ('Pinky',  d4, a3, 17, [18, 19, 20])
        ]

        for fname, st, sp, mcp_idx, joint_idxs in fingers_data:
            u = self.finger_spread_dirs_open[fname] if sp == 0 else self.finger_spread_dirs_closed[fname]
            t1, t2, t3 = self.STAGE_ANGLES[st]
            a1_r = math.radians(t1)
            a2_r = math.radians(t1 + t2)
            a3_r = math.radians(t1 + t2 + t3)

            v1 = math.cos(a1_r) * u + math.sin(a1_r) * self.rz
            v2 = math.cos(a2_r) * u + math.sin(a2_r) * self.rz
            v3 = math.cos(a3_r) * u + math.sin(a3_r) * self.rz

            l1, l2, l3 = self.finger_lengths[fname]
            p_mcp = landmarks[mcp_idx]
            landmarks[joint_idxs[0]] = p_mcp + l1 * v1
            landmarks[joint_idxs[1]] = landmarks[joint_idxs[0]] + l2 * v2
            landmarks[joint_idxs[2]] = landmarks[joint_idxs[1]] + l3 * v3

        # 3. Polegar: reconstruído com comprimentos rígidos da palma estendida
        src_thumb = self.get_source_for_thumb(thumb_state)
        l1, l2, l3 = self.thumb_lengths

        v1 = src_thumb[2] - src_thumb[1]
        n1 = np.linalg.norm(v1)
        u1 = v1 / n1 if n1 > 1e-4 else -self.rx - 0.5 * self.ry

        v2 = src_thumb[3] - src_thumb[2]
        n2 = np.linalg.norm(v2)
        u2 = v2 / n2 if n2 > 1e-4 else u1

        v3 = src_thumb[4] - src_thumb[3]
        n3 = np.linalg.norm(v3)
        u3 = v3 / n3 if n3 > 1e-4 else u2

        landmarks[1] = src_thumb[1].copy()  # CMC 100% real lido da captura do polegar
        landmarks[2] = landmarks[1] + l1 * (u1 / np.linalg.norm(u1))
        landmarks[3] = landmarks[2] + l2 * (u2 / np.linalg.norm(u2))
        landmarks[4] = landmarks[3] + l3 * (u3 / np.linalg.norm(u3))

        return landmarks

    def generate_all_valid_seeds(self, max_spread_stage: int = 1) -> Dict[str, List[Dict[str, float]]]:
        """
        Gera todas as sementes biomecanicamente válidas com os 3 estados simplificados de polegar
        e flexão sagital para a frente dos dedos longos.
        """
        seeds: Dict[str, List[Dict[str, float]]] = {}
        valid_count = 0
        pruned_count = 0

        regular_states = [0, 1, 2, 3, 4]

        for d4 in regular_states:
            for d3 in regular_states:
                a3_options = [0, 1] if (d4 <= max_spread_stage and d3 <= max_spread_stage) else [1]
                for a3 in a3_options:

                    for d2 in regular_states:
                        a2_options = [0, 1] if (d3 <= max_spread_stage and d2 <= max_spread_stage) else [1]
                        for a2 in a2_options:

                            for d1 in regular_states:
                                a1_options = [0, 1] if (d2 <= max_spread_stage and d1 <= max_spread_stage) else [1]
                                for a1 in a1_options:

                                    # 3 Estados do Polegar (P sempre 0):
                                    # (0, 0): Aberto esticado (apenas se D1 <= max_spread_stage)
                                    # (1, 0): Junto aos dedos
                                    # (1, 1): Na transversal
                                    thumb_states = [(1, 0), (1, 1)]
                                    if d1 <= max_spread_stage:
                                        thumb_states.insert(0, (0, 0))

                                    for a0, f in thumb_states:
                                        code = f"{d4}{a3}{d3}{a2}{d2}{a1}{d1}{a0}{f}0"

                                        is_valid, _ = self.is_valid_pose(code)
                                        if not is_valid:
                                            pruned_count += 1
                                            continue

                                        lms_3d = self.build_landmarks_from_code(code)

                                        seeds[code] = [
                                            {
                                                "x": float(round(pt[0], 6)),
                                                "y": float(round(pt[1], 6)),
                                                "z": float(round(pt[2], 6))
                                            }
                                            for pt in lms_3d
                                        ]
                                        valid_count += 1

        print(f"[HandKinematicsDirect] Geradas {valid_count} sementes simplificadas (Podadas {pruned_count} poses inválidas).")
        return seeds

    def export_seeds_json(self, output_file_path: str) -> None:
        """
        Generate and save seeds.json with all valid 21-point landmarks.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
        seeds_dict = self.generate_all_valid_seeds()

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(seeds_dict, f, indent=2)

        print(f"[HandKinematicsDirect] Exported {len(seeds_dict)} seeds successfully to: {output_file_path}")


# ---------------------------------------------------------------------------
# CLI / MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    print("===================================================================")
    print("  GERADOR DE SEMENTES CINEMÁTICAS 3D (MEDIA-PIPE / DADADADAFP)     ")
    print("===================================================================")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seeds_file = os.path.join(base_dir, 'data', 'seeds', 'seeds.json')
    calib_file = os.path.join(base_dir, 'data', 'calibration_settings.json')

    if os.path.exists(calib_file):
        print(f"[HandKinematicsDirect] Carregando calibração real dual-angle: {calib_file}")
        generator = HandKinematicsDirect.from_calibration_file(calib_file)
    else:
        print("[HandKinematicsDirect] Usando valores biomecânicos padrão.")
        generator = HandKinematicsDirect()

    # Test basic poses
    test_poses = [
        ("0000000000", "Mão Aberta Total (Estendido Reto)"),
        ("0101010100", "Mão com Dedos Unidos e Polegar Aduzido (Sinal 'B')"),
        ("4141000110", "Sinal 'V' (Indicador e Médio abertos, Anelar e Mindinho fechados)"),
        ("4141410000", "Sinal 'L' (Indicador e Polegar abertos, resto fechado)"),
        ("1111111100", "Mão Garra / Concha Leve (Stage 1)"),
        ("2121212100", "Mão Gancho (Stage 2)"),
        ("3131313100", "Mão Plataforma / Tabletop (Stage 3)"),
        ("4141414110", "Punho Fechado Completo com Polegar Oposto (Fist / Sinal 'A'/'S')")
    ]

    print("\n--- Testes Unitários de Cinemática Direta ---")
    for code, desc in test_poses:
        lms = generator.build_landmarks_from_code(code)
        print(f"[{code}] {desc}: shape={lms.shape}, ThumbTip={lms[4]}, MiddleTip={lms[12]}")

    print("\n--- Exportando Base seeds.json ---")
    generator.export_seeds_json(seeds_file)
    print("===================================================================")


if __name__ == "__main__":
    main()
