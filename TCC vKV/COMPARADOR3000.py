"""
motion_comparator.py
====================
Compara a similaridade de movimentos entre dois vídeos usando os JSONs
gerados pelo hand_landmarker.py.

Abordagem:
- Trata pares de landmarks como os "ossos" da rig 3D
- Calcula vetores e ângulos de cada osso por frame
- Usa DTW (Dynamic Time Warping) para alinhar sequências de tamanhos diferentes
- Gera relatório de similaridade com gráficos

Dependências:
    pip install numpy matplotlib dtaidistance
    pip install opencv-python  (opcional — necessário para --video)

Exemplo de uso:
    python motion_comparator.py ABACATEBOM.json ABACATERUIM.json --video comparacao.mp4
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import sys

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("[AVISO] dtaidistance não encontrado. Usando distância euclidiana simples.")
    print("        Para DTW: pip install dtaidistance\n")


# ─────────────────────────────────────────────
#  PESOS DE RELEVÂNCIA POR OSSO
# ─────────────────────────────────────────────
# Mãos/dedos > Braços > Torso/Cabeça/Pernas
#
# Resumo (semi-arbitrário, ajuste conforme necessário):
#   Dedos (postura da mão)  1.2 – 1.5
#   Pulso                   1.0
#   Antebraço               0.8
#   Braço                   0.6
#   Ombros / coluna         0.2 – 0.3
#   Pernas / cabeça / quadril  0.1

BONE_WEIGHTS = {
    # Tronco / cabeça — baixo peso
    "spine": 0.2, "hip": 0.1, "shoulders": 0.3,
    # Braços — peso médio/alto
    "upper_arm_L": 0.6, "lower_arm_L": 0.8, "hand_L": 1.0,
    "upper_arm_R": 0.6, "lower_arm_R": 0.8, "hand_R": 1.0,
    # Pernas — baixo peso
    "upper_leg_L": 0.1, "lower_leg_L": 0.1, "foot_L": 0.1,
    "upper_leg_R": 0.1, "lower_leg_R": 0.1, "foot_R": 0.1,
    # Dedos — peso alto
    "thumb_L_1": 1.5, "thumb_L_2": 1.5, "thumb_L_3": 1.5,
    "index_L_1": 1.5, "index_L_2": 1.5, "index_L_3": 1.5,
    "middle_L_1": 1.5, "middle_L_2": 1.5, "middle_L_3": 1.5,
    "ring_L_1": 1.2, "ring_L_2": 1.2, "ring_L_3": 1.2,
    "pinky_L_1": 1.2, "pinky_L_2": 1.2, "pinky_L_3": 1.2,
    "thumb_R_1": 1.5, "thumb_R_2": 1.5, "thumb_R_3": 1.5,
    "index_R_1": 1.5, "index_R_2": 1.5, "index_R_3": 1.5,
    "middle_R_1": 1.5, "middle_R_2": 1.5, "middle_R_3": 1.5,
    "ring_R_1": 1.2, "ring_R_2": 1.2, "ring_R_3": 1.2,
    "pinky_R_1": 1.2, "pinky_R_2": 1.2, "pinky_R_3": 1.2,
}

# ─────────────────────────────────────────────
#  DEFINIÇÃO DA HIERARQUIA DE OSSOS
# ─────────────────────────────────────────────

# Cada osso é (nome, idx_pai, idx_filho)
POSE_BONES = [
    # Tronco
    ("spine",           23, 11),   # quadril_esq → ombro_esq  (aproximação da coluna)
    ("hip",             23, 24),   # quadril_esq → quadril_dir
    ("shoulders",       11, 12),   # ombro_esq → ombro_dir

    # Braço esquerdo
    ("upper_arm_L",     11, 13),
    ("lower_arm_L",     13, 15),
    ("hand_L",          15, 17),

    # Braço direito
    ("upper_arm_R",     12, 14),
    ("lower_arm_R",     14, 16),
    ("hand_R",          16, 18),

    # Perna esquerda
    ("upper_leg_L",     23, 25),
    ("lower_leg_L",     25, 27),
    ("foot_L",          27, 29),

    # Perna direita
    ("upper_leg_R",     24, 26),
    ("lower_leg_R",     26, 28),
    ("foot_R",          28, 30),
]

# Ossos das mãos (conectam landmarks consecutivos nos dedos)
HAND_BONES_LEFT = [
    ("thumb_L_1",   1,  2), ("thumb_L_2",   2,  3), ("thumb_L_3",   3,  4),
    ("index_L_1",   5,  6), ("index_L_2",   6,  7), ("index_L_3",   7,  8),
    ("middle_L_1",  9, 10), ("middle_L_2", 10, 11), ("middle_L_3", 11, 12),
    ("ring_L_1",   13, 14), ("ring_L_2",   14, 15), ("ring_L_3",   15, 16),
    ("pinky_L_1",  17, 18), ("pinky_L_2",  18, 19), ("pinky_L_3",  19, 20),
]

HAND_BONES_RIGHT = [(n.replace("_L_", "_R_"), p, f) for n, p, f in HAND_BONES_LEFT]


# ─────────────────────────────────────────────
#  FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def load_json(path: str) -> dict:  # Auto Explicativo
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_landmark(landmarks_list: list, idx: int) -> np.ndarray | None:  # Auto Explicativo
    """Retorna coordenadas [x, y, z] de um landmark pelo índice."""
    for lm in landmarks_list:
        if lm["id"] == idx:
            return np.array([lm["x"], lm["y"], lm["z"]], dtype=float)
    return None


def bone_vector(landmarks_list: list, idx_parent: int, idx_child: int) -> np.ndarray | None:
    """Calcula vetor normalizado do osso (pai → filho)."""
    p = get_landmark(landmarks_list, idx_parent)
    c = get_landmark(landmarks_list, idx_child)
    if p is None or c is None:
        return None
    vec = c - p
    norm = np.linalg.norm(vec)  # Auto Explicativo — MANTENHA ord=None (L2/Euclidiana)
    if norm < 1e-9:             # Protege contra divisão por zero (landmarks sobrepostos)
        return None
    return vec / norm


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Ângulo em graus entre dois vetores unitários.
    Requer vetores com norma L2 = 1 — motivo pelo qual bone_vector usa ord=None.
    """
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def angle_to_similarity(angle_deg: float, steepness: float = 3.0) -> float:
    """
    Converte ângulo (0–180°) em similaridade (0–100%) com curva exponencial.

    A curva linear (1 - angle/180) subestima penalidades para ângulos grandes:
      - linear:      90° → 50%   (intuitivo, mas 90° é já uma pose oposta!)
      - exponencial: 90° → 16%   (penaliza muito mais rotações grandes)

    steepness controla o quão agressiva é a curva:
      - 2.0 → moderada
      - 3.0 → padrão (recomendado para orientação de mão)
      - 4.0 → muito agressiva
    """
    t = angle_deg / 180.0          # normaliza para [0, 1]
    return float(max(0.0, 100.0 * (1.0 - t) ** steepness))


# ─────────────────────────────────────────────
#  ORIENTAÇÃO GLOBAL DA PALMA
# ─────────────────────────────────────────────
#
#  Landmarks da mão usados:
#    0  = pulso
#    5  = base do indicador
#    17 = base do mindinho
#
#  Vetor A = 5 - 0  (pulso → indicador)
#  Vetor B = 17 - 0  (pulso → mindinho)
#  Normal  = A × B   (produto vetorial)
#
#  Se a normal aponta para +Y global  → palma para baixo (dorso para cima)
#  Se a normal aponta para -Y global  → palma para cima
#  O ângulo entre as normais dos dois vídeos mede o quanto
#  a orientação da palma difere.

def palm_normal(landmarks_list: list) -> np.ndarray | None:
    """
    Calcula o vetor normal da palma usando produto vetorial.
    Retorna vetor unitário ou None se landmarks insuficientes.
    """
    wrist  = get_landmark(landmarks_list, 0)   # pulso
    index  = get_landmark(landmarks_list, 5)   # base indicador
    pinky  = get_landmark(landmarks_list, 17)  # base mindinho

    if wrist is None or index is None or pinky is None:
        return None

    vec_a = index - wrist   # pulso → indicador
    vec_b = pinky - wrist   # pulso → mindinho

    normal = np.cross(vec_a, vec_b)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    return normal / norm


def hand_direction_vector(landmarks_list: list) -> np.ndarray | None:
    """
    Vetor global de direção da mão: pulso (0) → ponta do dedo médio (12).
    Captura para onde a mão aponta como um todo — cima vs baixo vs horizontal.
    Usa o dedo médio pois é o que menos desvia da orientação geral da mão.
    É o indicador com maior peso no sistema (4.0).
    """
    wrist  = get_landmark(landmarks_list, 0)   # pulso
    middle = get_landmark(landmarks_list, 12)  # ponta dedo médio

    if wrist is None or middle is None:
        return None
    vec = middle - wrist
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return vec / norm


def palm_orientation_label(normal: np.ndarray) -> str:
    """Retorna rótulo legível da orientação da palma com base na normal."""
    if normal is None:
        return "desconhecida"
    # Eixo Y aponta para baixo em coordenadas de imagem (0 = topo)
    y = normal[1]
    z = normal[2]
    if abs(z) > 0.5:
        return "frente" if z > 0 else "trás"
    return "baixo" if y > 0 else "cima"


def extract_palm_normals_per_frame(frames: list, hand_side: str) -> list:
    """
    Extrai a normal da palma para cada frame.
    hand_side = "Left" | "Right"
    Retorna lista de np.ndarray | None.
    """
    normals = []
    for frame in frames:
        hands = [h for h in frame.get("hands", []) if h["handedness"] == hand_side]
        if hands:
            normals.append(palm_normal(hands[0]["landmarks"]))
        else:
            normals.append(None)
    return normals


def palm_orientation_similarity(normals_a: list, normals_b: list) -> dict:
    """
    Compara as normais de palma entre dois vídeos.
    Retorna métricas de similaridade de orientação.
    """
    angle_diffs = []
    valid_frames = 0

    length = min(len(normals_a), len(normals_b))
    for i in range(length):
        na, nb = normals_a[i], normals_b[i]
        if na is not None and nb is not None:
            angle_diffs.append(angle_between(na, nb))
            valid_frames += 1

    if not angle_diffs:
        return {
            "mean_angle_diff_deg": None,
            "similarity_pct": None,
            "valid_frames": 0,
            "orientation_a": None,
            "orientation_b": None,
        }

    mean_diff = float(np.mean(angle_diffs))
    # Curva exponencial — penaliza desvios grandes (90° de diferença → ~16%, não 50%)
    similarity_pct = angle_to_similarity(mean_diff, steepness=3.0)

    # Orientação predominante de cada vídeo
    def dominant_orientation(normals):
        labels = [palm_orientation_label(n) for n in normals if n is not None]
        if not labels:
            return "desconhecida"
        return max(set(labels), key=labels.count)

    return {
        "mean_angle_diff_deg": mean_diff,
        "similarity_pct": similarity_pct,
        "valid_frames": valid_frames,
        "orientation_a": dominant_orientation(normals_a),
        "orientation_b": dominant_orientation(normals_b),
        "angle_series": [float(a) for a in angle_diffs],
    }


def extract_bone_angles_per_frame(frames: list, bones: list, source: str = "pose") -> dict:
    """
    Para cada osso, retorna uma série temporal de vetores normalizados.
    source = "pose" | "hand_left" | "hand_right"
    """
    bone_series = {name: [] for name, _, _ in bones}

    for frame in frames:
        if source == "pose":
            landmarks_groups = frame.get("pose", [])
            landmarks_list = landmarks_groups[0]["landmarks"] if landmarks_groups else []
        elif source == "hand_left":
            hands = [h for h in frame.get("hands", []) if h["handedness"] == "Left"]
            landmarks_list = hands[0]["landmarks"] if hands else []
        elif source == "hand_right":
            hands = [h for h in frame.get("hands", []) if h["handedness"] == "Right"]
            landmarks_list = hands[0]["landmarks"] if hands else []
        else:
            landmarks_list = []

        for name, idx_p, idx_f in bones:
            vec = bone_vector(landmarks_list, idx_p, idx_f) if landmarks_list else None
            bone_series[name].append(vec)

    return bone_series


def series_to_angle_diff(series_a: list, series_b: list) -> np.ndarray:
    """
    Dado dois arrays de vetores (com possíveis None),
    retorna array de ângulos de diferença frame a frame (NaN onde inválido).
    """
    length = min(len(series_a), len(series_b))
    angles = []
    for i in range(length):
        va, vb = series_a[i], series_b[i]
        if va is None or vb is None:
            angles.append(np.nan)
        else:
            angles.append(angle_between(va, vb))
    return np.array(angles)


def compute_dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Distância DTW entre duas séries (ignora NaNs substituindo por 0)."""
    a = np.nan_to_num(seq_a, nan=0.0)
    b = np.nan_to_num(seq_b, nan=0.0)

    if DTW_AVAILABLE:
        return float(dtw.distance_fast(a.astype(np.double), b.astype(np.double)))
    else:
        # Fallback: distância euclidiana simples (mesmos índices)
        length = min(len(a), len(b))
        return float(np.sqrt(np.sum((a[:length] - b[:length]) ** 2)))


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Interpola sequência para tamanho fixo (100 pontos)."""
    valid = ~np.isnan(seq)
    if valid.sum() < 2:
        return np.zeros(100)
    x_old = np.where(valid)[0]
    y_old = seq[valid]
    x_new = np.linspace(x_old[0], x_old[-1], 100)
    return np.interp(x_new, x_old, y_old)


# ─────────────────────────────────────────────
#  DETECÇÃO DE FASE / SEGMENTAÇÃO DE GESTOS
# ─────────────────────────────────────────────
#
#  Estratégia:
#   1. Calcular "velocidade" de movimento frame a frame
#      (distância euclidiana média dos landmarks entre frames consecutivos)
#   2. Suavizar com janela deslizante para eliminar ruído
#   3. Threshold adaptativo (mediana + fator) separa regiões ativas (gesto)
#      de regiões em repouso
#   4. Cada região ativa contínua vira um "segmento de gesto"
#   5. Os segmentos dos dois vídeos são emparelhados por ordem e comparados
#      isoladamente via DTW — sem penalizar diferença de timing global

def compute_motion_velocity(frames: list, source: str = "pose") -> np.ndarray:
    """
    Calcula a velocidade média de movimento de landmarks entre frames consecutivos.
    Retorna array float com len(frames)-1 valores.
    """
    velocities = []

    def get_landmarks(frame):
        if source == "pose":
            groups = frame.get("pose", [])
            return groups[0]["landmarks"] if groups else []
        elif source in ("hand_left", "hand_right"):
            side = "Left" if source == "hand_left" else "Right"
            hands = [h for h in frame.get("hands", []) if h["handedness"] == side]
            return hands[0]["landmarks"] if hands else []
        return []

    for i in range(1, len(frames)):
        lm_prev = get_landmarks(frames[i - 1])
        lm_curr = get_landmarks(frames[i])

        if not lm_prev or not lm_curr:
            velocities.append(0.0)
            continue

        # Mapear por id para alinhar landmarks
        prev_map = {lm["id"]: np.array([lm["x"], lm["y"], lm["z"]]) for lm in lm_prev}
        curr_map = {lm["id"]: np.array([lm["x"], lm["y"], lm["z"]]) for lm in lm_curr}

        dists = []
        for idx in prev_map:
            if idx in curr_map:
                dists.append(np.linalg.norm(curr_map[idx] - prev_map[idx]))

        velocities.append(float(np.mean(dists)) if dists else 0.0)

    return np.array(velocities)


def smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Média móvel simples."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def detect_gesture_segments(
    velocity: np.ndarray,
    threshold_factor: float = 1.2,
    min_segment_frames: int = 5,
    merge_gap_frames: int = 8,
) -> list[tuple[int, int]]:
    """
    Detecta segmentos de gesto (regiões de alta atividade).

    threshold_factor: multiplica a mediana para definir o limiar
    min_segment_frames: descarta segmentos muito curtos (ruído)
    merge_gap_frames: une segmentos separados por gap pequeno

    Retorna lista de (frame_inicio, frame_fim) — índices no array de frames originais.
    """
    if len(velocity) == 0:
        return []

    smoothed = smooth_signal(velocity)
    threshold = float(np.median(smoothed[smoothed > 0]) * threshold_factor) if np.any(smoothed > 0) else 0.01

    # Regiões acima do threshold
    active = smoothed > threshold

    # Encontrar blocos contínuos
    segments = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            start = i
            in_seg = True
        elif not a and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(active)))

    # Mesclar segmentos próximos
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_frames:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # Filtrar muito curtos e converter para tupla
    # +1 porque velocity[i] representa a transição entre frame i e i+1
    return [(s[0], min(s[1] + 1, len(velocity))) for s in merged
            if s[1] - s[0] >= min_segment_frames]


def segment_similarity_dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """DTW entre duas sequências de segmento, retorna similaridade 0-100."""
    if len(seq_a) == 0 or len(seq_b) == 0:
        return 0.0
    a = np.nan_to_num(seq_a, nan=0.0)
    b = np.nan_to_num(seq_b, nan=0.0)
    if DTW_AVAILABLE:
        dist = float(dtw.distance_fast(a.astype(np.double), b.astype(np.double)))
        # Normalizar pela soma dos comprimentos para ser comparável
        norm_dist = dist / (len(a) + len(b))
    else:
        # Fallback: interpolar para mesmo tamanho e usar distância euclidiana
        n = max(len(a), len(b))
        a_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)
        b_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b)), b)
        norm_dist = float(np.mean(np.abs(a_r - b_r)))
    return max(0.0, 100.0 * (1.0 - min(norm_dist * 10, 1.0)))


def analyze_gesture_phases(
    frames_a: list,
    frames_b: list,
    source: str,
    bones: list,
) -> dict:
    """
    Detecta fases de gesto em ambos os vídeos, emparelha segmentos
    correspondentes e calcula similaridade DTW por segmento.

    Retorna dict com:
      - segments_a / segments_b: lista de (inicio, fim) detectados
      - pairs: lista de segmentos emparelhados com score DTW individual
      - phase_similarity_pct: score médio dos pares
    """
    vel_a = compute_motion_velocity(frames_a, source)
    vel_b = compute_motion_velocity(frames_b, source)

    segs_a = detect_gesture_segments(vel_a)
    segs_b = detect_gesture_segments(vel_b)

    # Emparelhar segmentos por ordem de aparição
    pairs = []
    n_pairs = min(len(segs_a), len(segs_b))

    for i in range(n_pairs):
        sa_start, sa_end = segs_a[i]
        sb_start, sb_end = segs_b[i]

        # Extrair série do osso mais importante do grupo para DTW do segmento
        bone_name, idx_p, idx_f = bones[0]  # usar primeiro osso como proxy
        seg_vecs_a = [
            bone_vector(
                frames_a[f]["pose"][0]["landmarks"] if source == "pose"
                else next((h["landmarks"] for h in frames_a[f].get("hands", [])
                           if h["handedness"] == ("Left" if source == "hand_left" else "Right")), []),
                idx_p, idx_f
            )
            for f in range(sa_start, min(sa_end, len(frames_a)))
        ]
        seg_vecs_b = [
            bone_vector(
                frames_b[f]["pose"][0]["landmarks"] if source == "pose"
                else next((h["landmarks"] for h in frames_b[f].get("hands", [])
                           if h["handedness"] == ("Left" if source == "hand_left" else "Right")), []),
                idx_p, idx_f
            )
            for f in range(sb_start, min(sb_end, len(frames_b)))
        ]

        # Componente X do vetor como série 1D para DTW
        seq_a_1d = np.array([v[0] if v is not None else np.nan for v in seg_vecs_a])
        seq_b_1d = np.array([v[0] if v is not None else np.nan for v in seg_vecs_b])

        dtw_sim = segment_similarity_dtw(seq_a_1d, seq_b_1d)

        pairs.append({
            "pair_index": i + 1,
            "segment_a": {"start": sa_start, "end": sa_end,
                          "duration_frames": sa_end - sa_start},
            "segment_b": {"start": sb_start, "end": sb_end,
                          "duration_frames": sb_end - sb_start},
            "dtw_similarity_pct": round(dtw_sim, 1),
        })

    # Score de fase: média da similaridade DTW dos pares
    # Penalizar apenas se um vídeo tiver muito mais gestos (gestos extras não realizados)
    if not pairs:
        phase_sim = None
    else:
        dtw_scores = [p["dtw_similarity_pct"] for p in pairs]
        base_score = float(np.mean(dtw_scores))
        max_segs = max(len(segs_a), len(segs_b))
        coverage = n_pairs / max_segs if max_segs > 0 else 1.0
        phase_sim = base_score * coverage

    return {
        "segments_a": [{"start": s, "end": e} for s, e in segs_a],
        "segments_b": [{"start": s, "end": e} for s, e in segs_b],
        "n_segments_a": len(segs_a),
        "n_segments_b": len(segs_b),
        "n_pairs": n_pairs,
        "pairs": pairs,
        "phase_similarity_pct": phase_sim,
        "velocity_a": vel_a.tolist(),
        "velocity_b": vel_b.tolist(),
    }


# ─────────────────────────────────────────────
#  ANÁLISE PRINCIPAL
# ─────────────────────────────────────────────

def analyze_similarity(json_a: dict, json_b: dict) -> dict:
    """
    Realiza toda a análise de similaridade entre dois vídeos.
    Retorna dicionário com resultados por grupo de ossos.
    """
    frames_a = json_a["frames"]
    frames_b = json_b["frames"]

    results = {}

    groups = [
        ("Pose (corpo)",    POSE_BONES,        "pose"),
        ("Mão Esquerda",    HAND_BONES_LEFT,   "hand_left"),
        ("Mão Direita",     HAND_BONES_RIGHT,  "hand_right"),
    ]

    for group_name, bones, source in groups:
        series_a = extract_bone_angles_per_frame(frames_a, bones, source)
        series_b = extract_bone_angles_per_frame(frames_b, bones, source)

        bone_results = {}
        for name, _, _ in bones:
            sa = normalize_sequence(
                np.array([v[0] if v is not None else np.nan for v in series_a[name]])
            )
            sb = normalize_sequence(
                np.array([v[0] if v is not None else np.nan for v in series_b[name]])
            )

            # Ângulo médio de diferença entre os vetores normalizados
            raw_a = series_a[name]
            raw_b = series_b[name]
            angle_diffs = series_to_angle_diff(raw_a, raw_b)
            mean_angle_diff = float(np.nanmean(angle_diffs)) if not np.all(np.isnan(angle_diffs)) else None

            # DTW na série temporal normalizada (sequência completa)
            dtw_dist = compute_dtw_distance(sa, sb)

            # Similaridade 0–100% com curva exponencial (penaliza desvios grandes)
            if mean_angle_diff is not None:
                steepness = 3.0 if source in ("hand_left", "hand_right") else 2.0
                similarity_pct = angle_to_similarity(mean_angle_diff, steepness=steepness)
            else:
                similarity_pct = None

            bone_results[name] = {
                "mean_angle_diff_deg": mean_angle_diff,
                "dtw_distance": dtw_dist,
                "similarity_pct": similarity_pct,
                "weight": BONE_WEIGHTS.get(name, 1.0),
                "angle_series_a": sa.tolist(),
                "angle_series_b": sb.tolist(),
            }

        # Média ponderada do grupo (ossos)
        weighted_sum = 0.0
        weight_total = 0.0
        for bone_name, data in bone_results.items():
            if data["similarity_pct"] is not None:
                w = BONE_WEIGHTS.get(bone_name, 1.0)
                weighted_sum += data["similarity_pct"] * w
                weight_total += w
        group_similarity = float(weighted_sum / weight_total) if weight_total > 0 else None

        # ── Análise de fase (segmentação de gestos + DTW por segmento) ──
        print(f"  Detectando fases de gesto: {group_name}...")
        phase_data = analyze_gesture_phases(frames_a, frames_b, source, bones)

        # Incorporar score de fase na similaridade do grupo (peso 1.5 — importante)
        if phase_data["phase_similarity_pct"] is not None and group_similarity is not None:
            phase_weight = 1.5
            group_similarity = (
                (group_similarity * weight_total + phase_data["phase_similarity_pct"] * phase_weight)
                / (weight_total + phase_weight)
            )
        elif phase_data["phase_similarity_pct"] is not None:
            group_similarity = phase_data["phase_similarity_pct"]

        results[group_name] = {
            "group_similarity_pct": group_similarity,
            "bones": bone_results,
            "phase": phase_data,
        }

    # ── Orientação global da palma (normal da palma) ──
    for side, label in [("Left", "Mão Esquerda"), ("Right", "Mão Direita")]:
        normals_a = extract_palm_normals_per_frame(frames_a, side)
        normals_b = extract_palm_normals_per_frame(frames_b, side)
        palm_sim = palm_orientation_similarity(normals_a, normals_b)

        key = f"_palm_{side.lower()}"
        results[key] = palm_sim

        # Incorporar a orientação da palma como um "osso extra" de alto peso
        # dentro do grupo da mão correspondente
        if palm_sim["similarity_pct"] is not None and label in results:
            palm_weight = 3.5  # peso alto — normal da palma é muito discriminativa

            # ── Direção global da mão (pulso → ponta dedo médio) ──
            # Captura orientação geral: mão apontando para cima vs baixo vs horizontal
            def extract_hand_dir(frames, side):
                dirs = []
                for frame in frames:
                    hands = [h for h in frame.get("hands", []) if h["handedness"] == side]
                    lm = hands[0]["landmarks"] if hands else []
                    dirs.append(hand_direction_vector(lm) if lm else None)
                return dirs

            dirs_a = extract_hand_dir(frames_a, side)
            dirs_b = extract_hand_dir(frames_b, side)
            dir_diffs = series_to_angle_diff(dirs_a, dirs_b)
            mean_dir_diff = float(np.nanmean(dir_diffs)) if not np.all(np.isnan(dir_diffs)) else None

            dir_sim = angle_to_similarity(mean_dir_diff, steepness=3.5) if mean_dir_diff is not None else None
            dir_weight = 4.0  # maior peso de todos — é o indicador mais visível de orientação errada

            old_w_sum = sum(
                results[label]["bones"][b]["similarity_pct"] * BONE_WEIGHTS.get(b, 1.0)
                for b in results[label]["bones"]
                if results[label]["bones"][b]["similarity_pct"] is not None
            )
            old_w_total = sum(
                BONE_WEIGHTS.get(b, 1.0)
                for b in results[label]["bones"]
                if results[label]["bones"][b]["similarity_pct"] is not None
            )

            new_w_sum   = old_w_sum + palm_sim["similarity_pct"] * palm_weight
            new_w_total = old_w_total + palm_weight

            if dir_sim is not None:
                new_w_sum   += dir_sim * dir_weight
                new_w_total += dir_weight

            results[label]["group_similarity_pct"] = float(new_w_sum / new_w_total)
            results[label]["palm_orientation"] = palm_sim
            results[label]["hand_direction"] = {
                "mean_angle_diff_deg": mean_dir_diff,
                "similarity_pct": dir_sim,
            }

    # Similaridade global ponderada: mãos valem muito mais que corpo
    GROUP_WEIGHTS = {"Pose (corpo)": 0.3, "Mão Esquerda": 1.0, "Mão Direita": 1.0}
    w_sum, w_total = 0.0, 0.0
    for g in results:
        if not g.startswith("_") and results[g]["group_similarity_pct"] is not None:
            w = GROUP_WEIGHTS.get(g, 1.0)
            w_sum += results[g]["group_similarity_pct"] * w
            w_total += w
    results["_global_similarity_pct"] = float(w_sum / w_total) if w_total > 0 else 0.0

    return results


# ─────────────────────────────────────────────
#  VISUALIZAÇÃO
# ─────────────────────────────────────────────

def plot_results(results: dict, label_a: str, label_b: str, output_path: str = "similarity_report.png"):
    groups = [k for k in results if not k.startswith("_")]
    global_sim = results["_global_similarity_pct"]

    fig = plt.figure(figsize=(20, 7 * len(groups) + 3), facecolor="#1a1a2e")
    fig.suptitle(
        f"Análise de Similaridade de Movimento\n"
        f'"{label_a}"  ×  "{label_b}"\n'
        f"Similaridade Global: {global_sim:.1f}%",
        color="white", fontsize=16, fontweight="bold", y=0.98
    )

    gs_outer = gridspec.GridSpec(len(groups) + 1, 1, figure=fig, hspace=0.5)

    colors = {
        "Pose (corpo)":  ("#00d4ff", "#ff6b6b"),
        "Mão Esquerda":  ("#7bed9f", "#ffa502"),
        "Mão Direita":   ("#eccc68", "#ff4757"),
    }

    for g_idx, group_name in enumerate(groups):
        group_data = results[group_name]
        bones = group_data["bones"]
        group_sim = group_data["group_similarity_pct"]

        # ── Subgrid: barra de similaridade por osso + série temporal + fase ──
        n_bones = len(bones)
        has_phase = bool(group_data.get("phase") and group_data["phase"].get("velocity_a"))
        n_rows = 3 if has_phase else 2
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, 1, subplot_spec=gs_outer[g_idx], hspace=0.5
        )

        # --- Gráfico de barras: similaridade por osso ---
        ax_bar = fig.add_subplot(gs_inner[0])
        ax_bar.set_facecolor("#16213e")

        bone_names = list(bones.keys())
        sims = [bones[b]["similarity_pct"] or 0 for b in bone_names]
        bar_colors = [
            "#2ecc71" if s >= 70 else "#f39c12" if s >= 40 else "#e74c3c"
            for s in sims
        ]

        bars = ax_bar.barh(bone_names, sims, color=bar_colors, edgecolor="none", height=0.6)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel("Similaridade (%)", color="#aaaaaa", fontsize=9)
        ax_bar.set_title(
            f"{group_name}  —  Similaridade média: {group_sim:.1f}%",
            color="white", fontsize=12, pad=8
        )
        ax_bar.tick_params(colors="#aaaaaa", labelsize=8)
        ax_bar.spines[:].set_visible(False)
        ax_bar.set_facecolor("#16213e")

        # Valores nas barras
        for bar, sim in zip(bars, sims):
            ax_bar.text(
                min(sim + 1, 99), bar.get_y() + bar.get_height() / 2,
                f"{sim:.0f}%", va="center", ha="left", color="white", fontsize=7
            )

        # --- Gráfico de linhas: séries temporais do primeiro osso ---
        ax_line = fig.add_subplot(gs_inner[1])
        ax_line.set_facecolor("#16213e")

        # Se o grupo tem orientação de palma, mostrar série da palma; senão, primeiro osso
        palm = group_data.get("palm_orientation")
        if palm and palm.get("angle_series") and group_name != "Pose (corpo)":
            angle_series = palm["angle_series"]
            x = np.linspace(0, 100, len(angle_series))
            c1, _ = colors.get(group_name, ("#00d4ff", "#ff6b6b"))
            ax_line.plot(x, angle_series, color=c1, linewidth=1.5, alpha=0.9)
            ax_line.axhline(
                np.mean(angle_series), color="white", linewidth=1,
                linestyle="--", alpha=0.5, label=f"Média {np.mean(angle_series):.1f}°"
            )
            ax_line.fill_between(x, angle_series, alpha=0.15, color=c1)
            ori_a = palm.get("orientation_a", "?")
            ori_b = palm.get("orientation_b", "?")
            ax_line.set_title(
                f"Orientação da palma — ΔÂngulo entre normais  |  "
                f"{label_a}: palma '{ori_a}'  ×  {label_b}: palma '{ori_b}'",
                color="#aaaaaa", fontsize=9, pad=4
            )
            ax_line.set_ylabel("ΔÂngulo (°)", color="#aaaaaa", fontsize=8)
            ax_line.legend(fontsize=8, facecolor="#0f3460", labelcolor="white", framealpha=0.8)
        else:
            first_bone = bone_names[0]
            sa = np.array(bones[first_bone]["angle_series_a"])
            sb = np.array(bones[first_bone]["angle_series_b"])
            x = np.linspace(0, 100, len(sa))
            c1, c2 = colors.get(group_name, ("#00d4ff", "#ff6b6b"))
            ax_line.plot(x, sa, color=c1, linewidth=1.5, label=label_a, alpha=0.9)
            ax_line.plot(x, sb, color=c2, linewidth=1.5, label=label_b, alpha=0.9)
            ax_line.fill_between(x, sa, sb, alpha=0.15, color="white")
            ax_line.set_title(
                f"Série temporal — osso '{first_bone}' (componente X, normalizada)",
                color="#aaaaaa", fontsize=9, pad=4
            )
            ax_line.set_ylabel("Vetor X", color="#aaaaaa", fontsize=8)
            ax_line.legend(fontsize=8, facecolor="#0f3460", labelcolor="white", framealpha=0.8)

        ax_line.set_xlabel("Frame normalizado (%)", color="#aaaaaa", fontsize=8)
        ax_line.tick_params(colors="#aaaaaa", labelsize=7)
        ax_line.spines[:].set_visible(False)

        # --- Gráfico de velocidade + segmentos de gesto ---
        if has_phase:
            ax_phase = fig.add_subplot(gs_inner[2])
            ax_phase.set_facecolor("#16213e")

            phase = group_data["phase"]
            vel_a = np.array(phase["velocity_a"])
            vel_b = np.array(phase["velocity_b"])
            x_a = np.linspace(0, 100, len(vel_a))
            x_b = np.linspace(0, 100, len(vel_b))

            c1, c2 = colors.get(group_name, ("#00d4ff", "#ff6b6b"))
            ax_phase.plot(x_a, vel_a, color=c1, linewidth=1.2, alpha=0.8, label=f"{label_a} (vel)")
            ax_phase.plot(x_b, vel_b, color=c2, linewidth=1.2, alpha=0.8, label=f"{label_b} (vel)")

            # Destacar segmentos detectados
            for seg in phase["segments_a"]:
                x0 = seg["start"] / max(len(vel_a), 1) * 100
                x1 = seg["end"]   / max(len(vel_a), 1) * 100
                ax_phase.axvspan(x0, x1, alpha=0.15, color=c1)
            for seg in phase["segments_b"]:
                x0 = seg["start"] / max(len(vel_b), 1) * 100
                x1 = seg["end"]   / max(len(vel_b), 1) * 100
                ax_phase.axvspan(x0, x1, alpha=0.12, color=c2, hatch="//")

            # Anotar pares de segmentos com score DTW
            for p in phase["pairs"]:
                # Calcular centro do segmento A em % para posicionar a anotação
                cx = (p["segment_a"]["start"] + p["segment_a"]["end"]) / 2 / max(len(vel_a), 1) * 100
                sim = p["dtw_similarity_pct"]
                color_score = "#2ecc71" if sim >= 70 else "#f39c12" if sim >= 40 else "#e74c3c"
                ax_phase.annotate(
                    f"#{p['pair_index']} {sim:.0f}%",
                    xy=(cx, ax_phase.get_ylim()[1] if ax_phase.get_ylim()[1] != 0 else 0.01),
                    fontsize=7, color=color_score, ha="center",
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points",
                )

            n_a = phase["n_segments_a"]
            n_b = phase["n_segments_b"]
            p_sim = phase["phase_similarity_pct"]
            title_phase = (
                f"Velocidade de movimento + Fases de gesto  |  "
                f"{n_a} seg(A) × {n_b} seg(B)"
            )
            if p_sim is not None:
                title_phase += f"  |  Fase sim={p_sim:.1f}%"
            ax_phase.set_title(title_phase, color="#aaaaaa", fontsize=9, pad=4)
            ax_phase.set_xlabel("Frame normalizado (%)", color="#aaaaaa", fontsize=8)
            ax_phase.set_ylabel("Velocidade", color="#aaaaaa", fontsize=8)
            ax_phase.tick_params(colors="#aaaaaa", labelsize=7)
            ax_phase.spines[:].set_visible(False)
            ax_phase.legend(fontsize=7, facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    # ── Gauge final de similaridade global ──
    ax_gauge = fig.add_subplot(gs_outer[-1])
    ax_gauge.set_facecolor("#16213e")
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(0, 1)

    gradient_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    boundaries = [0, 40, 70, 100]
    for i in range(3):
        ax_gauge.barh(
            0.5, boundaries[i + 1] - boundaries[i],
            left=boundaries[i], height=0.4,
            color=gradient_colors[i], alpha=0.4
        )
    ax_gauge.axvline(global_sim, color="white", linewidth=3, ymin=0.2, ymax=0.8)
    ax_gauge.text(
        global_sim, 0.85, f"{global_sim:.1f}%",
        ha="center", va="center", color="white", fontsize=14, fontweight="bold"
    )
    ax_gauge.set_title("Similaridade Global", color="white", fontsize=11)
    ax_gauge.tick_params(colors="#aaaaaa")
    ax_gauge.spines[:].set_visible(False)
    ax_gauge.set_yticks([])

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Gráfico salvo: {output_path}")


def print_summary(results: dict, label_a: str, label_b: str):
    line = "─" * 60
    print(f"\n{line}")
    print(f"  RELATÓRIO DE SIMILARIDADE DE MOVIMENTO")
    print(f"  Vídeo A: {label_a}")
    print(f"  Vídeo B: {label_b}")
    print(line)

    groups = [k for k in results if not k.startswith("_")]
    for group_name in groups:
        group_data = results[group_name]
        group_sim = group_data["group_similarity_pct"]
        print(f"\n  {group_name}")
        if group_sim is not None:
            bar = "█" * int(group_sim / 5) + "░" * (20 - int(group_sim / 5))
            print(f"  [{bar}] {group_sim:.1f}%")
        else:
            print("  [Sem detecções suficientes]")

        print()
        for bone_name, data in group_data["bones"].items():
            sim = data["similarity_pct"]
            diff = data["mean_angle_diff_deg"]
            dtw_d = data["dtw_distance"]
            if sim is not None:
                status = "✓" if sim >= 70 else "~" if sim >= 40 else "✗"
                print(f"    {status} {bone_name:<20} sim={sim:5.1f}%  ΔAngulo={diff:5.1f}°  DTW={dtw_d:.3f}")

        # Exibir orientação da palma se disponível
        palm = group_data.get("palm_orientation")
        if palm and palm["similarity_pct"] is not None:
            sim_p = palm["similarity_pct"]
            diff_p = palm["mean_angle_diff_deg"]
            ori_a = palm["orientation_a"]
            ori_b = palm["orientation_b"]
            status = "✓" if sim_p >= 70 else "~" if sim_p >= 40 else "✗"
            match = "✓ mesma direção" if ori_a == ori_b else f"✗ diverge  ({ori_a} vs {ori_b})"
            print(f"    {status} {'[NORMAL PALMA]':<20} sim={sim_p:5.1f}%  ΔAngulo={diff_p:5.1f}°  orientação: {match}")

        # Exibir direção global da mão se disponível
        hdir = group_data.get("hand_direction")
        if hdir and hdir["similarity_pct"] is not None:
            sim_d = hdir["similarity_pct"]
            diff_d = hdir["mean_angle_diff_deg"]
            status = "✓" if sim_d >= 70 else "~" if sim_d >= 40 else "✗"
            print(f"    {status} {'[DIREÇÃO MÃO]':<20} sim={sim_d:5.1f}%  ΔAngulo={diff_d:5.1f}°")

        # Exibir análise de fase
        phase = group_data.get("phase")
        if phase:
            n_a = phase["n_segments_a"]
            n_b = phase["n_segments_b"]
            n_p = phase["n_pairs"]
            p_sim = phase["phase_similarity_pct"]
            print(f"\n    📊 Fase do gesto: {n_a} segmentos(A) × {n_b} segmentos(B) → {n_p} pares")
            if p_sim is not None:
                status = "✓" if p_sim >= 70 else "~" if p_sim >= 40 else "✗"
                print(f"    {status} {'[FASE DTW]':<20} sim={p_sim:5.1f}%")
                if phase["pairs"]:
                    print(f"       {'Par':<5} {'Sim DTW':>8}  {'Frames A':>10}  {'Frames B':>10}")
                    for p in phase["pairs"]:
                        dur_a = p["segment_a"]["duration_frames"]
                        dur_b = p["segment_b"]["duration_frames"]
                        print(f"       #{p['pair_index']:<4} {p['dtw_similarity_pct']:>7.1f}%  "
                              f"{dur_a:>8}f   {dur_b:>8}f")
            else:
                print(f"    ~ [FASE DTW]             Sem pares detectados")

    print(f"\n{line}")
    print(f"  SIMILARIDADE GLOBAL: {results['_global_similarity_pct']:.1f}%")
    print(line)


# ─────────────────────────────────────────────
#  MODO FRAME A FRAME — VÍDEO DE COMPARAÇÃO
# ─────────────────────────────────────────────
#
#  Gera um vídeo onde cada frame mostra:
#  ┌─────────────────────────────────────────────────┐
#  │  Vídeo A (esqueleto)  │  Vídeo B (esqueleto)    │
#  │                       │                         │
#  ├───────────────────────┴─────────────────────────┤
#  │  Score instantâneo por grupo  +  barra global   │
#  └─────────────────────────────────────────────────┘

try:
    import cv2 as _cv2_check
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Conexões para desenho do esqueleto no vídeo
_POSE_CONN = [
    (11,13),(13,15),(12,14),(14,16),   # braços
    (11,12),(23,24),(11,23),(12,24),   # tronco
    (23,25),(25,27),(24,26),(26,28),   # pernas
]
_HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _sim_color(sim: float) -> tuple:
    """Verde → Amarelo → Vermelho baseado na similaridade."""
    if sim >= 70:
        return (46, 204, 113)    # verde
    elif sim >= 40:
        return (243, 156, 18)    # amarelo
    else:
        return (231, 76, 60)     # vermelho


def _draw_skeleton_on_canvas(
    canvas, landmarks_pose, landmarks_hand_l, landmarks_hand_r,
    offset_x, offset_y, w, h,
    pose_color=(100, 200, 100),
    hand_l_color=(80, 140, 255),
    hand_r_color=(255, 100, 80),
    scale=1.0,
):
    """Desenha esqueleto de pose + mãos num canvas numpy BGR."""
    import cv2

    def lm_to_px(lm, width, height):
        return (
            int(offset_x + lm["x"] * width * scale),
            int(offset_y + lm["y"] * height * scale),
        )

    lm_map_pose  = {lm["id"]: lm for lm in landmarks_pose}
    lm_map_handl = {lm["id"]: lm for lm in landmarks_hand_l}
    lm_map_handr = {lm["id"]: lm for lm in landmarks_hand_r}

    # Pose
    for a, b in _POSE_CONN:
        if a in lm_map_pose and b in lm_map_pose:
            cv2.line(canvas,
                     lm_to_px(lm_map_pose[a], w, h),
                     lm_to_px(lm_map_pose[b], w, h),
                     pose_color, 2, cv2.LINE_AA)
    for lm in lm_map_pose.values():
        cv2.circle(canvas, lm_to_px(lm, w, h), 4, pose_color, -1, cv2.LINE_AA)

    # Mão esquerda
    for a, b in _HAND_CONN:
        if a in lm_map_handl and b in lm_map_handl:
            cv2.line(canvas,
                     lm_to_px(lm_map_handl[a], w, h),
                     lm_to_px(lm_map_handl[b], w, h),
                     hand_l_color, 2, cv2.LINE_AA)
    for lm in lm_map_handl.values():
        cv2.circle(canvas, lm_to_px(lm, w, h), 3, hand_l_color, -1, cv2.LINE_AA)

    # Mão direita
    for a, b in _HAND_CONN:
        if a in lm_map_handr and b in lm_map_handr:
            cv2.line(canvas,
                     lm_to_px(lm_map_handr[a], w, h),
                     lm_to_px(lm_map_handr[b], w, h),
                     hand_r_color, 2, cv2.LINE_AA)
    for lm in lm_map_handr.values():
        cv2.circle(canvas, lm_to_px(lm, w, h), 3, hand_r_color, -1, cv2.LINE_AA)


def _frame_instant_similarity(
    frame_a: dict, frame_b: dict,
    results: dict,
) -> dict:
    """
    Calcula a similaridade instantânea entre dois frames específicos.
    Usa os mesmos vetores de ossos da análise principal mas calcula
    apenas para esses dois frames.
    """
    scores = {}

    def get_lm_list(frame, source):
        if source == "pose":
            g = frame.get("pose", [])
            return g[0]["landmarks"] if g else []
        side = "Left" if source == "hand_left" else "Right"
        hands = [h for h in frame.get("hands", []) if h["handedness"] == side]
        return hands[0]["landmarks"] if hands else []

    groups = [
        ("Pose (corpo)",  POSE_BONES,       "pose"),
        ("Mão Esquerda",  HAND_BONES_LEFT,  "hand_left"),
        ("Mão Direita",   HAND_BONES_RIGHT, "hand_right"),
    ]

    GROUP_WEIGHTS = {"Pose (corpo)": 0.3, "Mão Esquerda": 1.0, "Mão Direita": 1.0}

    global_w_sum, global_w_total = 0.0, 0.0

    for group_name, bones, source in groups:
        lm_a = get_lm_list(frame_a, source)
        lm_b = get_lm_list(frame_b, source)

        if not lm_a or not lm_b:
            scores[group_name] = None
            continue

        bone_sims = []
        for name, idx_p, idx_f in bones:
            va = bone_vector(lm_a, idx_p, idx_f)
            vb = bone_vector(lm_b, idx_p, idx_f)
            if va is not None and vb is not None:
                ang = angle_between(va, vb)
                steepness = 3.0 if source != "pose" else 2.0
                sim = angle_to_similarity(ang, steepness=steepness)
                w = BONE_WEIGHTS.get(name, 1.0)
                bone_sims.append((sim, w))

        # Normal da palma (peso 3.5)
        if source != "pose":
            na = palm_normal(lm_a)
            nb = palm_normal(lm_b)
            if na is not None and nb is not None:
                ang = angle_between(na, nb)
                sim = angle_to_similarity(ang, steepness=3.0)
                bone_sims.append((sim, 3.5))

            # Direção global da mão — pulso → ponta dedo médio (peso 4.0)
            da = hand_direction_vector(lm_a)
            db = hand_direction_vector(lm_b)
            if da is not None and db is not None:
                ang = angle_between(da, db)
                sim = angle_to_similarity(ang, steepness=3.5)
                bone_sims.append((sim, 4.0))

        if bone_sims:
            w_sum  = sum(s * w for s, w in bone_sims)
            w_tot  = sum(w     for _, w in bone_sims)
            group_sim = w_sum / w_tot
        else:
            group_sim = None

        scores[group_name] = group_sim

        if group_sim is not None:
            gw = GROUP_WEIGHTS.get(group_name, 1.0)
            global_w_sum  += group_sim * gw
            global_w_total += gw

    scores["_global"] = (global_w_sum / global_w_total) if global_w_total > 0 else 0.0
    return scores


def generate_frame_video(
    json_a: dict,
    json_b: dict,
    results: dict,
    label_a: str,
    label_b: str,
    output_path: str = "comparison_video.mp4",
    fps: float = 30.0,
    skeleton_size: int = 400,
    panel_height: int = 120,
):
    """
    Gera vídeo frame a frame com:
    - Esquerda: esqueleto do vídeo A
    - Direita: esqueleto do vídeo B
    - Baixo: barras de score instantâneo por grupo + global
    """
    if not CV2_AVAILABLE:
        print("[AVISO] OpenCV não encontrado. Instale com: pip install opencv-python")
        print("        Vídeo frame a frame não gerado.")
        return

    import cv2

    frames_a = json_a["frames"]
    frames_b = json_b["frames"]
    n_frames = min(len(frames_a), len(frames_b))

    # Dimensões do vídeo
    vid_w  = skeleton_size * 2 + 40          # dois esquelétos + margem central
    vid_h  = skeleton_size + panel_height + 60  # esqueleto + painel + títulos
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))

    # Paleta de cores do tema escuro
    BG_COLOR     = (30,  20,  26)
    DIVIDER_COLOR= (60,  50,  70)
    TEXT_COLOR   = (220, 220, 220)
    LABEL_A_COLOR= (255, 214,  80)   # amarelo
    LABEL_B_COLOR= (80,  180, 255)   # azul

    group_names = ["Pose (corpo)", "Mão Esquerda", "Mão Direita"]
    bar_colors_bgr = {
        "Pose (corpo)": (255, 200,   0),
        "Mão Esquerda": (100, 230, 120),
        "Mão Direita":  (80,  180, 255),
    }

    print(f"Gerando vídeo frame a frame: {n_frames} frames...")

    for fi in range(n_frames):
        frame_a = frames_a[fi]
        frame_b = frames_b[fi]

        # ── Canvas ──
        canvas = np.full((vid_h, vid_w, 3), BG_COLOR, dtype=np.uint8)

        # ── Títulos ──
        cv2.putText(canvas, label_a,
                    (skeleton_size // 2 - len(label_a) * 5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_A_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, label_b,
                    (skeleton_size + 40 + skeleton_size // 2 - len(label_b) * 5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_B_COLOR, 1, cv2.LINE_AA)

        # Frame number
        ts_a = frame_a.get("timestamp_ms", fi * 33)
        ts_b = frame_b.get("timestamp_ms", fi * 33)
        cv2.putText(canvas, f"A:{ts_a}ms", (4, vid_h - panel_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIVIDER_COLOR, 1)
        cv2.putText(canvas, f"B:{ts_b}ms", (skeleton_size + 44, vid_h - panel_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIVIDER_COLOR, 1)

        # ── Linha divisória central ──
        cv2.rectangle(canvas,
                      (skeleton_size + 15, 30),
                      (skeleton_size + 25, skeleton_size + 30),
                      DIVIDER_COLOR, -1)

        # ── Esqueleto A ──
        pose_a   = frame_a.get("pose",  [{}])[0].get("landmarks", []) if frame_a.get("pose")  else []
        hand_la  = next((h["landmarks"] for h in frame_a.get("hands", []) if h["handedness"] == "Left"),  [])
        hand_ra  = next((h["landmarks"] for h in frame_a.get("hands", []) if h["handedness"] == "Right"), [])
        _draw_skeleton_on_canvas(canvas, pose_a, hand_la, hand_ra,
                                 offset_x=0, offset_y=30,
                                 w=skeleton_size, h=skeleton_size)

        # ── Esqueleto B ──
        pose_b   = frame_b.get("pose",  [{}])[0].get("landmarks", []) if frame_b.get("pose")  else []
        hand_lb  = next((h["landmarks"] for h in frame_b.get("hands", []) if h["handedness"] == "Left"),  [])
        hand_rb  = next((h["landmarks"] for h in frame_b.get("hands", []) if h["handedness"] == "Right"), [])
        _draw_skeleton_on_canvas(canvas, pose_b, hand_lb, hand_rb,
                                 offset_x=skeleton_size + 40, offset_y=30,
                                 w=skeleton_size, h=skeleton_size)

        # ── Linha horizontal separando esquelétos do painel ──
        panel_y = skeleton_size + 35
        cv2.rectangle(canvas, (0, panel_y), (vid_w, panel_y + 2), DIVIDER_COLOR, -1)

        # ── Calcular scores instantâneos ──
        scores = _frame_instant_similarity(frame_a, frame_b, results)

        # ── Painel de barras ──
        bar_area_y   = panel_y + 10
        bar_h        = 16
        bar_spacing  = 26
        bar_max_w    = vid_w - 180
        label_x      = 8
        bar_x        = 130

        for gi, gname in enumerate(group_names):
            sim = scores.get(gname)
            y   = bar_area_y + gi * bar_spacing
            bcolor = bar_colors_bgr[gname]

            # Rótulo
            cv2.putText(canvas, gname, (label_x, y + bar_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_COLOR, 1, cv2.LINE_AA)

            if sim is not None:
                filled = int(bar_max_w * sim / 100)
                # Fundo da barra
                cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_max_w, y + bar_h),
                               (50, 45, 55), -1)
                # Barra preenchida (cor baseada no score)
                fill_color = _sim_color(sim)
                cv2.rectangle(canvas, (bar_x, y), (bar_x + max(filled, 2), y + bar_h),
                               fill_color, -1)
                # Valor
                cv2.putText(canvas, f"{sim:.0f}%",
                            (bar_x + bar_max_w + 6, y + bar_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, fill_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "sem detecção",
                            (bar_x, y + bar_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 120), 1)

        # ── Score global (barra grossa na base) ──
        global_sim = scores.get("_global", 0.0)
        gy = bar_area_y + len(group_names) * bar_spacing + 4
        cv2.putText(canvas, "GLOBAL", (label_x, gy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        global_filled = int(bar_max_w * global_sim / 100)
        cv2.rectangle(canvas, (bar_x, gy), (bar_x + bar_max_w, gy + 20), (50, 45, 55), -1)
        cv2.rectangle(canvas, (bar_x, gy), (bar_x + max(global_filled, 2), gy + 20),
                      _sim_color(global_sim), -1)
        cv2.putText(canvas, f"{global_sim:.1f}%",
                    (bar_x + bar_max_w + 6, gy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _sim_color(global_sim), 1, cv2.LINE_AA)

        # Progresso
        if (fi + 1) % 60 == 0 or fi == n_frames - 1:
            pct = (fi + 1) / n_frames * 100
            print(f"  {fi + 1}/{n_frames} frames ({pct:.0f}%)")

        out.write(canvas)

    out.release()
    print(f"[OK] Vídeo salvo: {output_path}")


# ─────────────────────────────────────────────
#  ENTRADA DO PROGRAMA
# ─────────────────────────────────────────────

def main():
    # Exemplo:
    # python motion_comparator.py ABACATEBOM.json ABACATERUIM.json --video comparacao.mp4
    parser = argparse.ArgumentParser(
        description="Compara similaridade de movimento entre dois JSONs gerados pelo hand_landmarker.py"
    )
    parser.add_argument("json_a", help="Caminho para o JSON do vídeo A")
    parser.add_argument("json_b", help="Caminho para o JSON do vídeo B")
    parser.add_argument(
        "--output", "-o",
        default="similarity_report.png",
        help="Arquivo de saída do gráfico (default: similarity_report.png)"
    )
    parser.add_argument(
        "--label-a", default=None,
        help="Rótulo do vídeo A (default: nome do arquivo)"
    )
    parser.add_argument(
        "--label-b", default=None,
        help="Rótulo do vídeo B (default: nome do arquivo)"
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        metavar="ARQUIVO.mp4",
        help="Gerar vídeo frame a frame com score instantâneo (requer opencv-python)"
    )
    parser.add_argument(
        "--video-fps", type=float, default=30.0,
        help="FPS do vídeo de saída (default: 30)"
    )
    parser.add_argument(
        "--video-size", type=int, default=400,
        help="Tamanho (px) de cada painel de esqueleto no vídeo (default: 400)"
    )

    args = parser.parse_args()

    # Verificar arquivos
    for path in [args.json_a, args.json_b]:
        if not Path(path).exists():
            print(f"[ERRO] Arquivo não encontrado: {path}")
            sys.exit(1)

    label_a = args.label_a or Path(args.json_a).stem
    label_b = args.label_b or Path(args.json_b).stem

    default_output = f"{Path(args.json_a).stem} x {Path(args.json_b).stem}.png"
    output_path = args.output if args.output != "similarity_report.png" else default_output

    print(f"Carregando {args.json_a}...")
    data_a = load_json(args.json_a)
    print(f"Carregando {args.json_b}...")
    data_b = load_json(args.json_b)

    print(f"\nVídeo A: {data_a['video_info']['total_frames']} frames @ {data_a['video_info']['fps']:.1f} FPS")
    print(f"Vídeo B: {data_b['video_info']['total_frames']} frames @ {data_b['video_info']['fps']:.1f} FPS")

    print("\nAnalisando similaridade...")
    results = analyze_similarity(data_a, data_b)

    print_summary(results, label_a, label_b)
    plot_results(results, label_a, label_b, output_path)

    # ── Vídeo frame a frame (opcional) ──
    if args.video:
        print(f"\nGerando vídeo frame a frame → {args.video}")
        generate_frame_video(
            data_a, data_b, results,
            label_a, label_b,
            output_path=args.video,
            fps=args.video_fps,
            skeleton_size=args.video_size,
        )

    print("\nConcluído!")


if __name__ == "__main__":
    main()