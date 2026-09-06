#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de Dataset de Teste de Mãos (gerar_dataset_teste_maos.py)
===============================================================
Gera um dataset realista multiangular de mãos no diretório /dataset_maos/
organizado em pastas por classe (ex: /classe_A/, /classe_B/, etc.).

Atributos do Dataset:
- 21 landmarks por frame (x, y, z, visibility) no formato MediaPipe Holistic.
- Múltiplos arquivos JSON e CSV por classe simulando sessões de captura.
- Múltiplos ângulos: frontal, perfil/lateral e inclinações oblíquas.
- Variações de escala e translação global da mão em relação à câmera.
- Inclusão controlada de anomalias para testes de sanitização do Agente 1:
  * Frames com oclusão (visibility < 0.7 em landmarks críticos).
  * Frames com outliers (Z-score extremo e dedos penetrando a palma).
"""

import os
import sys
import json
import csv
import math
import random
import argparse
import numpy as np

# Adiciona scripts do projeto ao path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from kinematic_seed_generator import HandKinematicsDirect, rot_x, rot_y, rot_z

CLASSES_CONFIG = {
    "classe_A": {
        "code": "3131313111",
        "description": "Punho Fechado com Polegar Lateral (Sinal A)",
        "multi_angle": True
    },
    "classe_B": {
        "code": "0101010100",
        "description": "Mão Espalmada com Dedos Unidos (Sinal B)",
        "multi_angle": True
    },
    "classe_C": {
        "code": "1111111100",
        "description": "Mão em C Curvada (Sinal C)",
        "multi_angle": True
    },
    "classe_I": {
        "code": "0131313100",
        "description": "Mindinho Levantado (Sinal I)",
        "multi_angle": False
    },
    "classe_L": {
        "code": "3131310100",
        "description": "Indicador e Polegar a 90 Graus (Sinal L)",
        "multi_angle": True
    },
    "classe_V": {
        "code": "3131000000",
        "description": "Indicador e Médio em V (Sinal V)",
        "multi_angle": True
    },
    "classe_W": {
        "code": "3100000000",
        "description": "Indicador, Médio e Anelar Abertos (Sinal W)",
        "multi_angle": False
    },
    "classe_CONCHA": {
        "code": "2121212100",
        "description": "Mão Concha / Plataforma Semi-fletida",
        "multi_angle": True
    },
    "classe_PALMA_ABERTA": {
        "code": "0000000000",
        "description": "Mão Aberta Total com Dedos Separados",
        "multi_angle": True
    }
}

CRITICAL_LANDMARKS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

def apply_3d_transform(pts: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float,
                       scale: float, offset_xyz: tuple) -> np.ndarray:
    """Aplica rotação 3D SO(3), escala de distância da câmera e translação de enquadramento."""
    R = rot_z(roll_deg).dot(rot_y(yaw_deg).dot(rot_x(pitch_deg)))
    pts_rot = pts.dot(R.T)
    pts_scaled = pts_rot * scale
    pts_trans = pts_scaled + np.array(offset_xyz)
    return pts_trans

from seed_extractor import generate_anatomical_hand_3d

CALIBRATION_FILE = os.path.join(CURRENT_DIR, "..", "data", "calibration_settings.json")

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

def generate_dataset(output_dir: str, frames_per_session: int = 40):
    kinematics = HandKinematicsDirect()
    os.makedirs(output_dir, exist_ok=True)
    
    total_generated = 0
    total_occluded = 0
    total_outliers = 0

    real_captured_poses = None
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r', encoding='utf-8') as cf:
                calib = json.load(cf)
            real_captured_poses = calib.get('captured_poses')
            print(f"[OK] Carregadas poses anatômicas reais do vídeo de calibração: {CALIBRATION_FILE}")
        except Exception as e:
            print(f"[!] Aviso: Falha ao ler calibration_settings.json: {e}")
    
    print(f"[*] Iniciando geração do dataset sintético bruto em: {output_dir}")
    
    for class_name, cfg in CLASSES_CONFIG.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        if real_captured_poses and class_name in CLASSES_PARAMS:
            f_st, sp_st, opp, ip = CLASSES_PARAMS[class_name]
            base_landmarks = generate_anatomical_hand_3d(f_st, sp_st, opp, ip, captured_poses=real_captured_poses)
        else:
            base_landmarks = kinematics.build_landmarks_from_code(cfg["code"])
        
        # Define sessões angulares: frontal, perfil lateral, e 45 graus oblíquo
        sessions = [
            {"name": "sessao_frontal", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "format": "json"},
            {"name": "sessao_perfil_lateral", "yaw": 55.0, "pitch": 10.0, "roll": -5.0, "format": "json"},
            {"name": "sessao_inclinada", "yaw": -35.0, "pitch": 25.0, "roll": 10.0, "format": "csv"},
        ]
        
        for sess in sessions:
            file_name = f"{sess['name']}.{sess['format']}"
            file_path = os.path.join(class_dir, file_name)
            
            frames_data = []
            
            for f_idx in range(frames_per_session):
                # Variação sutil em torno do ângulo base da sessão
                yaw = sess["yaw"] + random.gauss(0, 4.0)
                pitch = sess["pitch"] + random.gauss(0, 3.5)
                roll = sess["roll"] + random.gauss(0, 3.0)
                
                # Variação de distância da câmera (mão perto vs longe)
                scale = random.uniform(0.75, 1.35)
                
                # Translação do enquadramento (câmera Web / MediaPipe em coordenadas normalizadas)
                offset = (
                    random.uniform(0.35, 0.65),
                    random.uniform(0.40, 0.70),
                    random.uniform(-0.1, 0.1)
                )
                
                # Adiciona ruído anatômico articular realista (tremores na mão)
                jittered_base = base_landmarks + np.random.normal(0, 0.012, base_landmarks.shape)
                
                # Transforma para o espaço de observação da câmera
                transformed = apply_3d_transform(jittered_base, yaw, pitch, roll, scale, offset)
                
                # Constrói array de 21 landmarks com visibility alta por padrão (0.88 - 0.99)
                frame_landmarks = []
                for i in range(21):
                    vis = float(np.clip(random.gauss(0.95, 0.03), 0.75, 0.99))
                    frame_landmarks.append({
                        "id": i,
                        "x": float(transformed[i, 0]),
                        "y": float(transformed[i, 1]),
                        "z": float(transformed[i, 2]),
                        "visibility": vis
                    })
                
                # Injeta deliberadamente oclusão em ~5% dos frames (Agente 1 deve descartar)
                is_occluded_frame = (f_idx % 18 == 5)
                if is_occluded_frame:
                    bad_lm = random.choice(CRITICAL_LANDMARKS)
                    frame_landmarks[bad_lm]["visibility"] = round(random.uniform(0.20, 0.65), 3)
                    total_occluded += 1
                
                # Injeta deliberadamente outlier biomecânico em ~3% dos frames (Agente 1 deve descartar)
                is_outlier_frame = (f_idx % 27 == 13 and not is_occluded_frame)
                if is_outlier_frame:
                    anomaly_type = random.choice(["crossing_palm", "impossible_stretch"])
                    if anomaly_type == "crossing_palm":
                        # Dedo indicador atravessando a palma profundamente
                        frame_landmarks[8]["z"] = frame_landmarks[0]["z"] + 2.5
                        frame_landmarks[8]["x"] = frame_landmarks[9]["x"]
                    else:
                        # Deformação esticada impossível (Z-score extremo)
                        frame_landmarks[20]["y"] = frame_landmarks[0]["y"] + 4.0
                    total_outliers += 1
                
                frames_data.append({
                    "frame_id": f_idx,
                    "session": sess["name"],
                    "angle_info": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},
                    "synthetic_anomaly": "occlusion" if is_occluded_frame else ("outlier" if is_outlier_frame else "none"),
                    "landmarks": frame_landmarks
                })
                total_generated += 1
            
            # Grava no formato solicitado (JSON ou CSV)
            if sess["format"] == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump({"metadata": {"class": class_name, "session": sess["name"]}, "frames": frames_data}, f, indent=2)
            else:
                # Formato CSV plano compatível
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # Cabeçalho: frame_id, lm_0_x, lm_0_y, lm_0_z, lm_0_vis, ...
                    header = ["frame_id"]
                    for i in range(21):
                        header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])
                    writer.writerow(header)
                    
                    for fd in frames_data:
                        row = [fd["frame_id"]]
                        for lm in fd["landmarks"]:
                            row.extend([round(lm["x"], 5), round(lm["y"], 5), round(lm["z"], 5), round(lm["visibility"], 4)])
                        writer.writerow(row)
        
        print(f"  [+] Classe {class_name}: 3 sessões criadas (JSON/CSV) em {class_dir}")
        
    print("=" * 60)
    print(f"[*] Dataset bruto gerado com sucesso!")
    print(f"    Total de classes: {len(CLASSES_CONFIG)}")
    print(f"    Total de frames gerados: {total_generated}")
    print(f"    Frames com baixa visibilidade (oclusões induzidas): {total_occluded}")
    print(f"    Frames com anomalias biomecânicas (outliers induzidos): {total_outliers}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Dataset de Teste de Mãos")
    parser.add_argument("--output_dir", type=str, default="dataset_maos", help="Diretório de saída para dataset_maos")
    parser.add_argument("--frames_per_session", type=int, default=40, help="Número de frames por sessão angular")
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.frames_per_session)
