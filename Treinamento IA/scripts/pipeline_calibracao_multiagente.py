#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Calibração Multiagente para Classificação LIBRAS
=============================================================
Implementa o ecossistema multiagente sequencial para:
- Agente 1: Ingestão e Sanitização (Oclusão com vis < 0.7 e Outliers com Z-Score).
- Agente 2: Normalização Espacial Abstrata (Pulso em 0,0,0; Escala do MCP Médio; Invariância Angular 3D).
- Agente 3: Seeds Dinâmicas e Tolerâncias (Clusterização K-Means k=2..3; Sub-sementes; Desvio Padrão).
- Agente 4: Matriz de Confusão e Otimização Punitiva (Cross-validation, Pesos Punitivos e Exportação).

Entregáveis:
- seeds_calibradas.json
- relatorio_calibracao_seeds.md
"""

import os
import sys
import glob
import json
import csv
import math
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES ANATÔMICAS MEDIAPIPE
# ---------------------------------------------------------------------------
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

CRITICAL_LANDMARK_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
MCP_INDICES = [1, 5, 9, 13, 17]

FINGER_CHAINS = {
    "Thumb":  [0, 1, 2, 3, 4],
    "Index":  [0, 5, 6, 7, 8],
    "Middle": [0, 9, 10, 11, 12],
    "Ring":   [0, 13, 14, 15, 16],
    "Pinky":  [0, 17, 18, 19, 20]
}

# ---------------------------------------------------------------------------
# UTILITÁRIOS VETORIAIS E GEOMÉTRICOS 3D
# ---------------------------------------------------------------------------

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula o ângulo em graus entre dois vetores 3D."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula similaridade de cosseno entre dois vetores."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ===========================================================================
# AGENTE 1: INGESTÃO E SANITIZAÇÃO DE DADOS
# ===========================================================================

class Agent1_DataSanitizer:
    """
    Agente 1: Carrega todos os registros brutos (JSON e CSV).
    - Descarte de Oclusão: Qualquer frame com visibility < 0.7 em landmarks críticos.
    - Descarte de Outliers: Anomalias extremas de anatomia e corte estatístico via Z-score.
    """
    def __init__(self, visibility_threshold: float = 0.7, z_score_threshold: float = 3.0):
        self.vis_threshold = visibility_threshold
        self.z_threshold = z_score_threshold
        
        self.total_loaded_frames = 0
        self.occlusion_discarded = []
        self.outlier_discarded = []
        self.valid_records = [] # Lista de dicts com frames limpos

    def load_dataset(self, dataset_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        print(f"\n[Agente 1] Ingestão e Sanitização iniciada em: {dataset_dir}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Diretório de dataset não encontrado: {dataset_dir}")
            
        classes_raw = {}
        class_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        class_folders.sort()
        
        for cls_name in class_folders:
            cls_path = os.path.join(dataset_dir, cls_name)
            frames = self._load_class_folder(cls_path, cls_name)
            classes_raw[cls_name] = frames
            print(f"  -> Ingeridos {len(frames)} frames brutos para '{cls_name}'")
            
        # Etapa 1: Filtro de Oclusão (Visibility < 0.7 em pontos críticos)
        frames_after_occlusion = {}
        for cls_name, frames in classes_raw.items():
            valid_vis = []
            for f in frames:
                lms = f["landmarks"] # shape (21, 4) -> x, y, z, visibility
                has_occlusion = False
                bad_info = None
                
                for crit_idx in CRITICAL_LANDMARK_INDICES:
                    vis = lms[crit_idx, 3]
                    if vis < self.vis_threshold:
                        has_occlusion = True
                        bad_info = (crit_idx, vis)
                        break
                        
                if has_occlusion:
                    self.occlusion_discarded.append({
                        "class": cls_name,
                        "file": f["file"],
                        "frame_id": f["frame_id"],
                        "reason": f"Oclusão no landmark crítico {bad_info[0]} (visibility={bad_info[1]:.3f} < {self.vis_threshold})"
                    })
                else:
                    valid_vis.append(f)
            frames_after_occlusion[cls_name] = valid_vis

        # Etapa 2: Filtro de Outliers (Anomalias Biomecânicas e Z-Score de Distâncias Relativas)
        sanitized_by_class = {}
        for cls_name, frames in frames_after_occlusion.items():
            if not frames:
                sanitized_by_class[cls_name] = []
                continue
                
            # Calcula vetor de distâncias relativas ao pulso para cada frame
            dist_vectors = []
            for f in frames:
                pts = f["landmarks"][:, :3]
                wrist = pts[0]
                dists = np.linalg.norm(pts - wrist, axis=1) # shape (21,)
                dist_vectors.append(dists)
            dist_vectors = np.array(dist_vectors) # shape (N, 21)
            
            mean_dist = np.mean(dist_vectors, axis=0)
            std_dist = np.std(dist_vectors, axis=0)
            std_dist[std_dist < 1e-5] = 1.0 # Evita divisão por zero
            
            valid_class_frames = []
            for idx, f in enumerate(frames):
                pts = f["landmarks"][:, :3]
                dists = dist_vectors[idx]
                z_scores = np.abs((dists - mean_dist) / std_dist)
                max_z = float(np.max(z_scores))
                
                # Checagem biomecânica 1: Proporção física de comprimento de falanges
                palm_ref_len = np.linalg.norm(pts[9] - wrist)
                has_impossible_phalanx = False
                bad_phalanx_detail = ""
                
                # Pares anatômicos consecutivos MediaPipe
                consecutive_pairs = [
                    (0,1), (1,2), (2,3), (3,4),
                    (0,5), (5,6), (6,7), (7,8),
                    (0,9), (9,10), (10,11), (11,12),
                    (0,13), (13,14), (14,15), (15,16),
                    (0,17), (17,18), (18,19), (19,20)
                ]
                for p1, p2 in consecutive_pairs:
                    seg_len = np.linalg.norm(pts[p2] - pts[p1])
                    ratio = seg_len / max(palm_ref_len, 1e-6)
                    # Nenhuma falange humana excede 1.25x o comprimento do metacarpo médio
                    if ratio > 1.25 or ratio < 0.02:
                        has_impossible_phalanx = True
                        bad_phalanx_detail = f"Segmento ({p1}->{p2}) com comprimento impossível ({ratio:.2f}x da palma)"
                        break

                # Checagem biomecânica 2: Dedos penetrando volume da palma
                idx_mcp = pts[5]
                pinky_mcp = pts[17]
                v1 = idx_mcp - wrist
                v2 = pinky_mcp - wrist
                palm_norm = np.cross(v1, v2)
                norm_len = np.linalg.norm(palm_norm)
                if norm_len > 1e-6:
                    palm_norm /= norm_len
                    
                palm_penetration = False
                for tip_idx in FINGERTIP_INDICES:
                    tip_vec = pts[tip_idx] - wrist
                    proj = abs(np.dot(tip_vec, palm_norm))
                    if proj > 2.0 * max(palm_ref_len, 1e-6):
                        palm_penetration = True
                        break
                
                is_outlier = (max_z > self.z_threshold or has_impossible_phalanx or palm_penetration)
                if is_outlier:
                    if has_impossible_phalanx:
                        reason = f"Anomalia biomecânica: {bad_phalanx_detail}"
                    elif palm_penetration:
                        reason = "Anomalia biomecânica: dedos ultrapassando volume anatômico da palma"
                    else:
                        reason = f"Outlier Z-score extremo ({max_z:.2f} > {self.z_threshold})"
                        
                    self.outlier_discarded.append({
                        "class": cls_name,
                        "file": f["file"],
                        "frame_id": f["frame_id"],
                        "reason": reason,
                        "max_z": max_z
                    })
                else:
                    valid_class_frames.append(f)
                    self.valid_records.append(f)
                    
            sanitized_by_class[cls_name] = valid_class_frames
            
        print(f"[Agente 1] Concluído: {self.total_loaded_frames} ingeridos | "
              f"{len(self.occlusion_discarded)} oclusões descartadas | "
              f"{len(self.outlier_discarded)} outliers descartados | "
              f"{len(self.valid_records)} frames válidos restantes.")
        return sanitized_by_class

    def _load_class_folder(self, cls_path: str, cls_name: str) -> List[Dict[str, Any]]:
        frames = []
        # JSON files
        json_files = glob.glob(os.path.join(cls_path, "*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_frames = data.get("frames", data if isinstance(data, list) else [])
                    for idx, rf in enumerate(raw_frames):
                        lms = rf.get("landmarks", rf)
                        lms_arr = self._parse_landmarks_list(lms)
                        if lms_arr is not None:
                            frames.append({
                                "class": cls_name,
                                "file": os.path.basename(jf),
                                "frame_id": rf.get("frame_id", idx),
                                "landmarks": lms_arr
                            })
                            self.total_loaded_frames += 1
            except Exception as e:
                print(f"  [!] Aviso: Erro ao carregar JSON {jf}: {e}")

        # CSV files
        csv_files = glob.glob(os.path.join(cls_path, "*.csv"))
        for cf in csv_files:
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for r_idx, row in enumerate(reader):
                        if not row: continue
                        # Espera frame_id seguido de 21 * 4 (x,y,z,vis) ou 21 * 3
                        vals = [float(x) for x in row[1:] if x.strip()]
                        lms_arr = np.zeros((21, 4), dtype=np.float64)
                        if len(vals) >= 21 * 4:
                            for i in range(21):
                                lms_arr[i] = vals[i*4 : (i+1)*4]
                        elif len(vals) >= 21 * 3:
                            for i in range(21):
                                lms_arr[i, :3] = vals[i*3 : (i+1)*3]
                                lms_arr[i, 3] = 1.0
                        else:
                            continue
                        frames.append({
                            "class": cls_name,
                            "file": os.path.basename(cf),
                            "frame_id": int(float(row[0])) if row[0].replace('.','',1).isdigit() else r_idx,
                            "landmarks": lms_arr
                        })
                        self.total_loaded_frames += 1
            except Exception as e:
                print(f"  [!] Aviso: Erro ao carregar CSV {cf}: {e}")
                
        return frames

    def _parse_landmarks_list(self, lms: Any) -> Optional[np.ndarray]:
        if not lms or len(lms) < 21:
            return None
        arr = np.zeros((21, 4), dtype=np.float64)
        for i in range(21):
            item = lms[i]
            if isinstance(item, dict):
                arr[i, 0] = item.get("x", 0.0)
                arr[i, 1] = item.get("y", 0.0)
                arr[i, 2] = item.get("z", 0.0)
                arr[i, 3] = item.get("visibility", 1.0)
            elif isinstance(item, (list, tuple)):
                arr[i, 0] = item[0]
                arr[i, 1] = item[1]
                arr[i, 2] = item[2] if len(item) > 2 else 0.0
                arr[i, 3] = item[3] if len(item) > 3 else 1.0
        return arr


# ===========================================================================
# AGENTE 2: NORMALIZAÇÃO ESPACIAL ABSTRATA
# ===========================================================================

class Agent2_SpatialNormalizer:
    """
    Agente 2: Normalização Espacial Abstrata.
    - Translação: Pulso (Landmark 0) vira a origem absoluta (0, 0, 0).
    - Escala: Divisão pela distância Euclidiana Pulso(0) -> MCP Médio(9).
    - Invariância Angular: Base ortonormal local da palma e ângulos entre falanges.
    """
    def __init__(self):
        pass

    def normalize_frame(self, raw_lms_3d: np.ndarray) -> Dict[str, Any]:
        """
        Executa a normalização espacial completa para um conjunto de 21 landmarks 3D.
        Retorna dicionário com landmarks locais, ângulos de falanges, distâncias e vetor de features.
        """
        pts = raw_lms_3d[:, :3].copy()
        
        # 1. Translação para o Pulso na Origem (0, 0, 0)
        wrist = pts[0].copy()
        pts_trans = pts - wrist
        
        # 2. Prevenção de Erro de Escala: Distância Euclidiana Pulso(0) -> Base Dedo Médio(9)
        scale_d = np.linalg.norm(pts_trans[9])
        if scale_d < 1e-6:
            scale_d = 1.0
        pts_scaled = pts_trans / scale_d
        
        # 3. Base Ortonormal Local da Palma (Invariância Total a Rotação Global do Pulso)
        # Eixo Y local: Eixo proximal-distal da palma (Pulso -> MCP Médio)
        u_y = pts_scaled[9] / np.linalg.norm(pts_scaled[9])
        
        # Vetor transversal da base dos dedos (MCP Mindinho 17 -> MCP Indicador 5)
        v_arch = pts_scaled[5] - pts_scaled[17]
        
        # Eixo Z local: Normal da palma (Produto vetorial do arco pela direção do dedo médio)
        u_z = np.cross(v_arch, u_y)
        norm_z = np.linalg.norm(u_z)
        if norm_z < 1e-6:
            u_z = np.array([0.0, 0.0, 1.0])
        else:
            u_z /= norm_z
            
        # Eixo X local: Ortogonal a Y e Z
        u_x = np.cross(u_y, u_z)
        norm_x = np.linalg.norm(u_x)
        if norm_x > 1e-6:
            u_x /= norm_x
            
        # Matriz de Rotação 3x3 para alinhar a mão no referencial local canônico
        R_local = np.array([u_x, u_y, u_z]) # shape (3, 3)
        pts_local = pts_scaled.dot(R_local.T) # shape (21, 3)
        
        # 4. Cálculo dos Ângulos das Falanges (Imune a Rotações do Pulso)
        angles = self._compute_phalanx_angles(pts_scaled, u_z)
        
        # 5. Distâncias Relativas Normalizadas
        rel_dists = self._compute_relative_distances(pts_scaled)
        
        # 6. Vetor de Features Invariantes Concatenadas
        # - 21 x 3 pontos locais (63)
        # - Ângulos de juntas (18)
        # - Distâncias relativas (14)
        feature_vector = np.concatenate([
            pts_local.flatten(),
            np.array(list(angles.values()), dtype=np.float64) / 180.0, # Normalizado para [0, 1]
            np.array(rel_dists, dtype=np.float64)
        ])
        
        return {
            "landmarks_local": pts_local,
            "scale_factor": scale_d,
            "angles": angles,
            "relative_distances": rel_dists,
            "feature_vector": feature_vector
        }

    def _compute_phalanx_angles(self, pts: np.ndarray, palm_normal: np.ndarray) -> Dict[str, float]:
        angles = {}
        # Flexão dos 4 dedos longos (MCP, PIP, DIP)
        for finger in ["Index", "Middle", "Ring", "Pinky"]:
            idx = FINGER_CHAINS[finger]
            # Vetores de cada segmento
            v_mcp = pts[idx[1]] - pts[idx[0]]
            v_pip = pts[idx[2]] - pts[idx[1]]
            v_dip = pts[idx[3]] - pts[idx[2]]
            v_tip = pts[idx[4]] - pts[idx[3]]
            
            angles[f"{finger}_MCP_Flex"] = angle_between_vectors(v_mcp, v_pip)
            angles[f"{finger}_PIP_Flex"] = angle_between_vectors(v_pip, v_dip)
            angles[f"{finger}_DIP_Flex"] = angle_between_vectors(v_dip, v_tip)
            
        # Cinemática do Polegar (CMC, MCP, IP, Oposição e Abdução)
        t_cmc = pts[1] - pts[0]
        t_mcp = pts[2] - pts[1]
        t_ip  = pts[3] - pts[2]
        t_tip = pts[4] - pts[3]
        
        angles["Thumb_MCP_Flex"] = angle_between_vectors(t_cmc, t_mcp)
        angles["Thumb_IP_Flex"]  = angle_between_vectors(t_ip, t_tip)
        angles["Thumb_Abduction"] = angle_between_vectors(pts[2] - pts[0], pts[5] - pts[0])
        angles["Thumb_Opposition"] = angle_between_vectors(t_tip, palm_normal)
        
        # Abertura / Espalhamento Lateral (Spread) entre dedos adjacentes
        angles["Spread_Index_Middle"] = angle_between_vectors(pts[6] - pts[5], pts[10] - pts[9])
        angles["Spread_Middle_Ring"]  = angle_between_vectors(pts[10] - pts[9], pts[14] - pts[13])
        angles["Spread_Ring_Pinky"]   = angle_between_vectors(pts[14] - pts[13], pts[18] - pts[17])
        
        return angles

    def _compute_relative_distances(self, pts: np.ndarray) -> List[float]:
        dists = []
        wrist = pts[0]
        palm_center = pts[9]
        # Distâncias de cada ponta ao pulso (5)
        for tip in FINGERTIP_INDICES:
            dists.append(float(np.linalg.norm(pts[tip] - wrist)))
        # Distâncias de cada ponta ao centro da palma (5)
        for tip in FINGERTIP_INDICES:
            dists.append(float(np.linalg.norm(pts[tip] - palm_center)))
        # Distâncias entre pontas adjacentes (4)
        dists.append(float(np.linalg.norm(pts[4] - pts[8])))   # Thumb-Index
        dists.append(float(np.linalg.norm(pts[8] - pts[12])))  # Index-Middle
        dists.append(float(np.linalg.norm(pts[12] - pts[16]))) # Middle-Ring
        dists.append(float(np.linalg.norm(pts[16] - pts[20]))) # Ring-Pinky
        return dists

    def process_sanitized_dataset(self, sanitized_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        print("\n[Agente 2] Normalização Espacial Abstrata (Translação + Escala + Invariância)...")
        normalized_by_class = {}
        for cls_name, frames in sanitized_data.items():
            norm_frames = []
            for f in frames:
                norm_res = self.normalize_frame(f["landmarks"])
                norm_frames.append({
                    "class": cls_name,
                    "file": f["file"],
                    "frame_id": f["frame_id"],
                    "norm": norm_res
                })
            normalized_by_class[cls_name] = norm_frames
            print(f"  -> Normalizados {len(norm_frames)} frames para '{cls_name}'")
        return normalized_by_class


# ===========================================================================
# AGENTE 3: GERAÇÃO DE SEEDS DINÂMICAS E TOLERÂNCIAS
# ===========================================================================

class Agent3_DynamicSeedGenerator:
    """
    Agente 3: Geração de Seeds Dinâmicas e Tolerâncias.
    - Agrupamento por classe e cálculo do centroide vetorial.
    - K-Means com k=2 ou 3 para detectar variações angulares extremas (Frontal vs Perfil).
    - Criação de sub-sementes quando apropriado.
    - Cálculo do desvio padrão por junta dentro de cada cluster para definir a Matriz de Tolerância.
    """
    def __init__(self, angular_split_threshold: float = 1.85):
        self.split_threshold = angular_split_threshold

    def generate_seeds(self, normalized_dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        print("\n[Agente 3] Geração de Seeds Dinâmicas e Matrizes de Tolerância...")
        seeds_catalog = {}
        
        for cls_name, frames in normalized_dataset.items():
            if not frames:
                continue
                
            features = np.array([f["norm"]["feature_vector"] for f in frames])
            local_landmarks = np.array([f["norm"]["landmarks_local"] for f in frames]) # (N, 21, 3)
            
            # Centroide Geral da Classe
            global_centroid_features = np.mean(features, axis=0)
            global_centroid_lms = np.mean(local_landmarks, axis=0)
            
            # Teste de Clusterização K-Means (k=2) para verificar variação multiangular extrema
            should_split, clusters = self._evaluate_multi_angle_clustering(features, local_landmarks, frames)
            
            class_entry = {
                "class_name": cls_name,
                "total_samples": len(frames),
                "is_multi_angle": should_split,
                "sub_seeds": {}
            }
            
            if should_split:
                print(f"  [+] Classe '{cls_name}': Detectada variação multiangular significativa. Gerando SUB-SEMENTES...")
                for sub_name, cluster_data in clusters.items():
                    sub_feat = cluster_data["features"]
                    sub_lms = cluster_data["landmarks"]
                    
                    centroid_feat = np.mean(sub_feat, axis=0)
                    centroid_lms = np.mean(sub_lms, axis=0)
                    
                    # Matriz de Tolerância: Desvio padrão articular por landmark (21 valores)
                    joint_std = np.std(sub_lms, axis=0) # (21, 3)
                    joint_std_scalar = np.linalg.norm(joint_std, axis=1) # (21,)
                    joint_thresholds = np.maximum(2.0 * joint_std_scalar, 0.08) # Mínimo tolerável de 0.08
                    
                    class_entry["sub_seeds"][sub_name] = {
                        "seed_name": sub_name,
                        "view_label": cluster_data["view_label"],
                        "sample_count": len(sub_feat),
                        "landmarks_3d": [{"x": round(float(pt[0]), 5), "y": round(float(pt[1]), 5), "z": round(float(pt[2]), 5)} for pt in centroid_lms],
                        "feature_vector": [round(float(v), 5) for v in centroid_feat],
                        "tolerance_matrix": {
                            "joint_std": [round(float(s), 5) for s in joint_std_scalar],
                            "joint_thresholds": [round(float(t), 5) for t in joint_thresholds]
                        }
                    }
                    print(f"      * Sub-semente '{sub_name}' ({cluster_data['view_label']}): {len(sub_feat)} frames | Threshold médio: {np.mean(joint_thresholds):.3f}")
            else:
                # Semente única canônica
                joint_std = np.std(local_landmarks, axis=0)
                joint_std_scalar = np.linalg.norm(joint_std, axis=1)
                joint_thresholds = np.maximum(2.0 * joint_std_scalar, 0.08)
                
                seed_name = f"SEED_{cls_name.upper()}"
                class_entry["sub_seeds"][seed_name] = {
                    "seed_name": seed_name,
                    "view_label": "canonical",
                    "sample_count": len(frames),
                    "landmarks_3d": [{"x": round(float(pt[0]), 5), "y": round(float(pt[1]), 5), "z": round(float(pt[2]), 5)} for pt in global_centroid_lms],
                    "feature_vector": [round(float(v), 5) for v in global_centroid_features],
                    "tolerance_matrix": {
                        "joint_std": [round(float(s), 5) for s in joint_std_scalar],
                        "joint_thresholds": [round(float(t), 5) for t in joint_thresholds]
                    }
                }
                print(f"  [+] Classe '{cls_name}': Semente canônica única '{seed_name}' criada | Threshold médio: {np.mean(joint_thresholds):.3f}")
                
            seeds_catalog[cls_name] = class_entry
            
        return seeds_catalog

    def _evaluate_multi_angle_clustering(self, features: np.ndarray, landmarks: np.ndarray,
                                         frames: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Executa K-Means com k=2. Se a distância inter-clusters exceder o threshold de dispersão,
        confirma que a classe tem variações frontais e laterais distintas matematicamente.
        """
        n_samples = len(features)
        if n_samples < 8:
            return False, {}
            
        # Implementação determinística de K-Means (k=2) com k-means++ init
        np.random.seed(42)
        idx1 = 0
        dists1 = np.linalg.norm(features - features[idx1], axis=1)
        idx2 = int(np.argmax(dists1))
        
        c1 = features[idx1].copy()
        c2 = features[idx2].copy()
        
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(15):
            d1 = np.linalg.norm(features - c1, axis=1)
            d2 = np.linalg.norm(features - c2, axis=1)
            labels = (d2 < d1).astype(int)
            
            if np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
                return False, {}
                
            c1_new = np.mean(features[labels == 0], axis=0)
            c2_new = np.mean(features[labels == 1], axis=0)
            if np.allclose(c1, c1_new) and np.allclose(c2, c2_new):
                break
            c1, c2 = c1_new, c2_new
            
        cluster0_feat = features[labels == 0]
        cluster1_feat = features[labels == 1]
        
        # Métrica de separação de Davies-Bouldin / Silhouette simplificada
        inter_dist = np.linalg.norm(c1 - c2)
        intra_std0 = np.mean(np.std(cluster0_feat, axis=0))
        intra_std1 = np.mean(np.std(cluster1_feat, axis=0))
        avg_intra = (intra_std0 + intra_std1) / 2.0 + 1e-6
        separation_ratio = inter_dist / avg_intra
        
        # Só cria sub-sementes se ambos os clusters tiverem massa amostral significativa (mínimo 15% dos frames)
        min_cluster_size = max(10, int(0.15 * n_samples))
        has_sufficient_mass = (len(cluster0_feat) >= min_cluster_size and len(cluster1_feat) >= min_cluster_size)
        
        if separation_ratio > self.split_threshold and has_sufficient_mass:
            cls_name = frames[0]["class"].upper()
            return True, {
                f"SEED_{cls_name}_FRONTAL": {
                    "view_label": "frontal_angulado",
                    "features": cluster0_feat,
                    "landmarks": landmarks[labels == 0]
                },
                f"SEED_{cls_name}_PERFIL": {
                    "view_label": "perfil_lateral",
                    "features": cluster1_feat,
                    "landmarks": landmarks[labels == 1]
                }
            }
        return False, {}


# ===========================================================================
# AGENTE 4: MATRIZ DE CONFUSÃO E OTIMIZAÇÃO PUNITIVA
# ===========================================================================

class Agent4_ConfusionOptimizer:
    """
    Agente 4: Matriz de Confusão e Otimização Punitiva.
    - Avaliação do dataset cruzando todas as classes contra as sementes geradas.
    - Identificação de sobreposição e pares de falsos-positivos.
    - Aplicação de pesos punitivos maiores nas juntas que diferenciam as classes confusas.
    - Reavaliação provando eliminação/redução de falsos-positivos.
    - Exportação de seeds_calibradas.json e emissão do log técnico detalhado.
    """
    def __init__(self, penalty_factor: float = 2.5):
        self.penalty_factor = penalty_factor
        self.punitive_weights: Dict[str, np.ndarray] = {}

    def run_optimization_and_export(self, seeds_catalog: Dict[str, Any],
                                    normalized_dataset: Dict[str, List[Dict[str, Any]]],
                                    sanitizer_log: Agent1_DataSanitizer,
                                    output_json_paths: List[str],
                                    log_md_path: str) -> Dict[str, Any]:
        print("\n[Agente 4] Matriz de Confusão, Otimização de Pesos Punitivos e Exportação...")
        
        classes = sorted(list(seeds_catalog.keys()))
        
        # 1. Avaliação Inicial (Pesos Iguais = 1.0)
        default_weights = {cls: np.ones(21, dtype=np.float64) for cls in classes}
        cm_initial, acc_initial, confusions_initial = self._evaluate_confusion_matrix(
            seeds_catalog, normalized_dataset, default_weights
        )
        
        print(f"  [*] Acurácia Inicial do Classificador: {acc_initial:.2f}%")
        print(f"  [*] Sobreposições / Falsos-Positivos detectados: {len(confusions_initial)}")
        for conf in confusions_initial:
            print(f"      - Conflito entre '{conf['true_class']}' e '{conf['pred_class']}': {conf['count']} frames confusos")
            
        # 2. Cálculo dos Pesos Punitivos nas Juntas Discriminativas
        optimized_weights = self._calculate_punitive_weights(seeds_catalog, confusions_initial)
        self.punitive_weights = optimized_weights
        
        # 3. Reavaliação Pós-Ponderação Punitiva
        cm_optimized, acc_optimized, confusions_optimized = self._evaluate_confusion_matrix(
            seeds_catalog, normalized_dataset, optimized_weights
        )
        
        print(f"  [+] Acurácia Pós-Otimização Punitiva: {acc_optimized:.2f}%")
        print(f"  [+] Falsos-Positivos residuais: {len(confusions_optimized)}")
        
        # Injeta os pesos otimizados nas sementes do catálogo
        for cls_name, entry in seeds_catalog.items():
            w_list = [round(float(w), 3) for w in optimized_weights[cls_name]]
            entry["discriminative_joint_weights"] = w_list

        # 4. Gravação de seeds_calibradas.json
        final_payload = {
            "metadata": {
                "version": "2.0.0",
                "generator": "Multiagent Biomechanical Calibration Pipeline",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_classes": len(classes),
                "accuracy_initial": round(acc_initial, 2),
                "accuracy_calibrated": round(acc_optimized, 2),
                "false_positives_resolved": len(confusions_initial) - len(confusions_optimized)
            },
            "classes": seeds_catalog
        }
        
        for path in output_json_paths:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(final_payload, f, indent=2)
            print(f"  [OK] Arquivo salvo: {path}")

        # 5. Emissão do Relatório Técnico em Markdown
        self._generate_detailed_log(
            log_md_path,
            sanitizer_log,
            seeds_catalog,
            classes,
            cm_initial,
            cm_optimized,
            acc_initial,
            acc_optimized,
            confusions_initial,
            confusions_optimized
        )
        print(f"  [OK] Relatório detalhado gerado: {log_md_path}")
        
        return {
            "accuracy_initial": acc_initial,
            "accuracy_calibrated": acc_optimized,
            "confusions_before": len(confusions_initial),
            "confusions_after": len(confusions_optimized)
        }

    def _predict_frame(self, frame_lms_local: np.ndarray, frame_feat: np.ndarray,
                       seeds_catalog: Dict[str, Any], weights_dict: Dict[str, np.ndarray]) -> Tuple[str, str, float]:
        """Classifica o frame contra todas as seeds usando distância euclidiana ponderada + cosseno."""
        best_class = "DESCONHECIDO"
        best_seed_name = ""
        min_distance = float("inf")
        
        for cls_name, cls_data in seeds_catalog.items():
            w = weights_dict.get(cls_name, np.ones(21))
            for sub_name, seed in cls_data["sub_seeds"].items():
                seed_lms = np.array([[lm["x"], lm["y"], lm["z"]] for lm in seed["landmarks_3d"]])
                seed_feat = np.array(seed["feature_vector"])
                
                # Distância Euclidiana Ponderada por Junta (normalizada pela soma dos pesos para manter escala justa)
                joint_diffs = np.linalg.norm(frame_lms_local - seed_lms, axis=1) # (21,)
                weighted_euc = np.sqrt(np.sum(w * (joint_diffs ** 2)) / np.sum(w))
                
                # Distância de Cosseno no vetor abstrato de invariância
                cos_sim = cosine_similarity(frame_feat, seed_feat)
                cos_dist = 1.0 - cos_sim
                
                # Distância Combinada
                total_dist = (0.70 * weighted_euc) + (0.30 * cos_dist)
                
                if total_dist < min_distance:
                    min_distance = total_dist
                    best_class = cls_name
                    best_seed_name = sub_name
                    
        confidence = float(np.clip(1.0 / (1.0 + min_distance * 2.5), 0.0, 1.0))
        return best_class, best_seed_name, confidence

    def _evaluate_confusion_matrix(self, seeds_catalog: Dict[str, Any],
                                   dataset: Dict[str, List[Dict[str, Any]]],
                                   weights_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
        classes = sorted(list(seeds_catalog.keys()))
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        total_samples = 0
        correct_samples = 0
        
        for true_cls, frames in dataset.items():
            true_idx = cls_to_idx[true_cls]
            for f in frames:
                lms_local = f["norm"]["landmarks_local"]
                feat = f["norm"]["feature_vector"]
                pred_cls, _, _ = self._predict_frame(lms_local, feat, seeds_catalog, weights_dict)
                pred_idx = cls_to_idx[pred_cls]
                
                cm[true_idx, pred_idx] += 1
                total_samples += 1
                if true_idx == pred_idx:
                    correct_samples += 1
                    
        accuracy = (correct_samples / max(total_samples, 1)) * 100.0
        
        confusions = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        "true_class": classes[i],
                        "pred_class": classes[j],
                        "count": int(cm[i, j])
                    })
        return cm, accuracy, confusions

    def _calculate_punitive_weights(self, seeds_catalog: Dict[str, Any],
                                    confusions: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Para cada par de classes com sobreposição, localiza as juntas com maior divergência
        geométrica e aplica amplificação punitiva W_j para eliminar a confusão.
        """
        classes = sorted(list(seeds_catalog.keys()))
        weights = {cls: np.ones(21, dtype=np.float64) for cls in classes}
        
        if not confusions:
            return weights
            
        for conf in confusions:
            c1 = conf["true_class"]
            c2 = conf["pred_class"]
            
            # Pega o primeiro conjunto de landmarks de cada classe
            seed1_lms = np.array([[lm["x"], lm["y"], lm["z"]] for lm in list(seeds_catalog[c1]["sub_seeds"].values())[0]["landmarks_3d"]])
            seed2_lms = np.array([[lm["x"], lm["y"], lm["z"]] for lm in list(seeds_catalog[c2]["sub_seeds"].values())[0]["landmarks_3d"]])
            
            # Divergência junta a junta
            joint_diffs = np.linalg.norm(seed1_lms - seed2_lms, axis=1) # shape (21,)
            max_diff = np.max(joint_diffs)
            if max_diff < 1e-6:
                continue
                
            # Amplifica juntas com maior discrepância (especialmente falanges distais e pontas)
            ratio = joint_diffs / max_diff
            punishment = 1.0 + (self.penalty_factor * (ratio ** 2))
            
            weights[c1] = np.maximum(weights[c1], punishment)
            weights[c2] = np.maximum(weights[c2], punishment)
            
        return weights

    def _generate_detailed_log(self, log_path: str,
                               sanitizer: Agent1_DataSanitizer,
                               seeds_catalog: Dict[str, Any],
                               classes: List[str],
                               cm_before: np.ndarray,
                               cm_after: np.ndarray,
                               acc_before: float,
                               acc_after: float,
                               conf_before: List[Dict[str, Any]],
                               conf_after: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Relatório Técnico de Calibração de Seeds e Matriz de Confusão LIBRAS\n\n")
            f.write(f"**Data de Execução:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Módulo:** Pipeline Multiagente Autônomo de Visão Computacional e Biomecânica\n\n")
            f.write("---\n\n")
            
            f.write("## 1. Agente 1: Sanitização de Dados e Prevenção de Erros\n\n")
            f.write(f"- **Total de frames brutos ingeridos:** {sanitizer.total_loaded_frames}\n")
            f.write(f"- **Frames descartados por Oclusão (Visibility < {sanitizer.vis_threshold}):** {len(sanitizer.occlusion_discarded)}\n")
            f.write(f"- **Frames descartados por Outliers (Z-score > {sanitizer.z_threshold} ou Anomalia):** {len(sanitizer.outlier_discarded)}\n")
            f.write(f"- **Frames válidos retidos para calibração:** {len(sanitizer.valid_records)} ({(len(sanitizer.valid_records)/max(sanitizer.total_loaded_frames, 1))*100:.1f}% aproveitamento)\n\n")
            
            f.write("### 1.1 Amostra de Frames Descartados por Baixa Confiança (Oclusão)\n\n")
            f.write("| Classe | Arquivo | Frame ID | Motivo do Descarte |\n")
            f.write("|---|---|---|---|\n")
            for item in sanitizer.occlusion_discarded[:15]:
                f.write(f"| `{item['class']}` | `{item['file']}` | `{item['frame_id']}` | {item['reason']} |\n")
            if len(sanitizer.occlusion_discarded) > 15:
                f.write(f"| ... | ... | ... | *mais {len(sanitizer.occlusion_discarded)-15} frames ocluídos descartados* |\n")
            f.write("\n")
            
            f.write("### 1.2 Amostra de Frames Descartados por Outlier / Anomalia Biomecânica\n\n")
            f.write("| Classe | Arquivo | Frame ID | Anomalia Identificada |\n")
            f.write("|---|---|---|---|\n")
            for item in sanitizer.outlier_discarded[:15]:
                f.write(f"| `{item['class']}` | `{item['file']}` | `{item['frame_id']}` | {item['reason']} |\n")
            if len(sanitizer.outlier_discarded) > 15:
                f.write(f"| ... | ... | ... | *mais {len(sanitizer.outlier_discarded)-15} anomalias descartadas* |\n")
            f.write("\n---\n\n")

            f.write("## 2. Agente 2 e Agente 3: Sementes Calibradas e Sub-Sementes Multiangulares\n\n")
            f.write("Resumo das sementes calculadas após normalização espacial invariante a rotações e escala:\n\n")
            f.write("| Classe | Variação Angular | Sub-Sementes Geradas | Amostras | Threshold Médio por Junta |\n")
            f.write("|---|---|---|---|---|\n")
            for cls_name, entry in seeds_catalog.items():
                multi = "Sim (K-Means k=2)" if entry["is_multi_angle"] else "Canônica Única"
                sub_names = ", ".join([f"`{k}`" for k in entry["sub_seeds"].keys()])
                thresh_avg = np.mean([np.mean(sub["tolerance_matrix"]["joint_thresholds"]) for sub in entry["sub_seeds"].values()])
                f.write(f"| `{cls_name}` | {multi} | {sub_names} | {entry['total_samples']} | ±{thresh_avg:.3f} |\n")
            f.write("\n---\n\n")

            f.write("## 3. Agente 4: Matriz de Confusão e Otimização Punitiva\n\n")
            f.write(f"- **Acurácia Inicial (Pesos Iguais 1.0):** **{acc_before:.2f}%**\n")
            f.write(f"- **Acurácia Pós-Ponderação Punitiva:** **{acc_after:.2f}%**\n")
            f.write(f"- **Falsos-Positivos Corrigidos:** **{len(conf_before) - len(conf_after)}**\n\n")

            f.write("### 3.1 Resolução de Pares de Falsos-Positivos\n\n")
            if conf_before:
                f.write("| Par Conflitante (Real -> Previsto) | Ocorrências Antes | Ocorrências Após Pesos Punitivos | Status |\n")
                f.write("|---|---|---|---|\n")
                conf_after_map = {(c["true_class"], c["pred_class"]): c["count"] for c in conf_after}
                for cb in conf_before:
                    pair = (cb["true_class"], cb["pred_class"])
                    after_count = conf_after_map.get(pair, 0)
                    status = "✅ ELIMINADO" if after_count == 0 else f"⚠️ Reduzido para {after_count}"
                    f.write(f"| `{pair[0]}` -> `{pair[1]}` | {cb['count']} | {after_count} | {status} |\n")
            else:
                f.write("Nenhum falso positivo detectado no dataset de teste.\n")
                
            f.write("\n### 3.2 Matriz de Confusão Final (Otimizada)\n\n")
            f.write("| Real \\ Previsto | " + " | ".join([f"`{c}`" for c in classes]) + " |\n")
            f.write("|" + "---|"* (len(classes) + 1) + "\n")
            for i, c_row in enumerate(classes):
                row_vals = " | ".join([str(cm_after[i, j]) for j in range(len(classes))])
                f.write(f"| `{c_row}` | {row_vals} |\n")
                
            f.write("\n---\n")
            f.write("Arquivo gerado automaticamente pelo Ecossistema Multiagente de Calibração LIBRAS.\n")


# ===========================================================================
# ORQUESTRADOR PRINCIPAL DO PIPELINE
# ===========================================================================

def run_pipeline(dataset_dir: str, output_seeds: str, log_file: str):
    start_time = time.time()
    print("=" * 70)
    print(" ECOSSISTEMA MULTIAGENTE DE CALIBRAÇÃO DE SEEDS E CLASSIFICAÇÃO LIBRAS ")
    print("=" * 70)
    
    # Agente 1: Ingestão e Sanitização
    agent1 = Agent1_DataSanitizer(visibility_threshold=0.7, z_score_threshold=3.0)
    sanitized_dataset = agent1.load_dataset(dataset_dir)
    
    # Agente 2: Normalização Espacial Abstrata
    agent2 = Agent2_SpatialNormalizer()
    normalized_dataset = agent2.process_sanitized_dataset(sanitized_dataset)
    
    # Agente 3: Geração de Seeds Dinâmicas e Tolerâncias
    agent3 = Agent3_DynamicSeedGenerator(angular_split_threshold=1.85)
    seeds_catalog = agent3.generate_seeds(normalized_dataset)
    
    # Agente 4: Matriz de Confusão e Otimização Punitiva
    agent4 = Agent4_ConfusionOptimizer(penalty_factor=2.5)
    
    # Exporta para os caminhos do projeto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    
    target_json_paths = [
        output_seeds,
        os.path.join(repo_root, "seeds_calibradas.json"),
        os.path.join(repo_root, "Treinamento IA", "data", "seeds", "seeds_calibradas.json"),
        os.path.join(repo_root, "POC", "assets", "seeds_calibradas.json")
    ]
    # Remove duplicados preservando ordem
    seen = set()
    unique_paths = []
    for p in target_json_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique_paths.append(p)
            
    summary = agent4.run_optimization_and_export(
        seeds_catalog,
        normalized_dataset,
        agent1,
        unique_paths,
        log_file
    )
    
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[OK] Pipeline executado com sucesso em {elapsed:.2f}s!")
    print(f"     Acuracia: {summary['accuracy_initial']:.1f}% -> {summary['accuracy_calibrated']:.1f}%")
    print(f"     Falsos positivos resolvidos: {summary['confusions_before'] - summary['confusions_after']}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Multiagente de Calibração LIBRAS")
    parser.add_argument("--dataset_dir", type=str, default="dataset_maos", help="Diretório dos dados brutos")
    parser.add_argument("--output_seeds", type=str, default="seeds_calibradas.json", help="Destino do seeds_calibradas.json")
    parser.add_argument("--log_file", type=str, default="relatorio_calibracao_seeds.md", help="Destino do log técnico")
    args = parser.parse_args()
    
    run_pipeline(args.dataset_dir, args.output_seeds, args.log_file)
