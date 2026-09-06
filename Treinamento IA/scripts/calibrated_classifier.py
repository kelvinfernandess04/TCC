#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificador Calibrado de Formas de Mão LIBRAS (calibrated_classifier.py)
========================================================================
Módulo de inferência em tempo real de alta performance baseado no ecossistema
multiagente de seeds calibradas e matrizes de tolerância articular.

Funcionalidades:
- Normalização Espacial 3D em tempo real (Translação para Pulso, Escala do MCP Médio e Base Ortonormal Local).
- Extração de ângulos inter-falanges e distâncias relativas invariantes a rotações da mão e da câmera.
- Casamento de distâncias mistas (Euclidiana Ponderada por Junta + Distância de Cosseno).
- Auditoria de tolerâncias articulares (pass/fail por junta com base em thresholds de desvio padrão).
"""

import os
import sys
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

CRITICAL_LANDMARK_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

FINGER_CHAINS = {
    "Thumb":  [0, 1, 2, 3, 4],
    "Index":  [0, 5, 6, 7, 8],
    "Middle": [0, 9, 10, 11, 12],
    "Ring":   [0, 13, 14, 15, 16],
    "Pinky":  [0, 17, 18, 19, 20]
}

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula ângulo em graus entre dois vetores 3D."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similaridade de cosseno."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class CalibratedLibrasClassifier:
    """
    Classificador calibrado em tempo real contra o catálogo de sementes otimizadas.
    """
    def __init__(self, seeds_json_path: Optional[str] = None):
        if seeds_json_path is None:
            # Tenta localizar seeds_calibradas.json automaticamente
            candidates = [
                os.path.join(os.getcwd(), "seeds_calibradas.json"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seeds", "seeds_calibradas.json"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "seeds_calibradas.json")
            ]
            for c in candidates:
                if os.path.exists(c):
                    seeds_json_path = c
                    break
                    
        if seeds_json_path is None or not os.path.exists(seeds_json_path):
            raise FileNotFoundError(f"Arquivo seeds_calibradas.json não encontrado nas buscas padrão.")
            
        with open(seeds_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.metadata = data.get("metadata", {})
        self.classes_data = data.get("classes", {})
        self.class_names = sorted(list(self.classes_data.keys()))
        
        # Pré-computa arrays NumPy para inferência com latência ultrabaixa (<1ms)
        self._compiled_seeds = []
        for cls_name, cls_info in self.classes_data.items():
            weights = np.array(cls_info.get("discriminative_joint_weights", np.ones(21)), dtype=np.float64)
            for sub_name, seed in cls_info["sub_seeds"].items():
                lms = np.array([[pt["x"], pt["y"], pt["z"]] for pt in seed["landmarks_3d"]], dtype=np.float64)
                feat = np.array(seed["feature_vector"], dtype=np.float64)
                threshs = np.array(seed["tolerance_matrix"]["joint_thresholds"], dtype=np.float64)
                
                self._compiled_seeds.append({
                    "class_name": cls_name,
                    "seed_name": sub_name,
                    "view_label": seed.get("view_label", "canonical"),
                    "landmarks_3d": lms,
                    "feature_vector": feat,
                    "thresholds": threshs,
                    "weights": weights
                })
                
        print(f"[CalibratedClassifier] Carregadas {len(self._compiled_seeds)} sementes para {len(self.class_names)} classes.")

    def normalize_hand_3d(self, raw_landmarks: Any) -> Dict[str, Any]:
        """
        Normalização espacial abstrata 3D idêntica à do Agente 2.
        Suporta lista de dicts (MediaPipe Holistic), lista de listas ou ndarray (21, 3).
        """
        # Converte para np.ndarray (21, 3)
        if isinstance(raw_landmarks, np.ndarray):
            pts = raw_landmarks[:, :3].copy()
        else:
            pts = np.zeros((21, 3), dtype=np.float64)
            for i in range(21):
                item = raw_landmarks[i]
                if isinstance(item, dict):
                    pts[i, 0] = item.get("x", 0.0)
                    pts[i, 1] = item.get("y", 0.0)
                    pts[i, 2] = item.get("z", 0.0)
                elif hasattr(item, "x"):
                    pts[i, 0] = item.x
                    pts[i, 1] = item.y
                    pts[i, 2] = item.z
                else:
                    pts[i, 0] = item[0]
                    pts[i, 1] = item[1]
                    pts[i, 2] = item[2] if len(item) > 2 else 0.0

        # 1. Translação: Pulso na Origem (0, 0, 0)
        wrist = pts[0].copy()
        pts_trans = pts - wrist
        
        # 2. Escala: MCP Médio (Landmark 9) = 1.0
        scale_d = np.linalg.norm(pts_trans[9])
        if scale_d < 1e-6:
            scale_d = 1.0
        pts_scaled = pts_trans / scale_d
        
        # 3. Base Ortonormal Local da Palma (Invariância a Rotações do Pulso)
        u_y = pts_scaled[9] / np.linalg.norm(pts_scaled[9])
        v_arch = pts_scaled[5] - pts_scaled[17]
        u_z = np.cross(v_arch, u_y)
        norm_z = np.linalg.norm(u_z)
        if norm_z < 1e-6:
            u_z = np.array([0.0, 0.0, 1.0])
        else:
            u_z /= norm_z
            
        u_x = np.cross(u_y, u_z)
        norm_x = np.linalg.norm(u_x)
        if norm_x > 1e-6:
            u_x /= norm_x
            
        R_local = np.array([u_x, u_y, u_z])
        pts_local = pts_scaled.dot(R_local.T)
        
        # 4. Ângulos de Falanges
        angles = {}
        for finger in ["Index", "Middle", "Ring", "Pinky"]:
            idx = FINGER_CHAINS[finger]
            v_mcp = pts_scaled[idx[1]] - pts_scaled[idx[0]]
            v_pip = pts_scaled[idx[2]] - pts_scaled[idx[1]]
            v_dip = pts_scaled[idx[3]] - pts_scaled[idx[2]]
            v_tip = pts_scaled[idx[4]] - pts_scaled[idx[3]]
            
            angles[f"{finger}_MCP_Flex"] = angle_between_vectors(v_mcp, v_pip)
            angles[f"{finger}_PIP_Flex"] = angle_between_vectors(v_pip, v_dip)
            angles[f"{finger}_DIP_Flex"] = angle_between_vectors(v_dip, v_tip)
            
        t_cmc = pts_scaled[1] - pts_scaled[0]
        t_mcp = pts_scaled[2] - pts_scaled[1]
        t_ip  = pts_scaled[3] - pts_scaled[2]
        t_tip = pts_scaled[4] - pts_scaled[3]
        angles["Thumb_MCP_Flex"] = angle_between_vectors(t_cmc, t_mcp)
        angles["Thumb_IP_Flex"]  = angle_between_vectors(t_ip, t_tip)
        angles["Thumb_Abduction"] = angle_between_vectors(pts_scaled[2] - pts_scaled[0], pts_scaled[5] - pts_scaled[0])
        angles["Thumb_Opposition"] = angle_between_vectors(t_tip, u_z)
        
        angles["Spread_Index_Middle"] = angle_between_vectors(pts_scaled[6] - pts_scaled[5], pts_scaled[10] - pts_scaled[9])
        angles["Spread_Middle_Ring"]  = angle_between_vectors(pts_scaled[10] - pts_scaled[9], pts_scaled[14] - pts_scaled[13])
        angles["Spread_Ring_Pinky"]   = angle_between_vectors(pts_scaled[14] - pts_scaled[13], pts_scaled[18] - pts_scaled[17])
        
        # 5. Distâncias Relativas
        dists = []
        wrist_local = pts_scaled[0]
        palm_center = pts_scaled[9]
        for tip in FINGERTIP_INDICES:
            dists.append(float(np.linalg.norm(pts_scaled[tip] - wrist_local)))
        for tip in FINGERTIP_INDICES:
            dists.append(float(np.linalg.norm(pts_scaled[tip] - palm_center)))
        dists.append(float(np.linalg.norm(pts_scaled[4] - pts_scaled[8])))
        dists.append(float(np.linalg.norm(pts_scaled[8] - pts_scaled[12])))
        dists.append(float(np.linalg.norm(pts_scaled[12] - pts_scaled[16])))
        dists.append(float(np.linalg.norm(pts_scaled[16] - pts_scaled[20])))
        
        # 6. Vetor Invariante
        feature_vector = np.concatenate([
            pts_local.flatten(),
            np.array(list(angles.values()), dtype=np.float64) / 180.0,
            np.array(dists, dtype=np.float64)
        ])
        
        return {
            "landmarks_local": pts_local,
            "feature_vector": feature_vector,
            "angles": angles,
            "relative_distances": dists,
            "scale_factor": scale_d
        }

    def predict(self, raw_landmarks: Any) -> Dict[str, Any]:
        """
        Infere a classe da mão comparando contra o catálogo de sementes e thresholds.
        """
        norm = self.normalize_hand_3d(raw_landmarks)
        pts_local = norm["landmarks_local"]
        feat = norm["feature_vector"]
        
        best_match = None
        min_dist = float("inf")
        
        for seed in self._compiled_seeds:
            seed_lms = seed["landmarks_3d"]
            seed_feat = seed["feature_vector"]
            w = seed["weights"]
            
            # Distância Euclidiana Ponderada por Junta
            joint_diffs = np.linalg.norm(pts_local - seed_lms, axis=1) # (21,)
            weighted_euc = np.sqrt(np.sum(w * (joint_diffs ** 2)) / np.sum(w))
            
            # Distância de Cosseno no vetor abstrato
            cos_dist = 1.0 - cosine_similarity(feat, seed_feat)
            
            # Distância Combinada
            total_dist = (0.70 * weighted_euc) + (0.30 * cos_dist)
            
            if total_dist < min_dist:
                min_dist = total_dist
                best_match = {
                    "seed": seed,
                    "joint_diffs": joint_diffs,
                    "total_dist": total_dist,
                    "weighted_euc": weighted_euc,
                    "cos_dist": cos_dist
                }
                
        if best_match is None:
            return {
                "class_name": "DESCONHECIDO",
                "clean_label": "NENHUM",
                "confidence": 0.0,
                "seed_name": "",
                "tolerance_passed": False
            }
            
        seed_ref = best_match["seed"]
        joint_diffs = best_match["joint_diffs"]
        thresholds = seed_ref["thresholds"]
        
        # Teste de tolerância articular (quantas juntas respeitam o threshold de desvio padrão)
        joint_passes = joint_diffs <= thresholds
        tolerance_score = float(np.mean(joint_passes))
        tolerance_passed = bool(np.sum(~joint_passes) <= 3) # Tolera até 3 micro-variações
        
        confidence = float(np.clip(1.0 / (1.0 + min_dist * 2.5), 0.0, 1.0))
        cls_name = seed_ref["class_name"]
        
        # Rótulo limpo para interfaces (remove 'classe_' se presente)
        clean_label = cls_name.replace("classe_", "")
        
        # Status por dedo
        finger_errors = {}
        for f_name, f_indices in FINGER_CHAINS.items():
            f_err = float(np.mean(joint_diffs[f_indices]))
            f_pass = bool(np.all(joint_passes[f_indices]))
            finger_errors[f_name] = {"error": round(f_err, 4), "passed": f_pass}
            
        return {
            "class_name": cls_name,
            "clean_label": clean_label,
            "confidence": round(confidence, 4),
            "seed_name": seed_ref["seed_name"],
            "view_label": seed_ref["view_label"],
            "distance": round(float(min_dist), 4),
            "tolerance_passed": tolerance_passed,
            "tolerance_score": round(tolerance_score, 4),
            "joint_errors": [round(float(d), 4) for d in joint_diffs],
            "finger_errors": finger_errors,
            "norm_landmarks": pts_local
        }


if __name__ == "__main__":
    print("[*] Testando CalibratedLibrasClassifier...")
    classifier = CalibratedLibrasClassifier()
    
    # Testa uma classe do próprio catálogo para verificação unitária
    sample_seed = classifier._compiled_seeds[0]
    result = classifier.predict(sample_seed["landmarks_3d"])
    print("\nResultado da Classificação de Teste:")
    print(f"  Classe Predita: {result['class_name']} ({result['clean_label']})")
    print(f"  Semente Casada: {result['seed_name']} [{result['view_label']}]")
    print(f"  Confiança: {result['confidence']*100:.2f}%")
    print(f"  Tolerância Aprovada: {result['tolerance_passed']} (Score: {result['tolerance_score']*100:.1f}%)")
    print("  Status dos Dedos:")
    for f, st in result["finger_errors"].items():
        print(f"    - {f}: Erro {st['error']:.3f} | {'PASS' if st['passed'] else 'FAIL'}")
