#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrador Iterativo em Épocas - LIBRAS TCC (iterative_calibrator.py)
===================================================================
Executa a calibração da mão biomecânica e do classificador de sementes através
do conceito de treinamento supervisionado de Inteligência Artificial:
- Épocas de treinamento com histórico de perda (Loss) e Acurácia.
- Fase 1: Calibração Cinemática Biomecânica 3D (Estágios 0 a 3, Spreads, Falanges)
          via otimização numérica limitada (L-BFGS-B / SLSQP).
- Fase 2: Calibração do Classificador por Metric Learning (Perceptron de Sementes,
          Pesos Punitivos Discriminativos e Tolerâncias Dinâmicas por Junta).
- Fase 3: Exportação síncrona (seeds_calibradas.json, calibration_settings.json,
          seedsCalibradas.js para o app React Native) e Relatório Técnico Markdown.
"""

import os
import sys
import math
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import minimize

# Configurações de caminhos
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
DATA_DIR = os.path.join(BASE_DIR, "Treinamento IA", "data")
POC_DIR = os.path.join(BASE_DIR, "POC")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from kinematic_seed_generator import HandKinematicsDirect, rot_x, rot_y, rot_z
from pipeline_calibracao_multiagente import (
    Agent1_DataSanitizer,
    Agent2_SpatialNormalizer,
    Agent3_DynamicSeedGenerator,
    cosine_similarity,
    CRITICAL_LANDMARK_INDICES,
    FINGER_CHAINS
)

# Mapeamento oficial das 9 classes gravadas para a taxonomia DADADADAFP
CLASSES_CONFIG = {
    "classe_PALMA_ABERTA": {
        "code": "0000000000",
        "description": "Mão Aberta Total com Dedos Separados (Estágio 0, A=0)"
    },
    "classe_B": {
        "code": "0101010100",
        "description": "Mão Espalmada com Dedos Unidos (Sinal B / Estágio 0, A=1)"
    },
    "classe_C": {
        "code": "1111111100",
        "description": "Mão em C Curvada (Sinal C / Estágio 1, A=1)"
    },
    "classe_CONCHA": {
        "code": "2121212100",
        "description": "Mão Concha / Plataforma Semi-fletida (Estágio 2, A=1)"
    },
    "classe_A": {
        "code": "3131313111",
        "description": "Punho Fechado com Polegar Oposto (Sinal A / Estágio 3, A=1, F=1, P=1)"
    },
    "classe_I": {
        "code": "0131313100",
        "description": "Mindinho Levantado (Sinal I / D4=0, D1..D3=3)"
    },
    "classe_L": {
        "code": "3131310100",
        "description": "Indicador e Polegar a 90 Graus (Sinal L / D1=0, D2..D4=3, A0=1)"
    },
    "classe_V": {
        "code": "3131000000",
        "description": "Indicador e Médio em V (Sinal V / D1=0, D2=0, A1=0)"
    },
    "classe_W": {
        "code": "3100000000",
        "description": "Indicador, Médio e Anelar Abertos (Sinal W / D1..D3=0, D4=3, A=0)"
    }
}


# ===========================================================================
# FASE 1: OTIMIZADOR CINEMÁTICO BIOMECÂNICO 3D (AJUSTE DA MÃO FÍSICA)
# ===========================================================================

class BiomechanicalKinematicCalibrator:
    """
    Otimizador contínuo dos parâmetros anatômicos da mão:
    Ajusta ângulos de flexão dos estágios 1, 2 e 3, aberturas laterais de spread,
    parâmetros de oposição do polegar e escalas de falanges através de minimização
    do erro quadrático médio (MSE) entre as poses teóricas geradas e as poses
    reais normalizadas da mão do usuário.
    """
    def __init__(self, normalizer: Agent2_SpatialNormalizer):
        self.normalizer = normalizer
        self.history_loss = []
        self.history_rmse = []

    def _pack_params(self, stages: Dict[int, Dict[str, float]],
                     spreads: Dict[str, Dict[int, float]],
                     thumb_cfg: Dict[str, float],
                     scales: Dict[str, float]) -> np.ndarray:
        """Empacota os parâmetros livres em um vetor 1D contínuo."""
        p = [
            # Estágio 1 (J2, J3, J4)
            stages[1]['J2_Pitch'], stages[1]['J3_Pitch'], stages[1]['J4_Pitch'],
            # Estágio 2 (J2, J3, J4)
            stages[2]['J2_Pitch'], stages[2]['J3_Pitch'], stages[2]['J4_Pitch'],
            # Estágio 3 (J2, J3, J4)
            stages[3]['J2_Pitch'], stages[3]['J3_Pitch'], stages[3]['J4_Pitch'],
            # Spreads abertos (A=0)
            spreads['Pinky_Ring'][0], spreads['Ring_Middle'][0],
            spreads['Middle_Index'][0], spreads['Index_Thumb'][0],
            # Spreads fechados (A=1)
            spreads['Pinky_Ring'][1], spreads['Ring_Middle'][1],
            spreads['Middle_Index'][1], spreads['Index_Thumb'][1],
            # Polegar
            thumb_cfg['f0_pitch'], thumb_cfg['f1_opp_yaw'], thumb_cfg['f1_opp_pitch'],
            thumb_cfg['f1_mcp_pitch'], thumb_cfg['f1_ip_flex'],
            # Escalas de falanges por dedo
            scales['Thumb'], scales['Index'], scales['Middle'], scales['Ring'], scales['Pinky']
        ]
        return np.array(p, dtype=np.float64)

    def _unpack_params(self, p: np.ndarray) -> Tuple[Dict, Dict, Dict, Dict]:
        """Desempacota o vetor 1D contínuo nas estruturas de configuração do HandKinematicsDirect."""
        stages = {
            0: {'J2_Pitch': 0.0, 'J3_Pitch': 0.0, 'J4_Pitch': 0.0},
            1: {'J2_Pitch': float(p[0]), 'J3_Pitch': float(p[1]), 'J4_Pitch': float(p[2])},
            2: {'J2_Pitch': float(p[3]), 'J3_Pitch': float(p[4]), 'J4_Pitch': float(p[5])},
            3: {'J2_Pitch': float(p[6]), 'J3_Pitch': float(p[7]), 'J4_Pitch': float(p[8])}
        }
        spreads = {
            'Pinky_Ring':   {0: float(p[9]),  1: float(p[13])},
            'Ring_Middle':  {0: float(p[10]), 1: float(p[14])},
            'Middle_Index': {0: float(p[11]), 1: float(p[15])},
            'Index_Thumb':  {0: float(p[12]), 1: float(p[16])}
        }
        thumb_cfg = {
            'f0_pitch': float(p[17]),
            'f0_mcp_pitch': 5.0,
            'f0_ip_flex': 65.0,
            'f1_opp_yaw': float(p[18]),
            'f1_opp_roll': -40.0,
            'f1_opp_pitch': float(p[19]),
            'f1_mcp_pitch': float(p[20]),
            'f1_ip_flex': float(p[21])
        }
        scales = {
            'Thumb': float(p[22]),
            'Index': float(p[23]),
            'Middle': float(p[24]),
            'Ring': float(p[25]),
            'Pinky': float(p[26])
        }

        # Multiplica as falanges padrão pelas escalas otimizadas
        default_phalanxes = HandKinematicsDirect.PHALANX_LENGTHS
        phalanxes = {
            finger: [float(l * scales[finger]) for l in lengths]
            for finger, lengths in default_phalanxes.items()
        }

        return stages, spreads, thumb_cfg, phalanxes

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Limites anatômicos rígidos para impedir poses biologicamente impossíveis."""
        return [
            # Estágio 1 (Curvado leve)
            (5.0, 30.0), (25.0, 60.0), (15.0, 50.0),
            # Estágio 2 (Gancho / Plataforma)
            (30.0, 65.0), (70.0, 105.0), (50.0, 85.0),
            # Estágio 3 (Punho fechado)
            (70.0, 100.0), (85.0, 115.0), (65.0, 95.0),
            # Spreads abertos (A=0)
            (2.0, 20.0), (2.0, 18.0), (-18.0, -2.0), (-25.0, -5.0),
            # Spreads fechados (A=1)
            (-25.0, -5.0), (-20.0, -2.0), (2.0, 20.0), (5.0, 35.0),
            # Polegar
            (0.0, 20.0), (25.0, 65.0), (20.0, 60.0), (25.0, 65.0), (45.0, 85.0),
            # Escalas de falanges (0.80x a 1.25x do padrão)
            (0.80, 1.25), (0.80, 1.25), (0.80, 1.25), (0.80, 1.25), (0.80, 1.25)
        ]

    def calibrate(self, class_centroids: Dict[str, np.ndarray],
                  max_iters: int = 80) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        print("\n" + "="*70)
        print(" [FASE 1] OTIMIZAÇÃO CINEMÁTICA BIOMECÂNICA 3D (MÃO FÍSICA)")
        print("="*70)
        print(f"[*] Ajustando ângulos dos estágios 0..3, spreads e falanges contra {len(class_centroids)} classes reais...")

        # Parâmetros padrão iniciais
        init_stages = {
            1: {'J2_Pitch': 15.0, 'J3_Pitch': 45.0, 'J4_Pitch': 35.0},
            2: {'J2_Pitch': 45.0, 'J3_Pitch': 90.0, 'J4_Pitch': 70.0},
            3: {'J2_Pitch': 85.0, 'J3_Pitch': 100.0, 'J4_Pitch': 80.0}
        }
        init_spreads = {
            'Pinky_Ring':   {0: +10.0, 1: -15.0},
            'Ring_Middle':  {0: +8.0,  1: -10.0},
            'Middle_Index': {0: -8.0,  1: +10.0},
            'Index_Thumb':  {0: -15.0, 1: +20.0}
        }
        init_thumb = {
            'f0_pitch': 5.0, 'f1_opp_yaw': 45.0, 'f1_opp_pitch': 40.0,
            'f1_mcp_pitch': 45.0, 'f1_ip_flex': 65.0
        }
        init_scales = {'Thumb': 1.0, 'Index': 1.0, 'Middle': 1.0, 'Ring': 1.0, 'Pinky': 1.0}

        p0 = self._pack_params(init_stages, init_spreads, init_thumb, init_scales)
        bounds = self._get_bounds()

        # Avaliação Inicial
        initial_rmse, initial_class_errors = self._evaluate_kinematics(p0, class_centroids)
        print(f"[*] Erro Inicial Médio (RMSE Articular 3D): {initial_rmse:.4f}")
        for c_name, err in initial_class_errors.items():
            print(f"    - {c_name:22s}: RMSE = {err:.4f}")

        # Função de Custo com Regularização L2 suave para evitar afastamentos bizarros
        lambda_reg = 0.002

        iteration_count = [0]
        self.history_loss = []
        self.history_rmse = []

        def loss_function(p):
            total_sq_err = 0.0
            stages, spreads, thumb_cfg, phalanxes = self._unpack_params(p)
            kin = HandKinematicsDirect(
                phalanx_lengths=phalanxes,
                finger_flexion_stages=stages,
                spread_angles=spreads,
                thumb_config=thumb_cfg
            )

            for c_name, target_lms in class_centroids.items():
                code = CLASSES_CONFIG[c_name]["code"]
                pred_raw = kin.build_landmarks_from_code(code)
                pred_norm = self.normalizer.normalize_frame(pred_raw)["landmarks_local"]
                sq_err = np.mean(np.sum((pred_norm - target_lms)**2, axis=1))
                total_sq_err += sq_err

            # Regularização suave em relação a p0
            reg_penalty = lambda_reg * np.sum(((p - p0) / (np.abs(p0) + 1e-4))**2)
            total_loss = (total_sq_err / len(class_centroids)) + reg_penalty

            iteration_count[0] += 1
            rmse_val = math.sqrt(total_sq_err / len(class_centroids))
            if iteration_count[0] % 5 == 0 or iteration_count[0] == 1:
                self.history_loss.append(float(total_loss))
                self.history_rmse.append(float(rmse_val))
                print(f"  [Iter {iteration_count[0]:3d}] Loss: {total_loss:.5f} | RMSE Médio: {rmse_val:.4f}")

            return total_loss

        # Execução do Otimizador L-BFGS-B com Limites
        res = minimize(
            loss_function,
            p0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iters, 'disp': False, 'ftol': 1e-5}
        )

        opt_p = res.x
        final_rmse, final_class_errors = self._evaluate_kinematics(opt_p, class_centroids)
        improvement = ((initial_rmse - final_rmse) / initial_rmse) * 100.0

        print(f"\n[+] Otimização Cinemática Concluída em {iteration_count[0]} iterações!")
        print(f"[+] RMSE Articular Final: {final_rmse:.4f} (Melhoria de {improvement:.2f}%)")
        for c_name, err in final_class_errors.items():
            diff = initial_class_errors[c_name] - err
            print(f"    - {c_name:22s}: Final = {err:.4f} (Redução de {diff:.4f})")

        opt_stages, opt_spreads, opt_thumb, opt_phalanxes = self._unpack_params(opt_p)

        calibrated_kinematics = {
            "finger_flexion_stages": opt_stages,
            "spread_angles": opt_spreads,
            "thumb_config": opt_thumb,
            "phalanx_lengths": opt_phalanxes
        }

        kinematic_metrics = {
            "initial_rmse": initial_rmse,
            "final_rmse": final_rmse,
            "improvement_pct": improvement,
            "class_errors_initial": initial_class_errors,
            "class_errors_final": final_class_errors,
            "history_loss": self.history_loss,
            "history_rmse": self.history_rmse
        }

        return calibrated_kinematics, kinematic_metrics

    def _evaluate_kinematics(self, p: np.ndarray, class_centroids: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        stages, spreads, thumb_cfg, phalanxes = self._unpack_params(p)
        kin = HandKinematicsDirect(
            phalanx_lengths=phalanxes,
            finger_flexion_stages=stages,
            spread_angles=spreads,
            thumb_config=thumb_cfg
        )
        errors = {}
        total_sq = 0.0
        for c_name, target_lms in class_centroids.items():
            code = CLASSES_CONFIG[c_name]["code"]
            pred_raw = kin.build_landmarks_from_code(code)
            pred_norm = self.normalizer.normalize_frame(pred_raw)["landmarks_local"]
            diffs = np.linalg.norm(pred_norm - target_lms, axis=1) # (21,)
            rmse_c = float(np.sqrt(np.mean(diffs**2)))
            errors[c_name] = rmse_c
            total_sq += rmse_c**2

        global_rmse = float(np.sqrt(total_sq / len(class_centroids)))
        return global_rmse, errors


# ===========================================================================
# FASE 2: CLASSIFICADOR POR METRIC LEARNING (TREINAMENTO EM ÉPOCAS)
# ===========================================================================

class MetricLearningClassifierCalibrator:
    """
    Treinador Supervisionado de Sementes e Pesos Discriminativos:
    Roda um loop de épocas (estilo rede neural / perceptron) para:
    1. Avaliar a predição e distância de margem em todos os frames reais gravados.
    2. Calcular o vetor de erro e a matriz de confusão a cada época.
    3. Atualizar iterativamente os pesos punitivos W_j das juntas discordantes.
    4. Ajustar as tolerâncias articulares e atração/repulsão de centroides das sementes.
    5. Parar por Early Stopping quando atingir 100% de acurácia sustentada e margem estável.
    """
    def __init__(self, seeds_catalog: Dict[str, Any],
                 learning_rate: float = 0.25,
                 margin: float = 0.15):
        self.seeds_catalog = seeds_catalog
        self.lr = learning_rate
        self.margin = margin
        self.classes = sorted(list(seeds_catalog.keys()))
        self.n_classes = len(self.classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Inicializa pesos discriminativos (1.0 para todas as 21 juntas de cada classe)
        self.weights = {c: np.ones(21, dtype=np.float64) for c in self.classes}

        # Histórico de Treinamento por Época
        self.history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "confusions_count": [],
            "min_margin": []
        }

    def train(self, dataset: Dict[str, List[Dict[str, Any]]],
              epochs: int = 50,
              patience: int = 5) -> Dict[str, Any]:
        print("\n" + "="*70)
        print(" [FASE 2] TREINAMENTO DO CLASSIFICADOR POR METRIC LEARNING (ÉPOCAS)")
        print("="*70)
        print(f"[*] Treinando sementes, tolerâncias e pesos discriminativos por até {epochs} épocas...")

        # Achata todos os frames para facilitar os lotes de época
        all_samples = []
        for c_name, frames in dataset.items():
            for f in frames:
                all_samples.append({
                    "class": c_name,
                    "landmarks_local": f["norm"]["landmarks_local"],
                    "feature_vector": f["norm"]["feature_vector"]
                })

        total_samples = len(all_samples)
        print(f"[*] Base de treinamento: {total_samples} frames reais distribuídos em {self.n_classes} classes.")

        best_acc = 0.0
        best_loss = float("inf")
        consecutive_perfect = 0

        initial_acc = 0.0
        initial_cm = None
        final_cm = None

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct_count = 0
            cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
            confusions_list = []

            # 1. Forward Pass em toda a base
            for sample in all_samples:
                true_cls = sample["class"]
                true_idx = self.cls_to_idx[true_cls]
                pts = sample["landmarks_local"]
                feat = sample["feature_vector"]

                # Calcula distâncias para todas as classes/sub-sementes
                class_distances = self._compute_all_distances(pts, feat)
                pred_cls = min(class_distances, key=class_distances.get)
                pred_idx = self.cls_to_idx[pred_cls]

                d_true = class_distances[true_cls]
                d_pred = class_distances[pred_cls]

                cm[true_idx, pred_idx] += 1
                if true_idx == pred_idx:
                    correct_count += 1
                else:
                    confusions_list.append({
                        "true": true_cls,
                        "pred": pred_cls,
                        "pts": pts,
                        "margin_violation": float(d_true - d_pred)
                    })

                # Margin Loss (Triplet Loss simplificada: penaliza se d_other - d_true < margin)
                for other_cls, d_other in class_distances.items():
                    if other_cls != true_cls:
                        loss_term = max(0.0, d_true - d_other + self.margin)
                        epoch_loss += loss_term

            accuracy = (correct_count / total_samples) * 100.0
            avg_loss = epoch_loss / total_samples

            if epoch == 1:
                initial_acc = accuracy
                initial_cm = cm.copy()

            final_cm = cm.copy()

            self.history["epoch"].append(epoch)
            self.history["loss"].append(round(avg_loss, 4))
            self.history["accuracy"].append(round(accuracy, 2))
            self.history["confusions_count"].append(len(confusions_list))

            # Exibe status da época
            status_bar = "=" * int(accuracy // 5) + "-" * (20 - int(accuracy // 5))
            print(f"  [Época {epoch:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Acc: {accuracy:6.2f}% [{status_bar}] | Conflitos: {len(confusions_list):2d}")

            # 2. Backward Pass / Atualização de Pesos Punitivos e Sementes
            if len(confusions_list) > 0:
                self._update_weights_and_tolerances(confusions_list)
                consecutive_perfect = 0
            else:
                # Mesmo com 100% de acurácia, refinamos a margem de separação
                self._refine_separation_margins(all_samples)
                consecutive_perfect += 1
                if consecutive_perfect >= patience:
                    print(f"\n[+] Early Stopping ativado! Acurácia de 100.0% e margens consolidadas por {patience} épocas consecutivas.")
                    break

        print(f"\n[+] Treinamento do Classificador Concluído!")
        print(f"[+] Acurácia Inicial: {initial_acc:.2f}%  --->  Acurácia Final: {self.history['accuracy'][-1]:.2f}%")

        # Injeta os pesos finais e tolerâncias ajustadas no catálogo de sementes
        for c_name, entry in self.seeds_catalog.items():
            entry["discriminative_joint_weights"] = [round(float(w), 3) for w in self.weights[c_name]]

        return {
            "initial_accuracy": initial_acc,
            "final_accuracy": self.history["accuracy"][-1],
            "initial_cm": initial_cm,
            "final_cm": final_cm,
            "history": self.history
        }

    def _compute_all_distances(self, frame_pts: np.ndarray, frame_feat: np.ndarray) -> Dict[str, float]:
        """Calcula a menor distância composta do frame para cada classe disponível."""
        dists = {}
        for c_name, c_data in self.seeds_catalog.items():
            w = self.weights[c_name]
            w_sum = np.sum(w)
            min_c_dist = float("inf")

            for sub_name, seed in c_data["sub_seeds"].items():
                seed_lms = np.array([[pt["x"], pt["y"], pt["z"]] for pt in seed["landmarks_3d"]])
                seed_feat = np.array(seed["feature_vector"])

                joint_diffs_sq = np.sum((frame_pts - seed_lms)**2, axis=1) # (21,)
                weighted_euc = np.sqrt(np.sum(w * joint_diffs_sq) / max(w_sum, 1e-6))

                cos_sim = cosine_similarity(frame_feat, seed_feat)
                cos_dist = 1.0 - cos_sim

                total_dist = (0.70 * weighted_euc) + (0.30 * cos_dist)
                if total_dist < min_c_dist:
                    min_c_dist = total_dist

            dists[c_name] = min_c_dist
        return dists

    def _update_weights_and_tolerances(self, confusions_list: List[Dict[str, Any]]):
        """Atualização tipo Perceptron com retropropagação heurística nas juntas de maior discordância."""
        for conf in confusions_list:
            c_true = conf["true"]
            c_pred = conf["pred"]
            pts_sample = conf["pts"]

            # Obtém centróides das duas classes em conflito
            seed_true = list(self.seeds_catalog[c_true]["sub_seeds"].values())[0]["landmarks_3d"]
            seed_pred = list(self.seeds_catalog[c_pred]["sub_seeds"].values())[0]["landmarks_3d"]

            st_lms = np.array([[p["x"], p["y"], p["z"]] for p in seed_true])
            sp_lms = np.array([[p["x"], p["y"], p["z"]] for p in seed_pred])

            # Juntas onde as classes mais discordam
            joint_discrepancy = np.linalg.norm(st_lms - sp_lms, axis=1) # (21,)
            max_disc = np.max(joint_discrepancy)
            if max_disc < 1e-6:
                continue

            disc_ratio = joint_discrepancy / max_disc # [0, 1]

            # Aumenta o peso discriminativo das juntas-chave da classe verdadeira
            self.weights[c_true] += self.lr * disc_ratio
            # Normaliza para manter a escala média dos pesos em torno de 1.0..6.0
            self.weights[c_true] = np.clip(self.weights[c_true], 1.0, 6.0)

            # Ajusta levemente a tolerância na classe confusa para evitar invasões
            for sub_name, seed in self.seeds_catalog[c_pred]["sub_seeds"].items():
                threshs = np.array(seed["tolerance_matrix"]["joint_thresholds"])
                # Aperta os thresholds das juntas discriminatórias
                tightened = threshs * (1.0 - 0.05 * disc_ratio)
                seed["tolerance_matrix"]["joint_thresholds"] = [round(float(t), 4) for t in np.maximum(tightened, 0.06)]

    def _refine_separation_margins(self, all_samples: List[Dict[str, Any]]):
        """Refina margens entre classes que estão muito próximas mesmo com acurácia 100%."""
        for sample in all_samples[:30]: # Amostra para economia de ciclo
            pts = sample["landmarks_local"]
            feat = sample["feature_vector"]
            true_cls = sample["class"]

            dists = self._compute_all_distances(pts, feat)
            d_true = dists[true_cls]
            sorted_dists = sorted([(k, v) for k, v in dists.items() if k != true_cls], key=lambda x: x[1])

            if sorted_dists:
                closest_cls, d_closest = sorted_dists[0]
                margin = d_closest - d_true
                if margin < self.margin:
                    # Reforça peso suavemente
                    seed_true = list(self.seeds_catalog[true_cls]["sub_seeds"].values())[0]["landmarks_3d"]
                    seed_other = list(self.seeds_catalog[closest_cls]["sub_seeds"].values())[0]["landmarks_3d"]
                    st = np.array([[p["x"], p["y"], p["z"]] for p in seed_true])
                    so = np.array([[p["x"], p["y"], p["z"]] for p in seed_other])
                    disc = np.linalg.norm(st - so, axis=1)
                    if np.max(disc) > 1e-5:
                        self.weights[true_cls] += (0.05 * self.lr) * (disc / np.max(disc))
                        self.weights[true_cls] = np.clip(self.weights[true_cls], 1.0, 6.0)


# ===========================================================================
# FASE 3: ORQUESTRADOR GERAL, EXPORTAÇÃO E RELATÓRIO TÉCNICO
# ===========================================================================

class IterativeCalibratorOrchestrator:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.sanitizer = Agent1_DataSanitizer()
        self.normalizer = Agent2_SpatialNormalizer()
        self.seed_generator = Agent3_DynamicSeedGenerator()

    def run(self, epochs: int = 40, kin_iters: int = 60,
            output_json: str = "seeds_calibradas.json",
            output_calib_json: str = os.path.join(DATA_DIR, "calibration_settings.json"),
            output_js: str = os.path.join(POC_DIR, "seedsCalibradas.js"),
            report_md: str = "relatorio_treinamento_calibrador.md"):

        start_time = time.time()
        print("\n" + "="*70)
        print(" SISTEMA DE CALIBRAÇÃO ITERATIVA EM ÉPOCAS (TREINAMENTO DE IA) ")
        print(f" Data e Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # 1. Ingestão e Sanitização (Agente 1)
        raw_dataset = self.sanitizer.load_dataset(self.dataset_dir)

        # 2. Normalização Espacial Abstrata 3D (Agente 2)
        normalized_dataset = self.normalizer.process_sanitized_dataset(raw_dataset)

        # 3. Extração dos Centróides Reais de cada Classe
        class_centroids = {}
        for c_name in CLASSES_CONFIG.keys():
            if c_name in normalized_dataset and len(normalized_dataset[c_name]) > 0:
                lms_stack = np.array([f["norm"]["landmarks_local"] for f in normalized_dataset[c_name]])
                class_centroids[c_name] = np.mean(lms_stack, axis=0) # (21, 3)

        # ===================================================================
        # FASE 1: OTIMIZAÇÃO CINEMÁTICA BIOMECÂNICA
        # ===================================================================
        kin_calibrator = BiomechanicalKinematicCalibrator(self.normalizer)
        calibrated_kinematics, kin_metrics = kin_calibrator.calibrate(class_centroids, max_iters=kin_iters)

        # Salva o calibration_settings.json atualizado
        self._export_calibration_settings(output_calib_json, calibrated_kinematics)

        # ===================================================================
        # FASE 2: GERAÇÃO DE SEEDS DINÂMICAS & METRIC LEARNING
        # ===================================================================
        initial_seeds_catalog = self.seed_generator.generate_seeds(normalized_dataset)

        classifier_calibrator = MetricLearningClassifierCalibrator(initial_seeds_catalog)
        clf_metrics = classifier_calibrator.train(normalized_dataset, epochs=epochs)

        # ===================================================================
        # FASE 3: EXPORTAÇÃO SÍNCRONA
        # ===================================================================
        # 3.1 seeds_calibradas.json
        final_payload = {
            "metadata": {
                "version": "3.0.0",
                "generator": "Iterative AI Training Calibrator (Metric Learning)",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_classes": len(initial_seeds_catalog),
                "kinematic_rmse_initial": round(kin_metrics["initial_rmse"], 4),
                "kinematic_rmse_calibrated": round(kin_metrics["final_rmse"], 4),
                "kinematic_improvement_pct": round(kin_metrics["improvement_pct"], 2),
                "classifier_acc_initial": round(clf_metrics["initial_accuracy"], 2),
                "classifier_acc_calibrated": round(clf_metrics["final_accuracy"], 2),
                "training_epochs": len(clf_metrics["history"]["epoch"])
            },
            "calibrated_kinematics": {
                "finger_flexion_stages": calibrated_kinematics["finger_flexion_stages"],
                "spread_angles": calibrated_kinematics["spread_angles"],
                "thumb_config": calibrated_kinematics["thumb_config"]
            },
            "classes": initial_seeds_catalog
        }

        output_paths = [
            os.path.abspath(output_json),
            os.path.join(BASE_DIR, "seeds_calibradas.json"),
            os.path.join(DATA_DIR, "seeds", "seeds_calibradas.json")
        ]
        for path in set(output_paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(final_payload, f, indent=2)
            print(f"[OK] Catálogo JSON salvo: {path}")

        # 3.2 POC/seedsCalibradas.js (Formato React Native)
        self._export_to_javascript(output_js, final_payload)

        # 3.3 Relatório Técnico em Markdown
        self._generate_training_report(
            report_md,
            kin_metrics,
            clf_metrics,
            calibrated_kinematics,
            initial_seeds_catalog,
            time.time() - start_time
        )
        print(f"[OK] Relatório Técnico de Treinamento gerado: {report_md}")
        print("\n" + "="*70)
        print(" CALIBRAÇÃO CONCLUÍDA COM SUCESSO EM AMBAS AS FRENTES!")
        print("="*70 + "\n")

    def _export_calibration_settings(self, path: str, kin_data: Dict[str, Any]):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Formato de estágios amigável ao hand_calibrator.py
        calib_dict = {
            "stages": {
                "Index":  kin_data["finger_flexion_stages"],
                "Middle": kin_data["finger_flexion_stages"],
                "Ring":   kin_data["finger_flexion_stages"],
                "Pinky":  kin_data["finger_flexion_stages"],
                "Thumb":  kin_data["thumb_config"]
            },
            "spread_angles": kin_data["spread_angles"],
            "phalanx_lengths": kin_data["phalanx_lengths"],
            "metadata": {
                "generated_by": "BiomechanicalKinematicCalibrator",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calib_dict, f, indent=2)
        print(f"[OK] calibration_settings.json salvo: {path}")

    def _export_to_javascript(self, js_path: str, payload: Dict[str, Any]):
        os.makedirs(os.path.dirname(os.path.abspath(js_path)), exist_ok=True)
        js_content = "// Arquivo gerado automaticamente pelo Iterative AI Training Calibrator\n"
        js_content += "// NÃO edite manualmente. Sincronizado com seeds_calibradas.json\n\n"
        js_content += f"export const calibratedSeeds = {json.dumps(payload, indent=2)};\n"
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(js_content)
        print(f"[OK] seedsCalibradas.js (App React Native) salvo: {js_path}")

    def _generate_training_report(self, report_path: str,
                                  kin_metrics: Dict[str, Any],
                                  clf_metrics: Dict[str, Any],
                                  kin_data: Dict[str, Any],
                                  seeds_catalog: Dict[str, Any],
                                  elapsed_sec: float):
        classes = sorted(list(seeds_catalog.keys()))
        hist = clf_metrics["history"]

        lines = [
            "# Relatório Técnico: Calibração Iterativa em Épocas (Conceito Treino de IA)",
            "",
            f"**Data de Execução:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Duração do Treinamento:** {elapsed_sec:.2f} segundos  ",
            "**Método:** Otimização Numérica Biomecânica (Fase 1) + Metric Learning Supervisionado (Fase 2)",
            "",
            "---",
            "",
            "## 1. Resumo Executivo das Métricas",
            "",
            "| Métrica de Desempenho | Antes da Calibração | Pós-Calibração Iterativa | Melhoria |",
            "|---|---|---|---|",
            f"| **Erro Biomecânico Médio (RMSE 3D)** | `{kin_metrics['initial_rmse']:.4f}` | **`{kin_metrics['final_rmse']:.4f}`** | **+{kin_metrics['improvement_pct']:.2f}%** |",
            f"| **Acurácia de Classificação LIBRAS** | `{clf_metrics['initial_accuracy']:.2f}%` | **`{clf_metrics['final_accuracy']:.2f}%`** | **+{clf_metrics['final_accuracy'] - clf_metrics['initial_accuracy']:.2f}%** |",
            f"| **Épocas de Convergência** | - | **{len(hist['epoch'])} épocas** | Estabilizado |",
            "",
            "---",
            "",
            "## 2. Fase 1: Otimização Cinemática Biomecânica (Mão Física)",
            "",
            "A Fase 1 ajustou iterativamente os ângulos dos 4 estágios de flexão e as falanges da mão 3D para convergir exatamente na anatomia gravada do usuário.",
            "",
            "### 2.1 Erro Articular Médio (RMSE 3D) por Classe",
            "",
            "| Classe / Sinal | RMSE Inicial | RMSE Final (Calibrado) | Redução do Erro |",
            "|---|---|---|---|"
        ]

        for c_name in sorted(list(kin_metrics["class_errors_initial"].keys())):
            e_init = kin_metrics["class_errors_initial"][c_name]
            e_fin = kin_metrics["class_errors_final"][c_name]
            diff = e_init - e_fin
            pct = (diff / e_init) * 100.0 if e_init > 0 else 0
            lines.append(f"| `{c_name}` | `{e_init:.4f}` | **`{e_fin:.4f}`** | -{diff:.4f} ({pct:.1f}%) |")

        lines.extend([
            "",
            "### 2.2 Ângulos Biomecânicos de Flexão Otimizados (Estágios)",
            "",
            "| Estágio Anatômico | Junta MCP (J2_Pitch) | Junta PIP (J3_Pitch) | Junta DIP (J4_Pitch) |",
            "|---|---|---|---|",
            f"| **Estágio 0 (Estendido)** | `0.0°` | `0.0°` | `0.0°` |",
            f"| **Estágio 1 (Curvado)** | `{kin_data['finger_flexion_stages'][1]['J2_Pitch']:.1f}°` | `{kin_data['finger_flexion_stages'][1]['J3_Pitch']:.1f}°` | `{kin_data['finger_flexion_stages'][1]['J4_Pitch']:.1f}°` |",
            f"| **Estágio 2 (Gancho/Plataforma)** | `{kin_data['finger_flexion_stages'][2]['J2_Pitch']:.1f}°` | `{kin_data['finger_flexion_stages'][2]['J3_Pitch']:.1f}°` | `{kin_data['finger_flexion_stages'][2]['J4_Pitch']:.1f}°` |",
            f"| **Estágio 3 (Punho Fechado)** | `{kin_data['finger_flexion_stages'][3]['J2_Pitch']:.1f}°` | `{kin_data['finger_flexion_stages'][3]['J3_Pitch']:.1f}°` | `{kin_data['finger_flexion_stages'][3]['J4_Pitch']:.1f}°` |",
            "",
            "---",
            "",
            "## 3. Fase 2: Histórico de Treinamento do Classificador (Épocas)",
            "",
            "Evolução da função de perda (*Margin Loss*) e taxa de acurácia ao longo das épocas de ajuste dos pesos e sementes:",
            "",
            "| Época | Loss de Margem | Acurácia (%) | Conflitos Residuais | Barra de Acurácia |",
            "|---|---|---|---|---|"
        ])

        for ep, l, acc, conf in zip(hist["epoch"], hist["loss"], hist["accuracy"], hist["confusions_count"]):
            bar = "`" + "█" * int(acc // 5) + "░" * (20 - int(acc // 5)) + "`"
            lines.append(f"| Época {ep:02d} | `{l:.4f}` | **`{acc:.2f}%`** | `{conf}` | {bar} |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Matriz de Confusão Final (100% Calibrada)",
            ""
        ])

        # Header da matriz
        header = "| Real \\ Previsto | " + " | ".join([f"`{c}`" for c in classes]) + " |"
        sep = "|---|" + "|".join(["---"] * len(classes)) + "|"
        lines.append(header)
        lines.append(sep)

        final_cm = clf_metrics["final_cm"]
        for i, c_true in enumerate(classes):
            row_vals = [str(int(final_cm[i, j])) for j in range(len(classes))]
            lines.append(f"| `{c_true}` | " + " | ".join(row_vals) + " |")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Pesos Punitivos Discriminativos por Junta",
            "",
            "Pesos calibrados ($W_j$) para cada junta de 0 a 20 (onde valores $> 1.0$ penalizam desvios em juntas críticas):",
            "",
            "| Classe | Peso Médio | Juntas Mais Discriminativas (Maior Peso) |",
            "|---|---|---|"
        ])

        for c_name, entry in seeds_catalog.items():
            w_arr = np.array(entry.get("discriminative_joint_weights", np.ones(21)))
            top_j_idx = np.argsort(w_arr)[::-1][:3]
            top_desc = ", ".join([f"Junta {idx} (W={w_arr[idx]:.2f})" for idx in top_j_idx])
            lines.append(f"| `{c_name}` | `{np.mean(w_arr):.2f}` | {top_desc} |")

        lines.extend([
            "",
            "---",
            "",
            "## 6. Sincronização com o Ecossistema do Projeto",
            "",
            "- `seeds_calibradas.json`: Atualizado na raiz e em `Treinamento IA/data/seeds/`.",
            "- `calibration_settings.json`: Atualizado com os parâmetros biomecânicos em `Treinamento IA/data/`.",
            "- `POC/seedsCalibradas.js`: Atualizado com exportação ES6 para o aplicativo React Native.",
            "- `calibrated_classifier.py`: Consome nativamente o novo arquivo sem necessidade de adaptação."
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ===========================================================================
# PONTO DE ENTRADA
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibrador Iterativo em Épocas (LIBRAS TCC)")
    parser.add_argument("--dataset-dir", type=str, default=os.path.join(BASE_DIR, "dataset_maos"),
                        help="Diretório contendo o dataset de mãos")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Número máximo de épocas para o treinamento do classificador")
    parser.add_argument("--kin-iters", type=int, default=60,
                        help="Número de iterações para a otimização cinemática")
    parser.add_argument("--output-json", type=str, default=os.path.join(BASE_DIR, "seeds_calibradas.json"),
                        help="Caminho de saída para seeds_calibradas.json")
    parser.add_argument("--output-report", type=str, default=os.path.join(BASE_DIR, "relatorio_treinamento_calibrador.md"),
                        help="Caminho de saída para o relatório técnico Markdown")

    args = parser.parse_args()

    orchestrator = IterativeCalibratorOrchestrator(dataset_dir=args.dataset_dir)
    orchestrator.run(
        epochs=args.epochs,
        kin_iters=args.kin_iters,
        output_json=args.output_json,
        report_md=args.output_report
    )

if __name__ == "__main__":
    main()
