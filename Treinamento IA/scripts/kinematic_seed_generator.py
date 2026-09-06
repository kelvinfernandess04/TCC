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

class HandKinematicsDirect:
    """
    Biomechanical Forward Kinematics Engine for MediaPipe Hands (21 3D Landmarks).
    Enforces strict joint constraints, anatomical segment lengths, and realistic thumb opposition.
    """

    # Standard Anatomical Euclidean Segment Lengths
    METACARPAL_LENGTHS = {
        'Thumb':  0.52,  # 0 -> 1 (Thumb CMC)
        'Index':  1.00,  # 0 -> 5 (Index MCP)
        'Middle': 1.03,  # 0 -> 9 (Middle MCP)
        'Ring':   0.96,  # 0 -> 13 (Ring MCP)
        'Pinky':  0.88   # 0 -> 17 (Pinky MCP)
    }

    PHALANX_LENGTHS = {
        'Thumb':  [0.35, 0.32, 0.28],  # 1->2 (MCP), 2->3 (IP), 3->4 (TIP)
        'Index':  [0.42, 0.27, 0.20],  # 5->6 (PIP), 6->7 (DIP), 7->8 (TIP)
        'Middle': [0.46, 0.30, 0.22],  # 9->10 (PIP), 10->11 (DIP), 11->12 (TIP)
        'Ring':   [0.42, 0.28, 0.21],  # 13->14 (PIP), 14->15 (DIP), 15->16 (TIP)
        'Pinky':  [0.34, 0.22, 0.18]   # 17->18 (PIP), 18->19 (DIP), 19->20 (TIP)
    }

    # Base Metacarpal Angles from Wrist in the Palm Plane (XY coronal plane)
    METACARPAL_BASE_ANGLES = {
        'Thumb':  -45.0,  # Base divergente padrão
        'Index':  -15.0,
        'Middle':   0.0,  # Eixo central da mão
        'Ring':   +15.0,
        'Pinky':  +30.0
    }

    # Mapping of Long Finger Flexion Stages D (0 to 3) to Joint Pitch Angles
    FINGER_FLEXION_STAGES = {
        0: {'J2_Pitch':  0.0, 'J3_Pitch':   0.0, 'J4_Pitch':  0.0},  # Extended
        1: {'J2_Pitch': 15.0, 'J3_Pitch':  45.0, 'J4_Pitch': 35.0},  # Curved
        2: {'J2_Pitch': 45.0, 'J3_Pitch':  90.0, 'J4_Pitch': 70.0},  # Hooked / Claw
        3: {'J2_Pitch': 85.0, 'J3_Pitch': 100.0, 'J4_Pitch': 80.0}   # Fist / Clenched
    }

    # Spread Angle Values (A) for Fingers
    SPREAD_ANGLES = {
        # A3 (Mindinho): Base +30°. Aberto vai para +40°. Fechado recua para +15°.
        'Pinky_Ring':   {0: +10.0, 1: -15.0},
        # A2 (Anelar): Base +15°. Aberto vai para +23°. Fechado recua para +5°.
        'Ring_Middle':  {0: +8.0,  1: -10.0},
        # A1 (Indicador): Base -15°. Aberto vai para -23°. Fechado recua para -5°.
        'Middle_Index': {0: -8.0,  1: +10.0},
        # A0 (Polegar): Base -45°. Aberto vai para -60°. Fechado aduz para -25°.
        'Index_Thumb':  {0: -15.0, 1: +20.0}
    }

    def __init__(self,
                 metacarpal_lengths: Optional[Dict[str, float]] = None,
                 phalanx_lengths: Optional[Dict[str, List[float]]] = None,
                 metacarpal_base_angles: Optional[Dict[str, float]] = None,
                 finger_flexion_stages: Optional[Dict[int, Dict[str, float]]] = None,
                 spread_angles: Optional[Dict[str, Dict[int, float]]] = None,
                 thumb_config: Optional[Dict[str, Any]] = None):
        """Precompute palm base positions to ensure exact Euclidean metacarpal lengths."""
        self.metacarpal_lengths = dict(metacarpal_lengths) if metacarpal_lengths else dict(self.METACARPAL_LENGTHS)
        self.phalanx_lengths = {k: list(v) for k, v in (phalanx_lengths.items() if phalanx_lengths else self.PHALANX_LENGTHS.items())}
        self.metacarpal_base_angles = dict(metacarpal_base_angles) if metacarpal_base_angles else dict(self.METACARPAL_BASE_ANGLES)
        self.finger_flexion_stages = {k: dict(v) for k, v in (finger_flexion_stages.items() if finger_flexion_stages else self.FINGER_FLEXION_STAGES.items())}
        self.spread_angles = {k: dict(v) for k, v in (spread_angles.items() if spread_angles else self.SPREAD_ANGLES.items())}
        self.thumb_config = dict(thumb_config) if thumb_config else {
            'f0_pitch': 5.0,
            'f0_mcp_pitch': 5.0,
            'f0_ip_flex': 65.0,
            'f1_opp_yaw': 45.0,
            'f1_opp_roll': -40.0,
            'f1_opp_pitch': 40.0,
            'f1_mcp_pitch': 45.0,
            'f1_ip_flex': 65.0
        }

        self.palm_bases: Dict[str, np.ndarray] = {}
        for finger, length in self.metacarpal_lengths.items():
            deg = self.metacarpal_base_angles[finger]
            rad = math.radians(deg)
            # Bone extends along -Y (MediaPipe canonical orientation, distal direction)
            # and spreads along X in the coronal palm plane (Z = 0)
            vec = np.array([
                length * math.sin(rad),
                -length * math.cos(rad),
                0.0
            ], dtype=np.float64)
            self.palm_bases[finger] = vec

    @staticmethod
    def is_valid_pose(dadadafafp_code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate whether a 10-digit DADADADAFP taxonomy code represents a physically
        possible biomechanical pose.

        Pruning Rules:
        1. String length and digit domain checks.
        2. MCP Collateral Ligament Tightening:
           When MCP flexion D >= 2 (Garrado or Fechado), the collateral ligaments are
           taut, physically locking lateral abduction/adduction (spread) to closed (A = 1).
        3. Thumb Opposition vs Abduction compatibility:
           When Thumb opposes deeply across the palm (F = 1) with flexed fingers,
           hyper-abduction (A0 = 0) is pruned.
        """
        if not isinstance(dadadafafp_code, str) or len(dadadafafp_code) != 10:
            return False, f"Code must be a 10-character string, got '{dadadafafp_code}'"

        # Character format checks
        d4_c, a3_c, d3_c, a2_c, d2_c, a1_c, d1_c, a0_c, f_c, p_c = dadadafafp_code

        if d4_c not in '0123' or d3_c not in '0123' or d2_c not in '0123' or d1_c not in '0123':
            return False, "Finger flexion stages [D] must be digits in '0123'"

        if a3_c not in '01' or a2_c not in '01' or a1_c not in '01' or a0_c not in '01':
            return False, "Spread states [A] must be '0' (Open) or '1' (Closed)"

        if f_c not in '01' or p_c not in '01':
            return False, "Thumb states [F] and [P] must be '0' or '1'"

        d4 = int(d4_c)  # Pinky
        a3 = int(a3_c)  # Pinky-Ring Spread
        d3 = int(d3_c)  # Ring
        a2 = int(a2_c)  # Ring-Middle Spread
        d2 = int(d2_c)  # Middle
        a1 = int(a1_c)  # Middle-Index Spread
        d1 = int(d1_c)  # Index
        a0 = int(a0_c)  # Index-Thumb CMC Abduction
        f  = int(f_c)   # Thumb Opposition
        p  = int(p_c)   # Thumb IP Flexion

        # --- BIOMECHANICAL CONSTRAINT 1: Collateral Ligament Lock (D >= 2) ---
        # Pinky-Ring spread locked closed (A3 = 1) if either adjacent finger is flexed (D >= 2)
        if (d4 >= 2 or d3 >= 2) and a3 == 0:
            return False, f"Pinky-Ring spread (A3=0) impossible when Pinky (D4={d4}) or Ring (D3={d3}) >= 2"

        # Ring-Middle spread locked closed (A2 = 1) if either adjacent finger is flexed (D >= 2)
        if (d3 >= 2 or d2 >= 2) and a2 == 0:
            return False, f"Ring-Middle spread (A2=0) impossible when Ring (D3={d3}) or Middle (D2={d2}) >= 2"

        # Middle-Index spread locked closed (A1 = 1) if either adjacent finger is flexed (D >= 2)
        if (d2 >= 2 or d1 >= 2) and a1 == 0:
            return False, f"Middle-Index spread (A1=0) impossible when Middle (D2={d2}) or Index (D1={d1}) >= 2"

        # --- BIOMECHANICAL CONSTRAINT 2: Thumb Opposition vs Abduction ---
        # When Thumb crosses palm (F = 1) and Index is closed (D1 >= 2), wide radial abduction (A0 = 0) is restricted
        if f == 1 and d1 >= 2 and a0 == 0:
            return False, f"Thumb wide abduction (A0=0) impossible during full opposition (F=1) with closed Index (D1={d1})"

        return True, None

    def build_landmarks_from_code(self, dadadafafp_code: str) -> np.ndarray:
        """
        Compute the exact 21 3D landmarks for a hand configuration given by the 10-digit
        DADADADAFP taxonomy code.

        Returns:
            np.ndarray of shape (21, 3) representing 3D coordinates (x, y, z)
            with Landmark 0 (Wrist) at (0.0, 0.0, 0.0).
        """
        is_valid, reason = self.is_valid_pose(dadadafafp_code)
        if not is_valid:
            raise ValueError(f"Invalid biomechanical pose code '{dadadafafp_code}': {reason}")

        # Parse digits
        d4 = int(dadadafafp_code[0])  # Pinky Flexion
        a3 = int(dadadafafp_code[1])  # Pinky-Ring Spread
        d3 = int(dadadafafp_code[2])  # Ring Flexion
        a2 = int(dadadafafp_code[3])  # Ring-Middle Spread
        d2 = int(dadadafafp_code[4])  # Middle Flexion
        a1 = int(dadadafafp_code[5])  # Middle-Index Spread
        d1 = int(dadadafafp_code[6])  # Index Flexion
        a0 = int(dadadafafp_code[7])  # Index-Thumb Abduction
        f  = int(dadadafafp_code[8])  # Thumb Opposition
        p  = int(dadadafafp_code[9])  # Thumb IP Flexion

        landmarks = np.zeros((21, 3), dtype=np.float64)
        landmarks[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Wrist (Root)

        # Set Metacarpal Bases (Landmarks 1, 5, 9, 13, 17)
        landmarks[1]  = self.palm_bases['Thumb']   # Thumb CMC
        landmarks[5]  = self.palm_bases['Index']   # Index MCP
        landmarks[9]  = self.palm_bases['Middle']  # Middle MCP
        landmarks[13] = self.palm_bases['Ring']    # Ring MCP
        landmarks[17] = self.palm_bases['Pinky']   # Pinky MCP

        # -------------------------------------------------------------------
        # 1. LONG FINGERS KINEMATICS (Index, Middle, Ring, Pinky)
        # -------------------------------------------------------------------
        long_fingers_cfg = [
            {
                'finger': 'Index',
                'mcp_idx': 5,
                'indices': [6, 7, 8],
                'stage': d1,
                'spread_yaw': self.spread_angles['Middle_Index'][a1],
                'lengths': self.phalanx_lengths['Index']
            },
            {
                'finger': 'Middle',
                'mcp_idx': 9,
                'indices': [10, 11, 12],
                'stage': d2,
                'spread_yaw': 0.0,  # Reference axis
                'lengths': self.phalanx_lengths['Middle']
            },
            {
                'finger': 'Ring',
                'mcp_idx': 13,
                'indices': [14, 15, 16],
                'stage': d3,
                'spread_yaw': self.spread_angles['Ring_Middle'][a2],
                'lengths': self.phalanx_lengths['Ring']
            },
            {
                'finger': 'Pinky',
                'mcp_idx': 17,
                'indices': [18, 19, 20],
                'stage': d4,
                'spread_yaw': self.spread_angles['Pinky_Ring'][a3],
                'lengths': self.phalanx_lengths['Pinky']
            }
        ]

        for cfg in long_fingers_cfg:
            finger = cfg['finger']
            mcp_idx = cfg['mcp_idx']
            pip_idx, dip_idx, tip_idx = cfg['indices']
            stage = cfg['stage']
            spread_yaw = cfg['spread_yaw']
            l1, l2, l3 = cfg['lengths']

            pitches = self.finger_flexion_stages[stage]
            j2_pitch = pitches['J2_Pitch']
            j3_pitch = pitches['J3_Pitch']
            j4_pitch = pitches['J4_Pitch']

            # Metacarpal Base Orientation
            base_yaw = self.metacarpal_base_angles[finger]
            r_base = rot_z(base_yaw)

            # Joint J1 (MCP Base Spread / Yaw)
            r_j1 = r_base.dot(rot_z(spread_yaw))

            # Joint J2 (MCP Flexion / Pitch) -> Computes PIP (Landmark idx_0)
            r_j2 = r_j1.dot(rot_x(j2_pitch))
            bone_1 = r_j2.dot(np.array([0.0, -l1, 0.0], dtype=np.float64))
            p_pip = landmarks[mcp_idx] + bone_1
            landmarks[pip_idx] = p_pip

            # Joint J3 (PIP Flexion / Pitch) -> Computes DIP (Landmark idx_1)
            r_j3 = r_j2.dot(rot_x(j3_pitch))
            bone_2 = r_j3.dot(np.array([0.0, -l2, 0.0], dtype=np.float64))
            p_dip = p_pip + bone_2
            landmarks[dip_idx] = p_dip

            # Joint J4 (DIP Flexion / Pitch) -> Computes TIP (Landmark idx_2)
            r_j4 = r_j3.dot(rot_x(j4_pitch))
            bone_3 = r_j4.dot(np.array([0.0, -l3, 0.0], dtype=np.float64))
            p_tip = p_dip + bone_3
            landmarks[tip_idx] = p_tip

        # -------------------------------------------------------------------
        # 2. THUMB KINEMATICS (Landmarks 0 -> 1(CMC) -> 2(MCP) -> 3(IP) -> 4(TIP))
        # -------------------------------------------------------------------
        l_mcp, l_ip, l_tip = self.phalanx_lengths['Thumb']
        spread_offset = self.spread_angles['Index_Thumb'][a0]
        base_thumb_yaw = self.metacarpal_base_angles['Thumb']  # -45.0°
        total_thumb_yaw = base_thumb_yaw + spread_offset

        tc = self.thumb_config
        j4_pitch = tc.get('f0_ip_flex', 65.0) if (p == 1 and f == 0) else (tc.get('f1_ip_flex', 65.0) if (p == 1 and f == 1) else 0.0)

        if f == 0:
            # F=0: Polegar no plano lateral da mão
            r_j1_thumb = rot_z(total_thumb_yaw).dot(rot_x(tc.get('f0_pitch', 5.0)))
            
            bone_thumb_1 = r_j1_thumb.dot(np.array([0.0, -l_mcp, 0.0], dtype=np.float64))
            p_thumb_mcp = landmarks[1] + bone_thumb_1
            landmarks[2] = p_thumb_mcp

            r_j2_thumb = r_j1_thumb.dot(rot_x(tc.get('f0_mcp_pitch', 5.0)))
            bone_thumb_2 = r_j2_thumb.dot(np.array([0.0, -l_ip, 0.0], dtype=np.float64))
            p_thumb_ip = p_thumb_mcp + bone_thumb_2
            landmarks[3] = p_thumb_ip

            r_j4_thumb = r_j2_thumb.dot(rot_x(j4_pitch))
            bone_thumb_3 = r_j4_thumb.dot(np.array([0.0, -l_tip, 0.0], dtype=np.float64))
            landmarks[4] = p_thumb_ip + bone_thumb_3

        else:
            # F=1: Oposição Transversal
            opp_yaw_dir = total_thumb_yaw + tc.get('f1_opp_yaw', 45.0)
            opp_roll = tc.get('f1_opp_roll', -40.0)
            opp_pitch = tc.get('f1_opp_pitch', 40.0)

            r_j1_thumb = (
                rot_z(opp_yaw_dir)
                .dot(rot_y(opp_roll))
                .dot(rot_x(opp_pitch))
            )

            bone_thumb_1 = r_j1_thumb.dot(np.array([0.0, -l_mcp, 0.0], dtype=np.float64))
            p_thumb_mcp = landmarks[1] + bone_thumb_1
            landmarks[2] = p_thumb_mcp

            # Em oposição, a articulação MCP flexiona para fechar o arco
            r_j2_thumb = r_j1_thumb.dot(rot_x(tc.get('f1_mcp_pitch', 45.0)))
            bone_thumb_2 = r_j2_thumb.dot(np.array([0.0, -l_ip, 0.0], dtype=np.float64))
            p_thumb_ip = p_thumb_mcp + bone_thumb_2
            landmarks[3] = p_thumb_ip

            r_j4_thumb = r_j2_thumb.dot(rot_x(j4_pitch))
            bone_thumb_3 = r_j4_thumb.dot(np.array([0.0, -l_tip, 0.0], dtype=np.float64))
            landmarks[4] = p_thumb_ip + bone_thumb_3

        return landmarks

    def generate_all_valid_seeds(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Generate all biomechanically valid seed configurations across the DADADADAFP taxonomy.
        Pruning invalid poses via is_valid_pose.

        Returns:
            Dict mapping 10-digit taxonomy keys to 21-landmark normalized coordinate dicts.
        """
        seeds: Dict[str, List[Dict[str, float]]] = {}
        valid_count = 0
        pruned_count = 0

        regular_states = [0, 1, 2, 3]

        for d4 in regular_states:
            for d3 in regular_states:
                a3_options = [0, 1] if (d4 <= 1 and d3 <= 1) else [1]
                for a3 in a3_options:

                    for d2 in regular_states:
                        a2_options = [0, 1] if (d3 <= 1 and d2 <= 1) else [1]
                        for a2 in a2_options:

                            for d1 in regular_states:
                                a1_options = [0, 1] if (d2 <= 1 and d1 <= 1) else [1]
                                for a1 in a1_options:

                                    for a0 in [0, 1]:
                                        for f in [0, 1]:
                                            for p in [0, 1]:
                                                code = f"{d4}{a3}{d3}{a2}{d2}{a1}{d1}{a0}{f}{p}"

                                                is_valid, _ = self.is_valid_pose(code)
                                                if not is_valid:
                                                    pruned_count += 1
                                                    continue

                                                lms_3d = self.build_landmarks_from_code(code)

                                                # Format to list of {"x": ..., "y": ..., "z": ...}
                                                seeds[code] = [
                                                    {
                                                        "x": float(round(pt[0], 6)),
                                                        "y": float(round(pt[1], 6)),
                                                        "z": float(round(pt[2], 6))
                                                    }
                                                    for pt in lms_3d
                                                ]
                                                valid_count += 1

        print(f"[HandKinematicsDirect] Generated {valid_count} valid seed poses (Pruned {pruned_count} impossible poses).")
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

    generator = HandKinematicsDirect()

    # Test basic poses
    test_poses = [
        ("0000000000", "Mão Aberta Total"),
        ("1111111100", "Mão Garra Leve (Stage 1)"),
        ("2121212100", "Mão Plataforma (Stage 2)"),
        ("3131313100", "Punho Fechado (Stage 3)"),
        ("3131313111", "Punho Fechado com Polegar Oposto e Ponta Flexionada (Sinal 'A')")
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
