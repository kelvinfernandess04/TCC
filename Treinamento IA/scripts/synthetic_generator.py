import os
import json
import math
import numpy as np
import random
import itertools
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'datasets', 'synthetic_dataset')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'synthetic_data.json')

# ---------------------------------------------------------
# CONSTANTES BIOMECÂNICAS E ANTROPOMÉTRICAS
# ---------------------------------------------------------

PALM_BASES = {
    'Thumb':  np.array([-0.10, 0.15, 0.0]), # Reajustado próximo ao indicador para o Ângulo 01-05
    'Index':  np.array([-0.08, 0.45, 0.0]),
    'Middle': np.array([ 0.00, 0.48, 0.0]),
    'Ring':   np.array([ 0.08, 0.45, 0.0]),
    'Pinky':  np.array([ 0.16, 0.38, 0.0])
}

DEFAULT_YAW = {
    'Index': 5.0,    # Aponta levemente esquerda (+)
    'Middle': 0.0,
    'Ring': -5.0,    # Aponta levemente direita (-)
    'Pinky': -15.0
}

BONE_LENGTHS = {
    'Thumb':  [0.14, 0.10, 0.08], # Reduzido substancialmente para corrigir proporção
    'Index':  [0.22, 0.14, 0.10],
    'Middle': [0.24, 0.15, 0.11],
    'Ring':   [0.23, 0.14, 0.10],
    'Pinky':  [0.17, 0.10, 0.08]
}

ROM_LIMITS = {
    'Thumb':  [(0, 45), (0, 55), (0, 80)],
    'Index':  [(-10, 90), (0, 100), (0, 80)],
    'Middle': [(-10, 90), (0, 100), (0, 80)],
    'Ring':   [(-10, 90), (0, 100), (0, 80)],
    'Pinky':  [(-10, 90), (0, 100), (0, 80)]
}

# ---------------------------------------------------------
# MATRIZES DE ROTAÇÃO PURA (EULER)
# ---------------------------------------------------------

def rot_x(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

def rot_y(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

def rot_z(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

def calc_finger_chain(base_pos, base_rot_mat, lengths, pitches):
    points = [base_pos]
    current_mat = base_rot_mat
    current_pos = base_pos
    
    for length, pitch in zip(lengths, pitches):
        joint_rot = rot_x(pitch) # Flexão ocorre estritamente no eixo local X
        current_mat = current_mat.dot(joint_rot)
        
        local_bone = np.array([0, length, 0]) # Osso cresce no eixo local Y
        world_bone = current_mat.dot(local_bone)
        current_pos = current_pos + world_bone
        points.append(current_pos.copy())
        
    return points

def generate_hand_3d(state_tuple, g_im):
    """
    Geração estática padronizada com Arquétipos Posturais Biomecânicos.
    """
    landmarks_3d = [np.array([0.0, 0.0, 0.0])]
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    for i, finger in enumerate(fingers):
        state = state_tuple[i]
        
        if finger == 'Thumb':
            if state == 0:
                # Estado 0 (Aberto / Abduzido): Projeta o polegar para fora e levemente para frente
                base_rot_mat = rot_y(40).dot(rot_z(35)).dot(rot_x(-20))
                pitches = [0.0, 0.0, 0.0]
            elif state == 1:
                # Estado 1 (Aduto Transversal): Alinha o polegar paralelamente à lateral da palma
                base_rot_mat = rot_y(70).dot(rot_z(15)).dot(rot_x(10))
                pitches = [5.0, 5.0, 0.0]
            elif state == 2:
                # Estado 2 (Oposição Plena): Metacarpo rotaciona pra dentro e articulações flexionam em espiral
                base_rot_mat = rot_y(85).dot(rot_z(5)).dot(rot_x(35))
                pitches = [15.0, 35.0, 55.0]
            else: # state == 3
                # Estado 3 (Gatilho): Base evertida com flexão isolada das falanges distal e medial
                base_rot_mat = rot_y(40).dot(rot_z(35)).dot(rot_x(-20))
                pitches = [0.0, 30.0, 65.0]
        else:
            if state == 0: pitches = [0.0, 0.0, 0.0]
            elif state == 1: pitches = [10.0, 80.0, 53.0] # Hook/Garra
            elif state == 2: pitches = [90.0, 100.0, 67.0] # Fechado total
            else: pitches = [90.0, 0.0, 0.0] # state == 3 (Plataforma/Teto)
            
            base_yaw = DEFAULT_YAW[finger]
            if finger == 'Index':
                if g_im == 0: base_yaw = 0.0 # Perfeitamente paralelo
                elif g_im == 2: base_yaw = 15.0 # Vão aberto
                else: base_yaw = 5.0 # Repouso natural
            elif finger == 'Middle':
                if g_im == 0: base_yaw = 0.0 # Perfeitamente paralelo
                elif g_im == 2: base_yaw = -15.0 # Vão aberto
                else: base_yaw = -5.0 # Repouso natural
            base_rot_mat = rot_z(base_yaw)
            
        lengths = BONE_LENGTHS[finger]
        base_pos = PALM_BASES[finger]
        
        chain = calc_finger_chain(base_pos, base_rot_mat, lengths, pitches)
        landmarks_3d.extend(chain)
        
    return landmarks_3d

# ---------------------------------------------------------
# TRANSFORMAÇÃO E PROJEÇÃO 2D
# ---------------------------------------------------------

def apply_global_transform(landmarks_3d, pitch_deg, yaw_deg, roll_deg):
    """
    Transformação espacial 3D absoluta usando matrizes puras.
    """
    Rx = rot_x(pitch_deg)
    Ry = rot_y(yaw_deg)
    Rz = rot_z(roll_deg)
    
    R = Rz.dot(Ry).dot(Rx)
    
    pts_2d = []
    for pt in landmarks_3d:
        pt_rot = R.dot(pt)
        # Projeção com perspectiva suave (focal distante) para dar noção 3D sem explodir
        z_offset = 4.0
        z_factor = z_offset / (z_offset - pt_rot[2])
        x2d = pt_rot[0] * z_factor
        y2d = pt_rot[1] * z_factor
        pts_2d.append([x2d, y2d])
        
    return pts_2d

def bounce_wave(progress, cycles):
    """
    Onda triangular linear. Vai de -1 a 1 e volta a -1 com velocidade constante.
    Evita a densidade extrema nas bordas que a função Sine (Seno) causa.
    """
    cycle_pos = (progress * cycles) % 1.0
    if cycle_pos < 0.5:
        return (cycle_pos * 4.0) - 1.0
    else:
        return 1.0 - ((cycle_pos - 0.5) * 4.0)

def normalize_and_add_noise(pts_2d):
    xs = [p[0] for p in pts_2d]
    ys = [p[1] for p in pts_2d]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    size = max(width, height)
    
    normalized = []
    for x, y in pts_2d:
        nx = (x - min_x) / size
        ny = (y - min_y) / size
        # Ruído de sensor mediapipe mantido apenas no 2D final para robustez (um pouco mais alto)
        nx += random.gauss(0, 0.005)
        ny += random.gauss(0, 0.005)
        normalized.append([nx, ny])
        
    return normalized

def check_self_collision(landmarks_3d):
    check_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
    collision_threshold = 0.02 # Threshold reduzido pois não há distorção de projeção
    
    for i in range(len(check_ids)):
        for j in range(i + 1, len(check_ids)):
            id1 = check_ids[i]
            id2 = check_ids[j]
            
            if (id1 == 3 and id2 == 4) or \
               (id1 == 7 and id2 == 8) or \
               (id1 == 11 and id2 == 12) or \
               (id1 == 15 and id2 == 16) or \
               (id1 == 19 and id2 == 20):
                continue
                
            p1 = landmarks_3d[id1]
            p2 = landmarks_3d[id2]
            dist = np.linalg.norm(p1 - p2)
            if dist < collision_threshold:
                return True
    return False

# ---------------------------------------------------------
# LOOP PRINCIPAL DE GERAÇÃO
# ---------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.info("--- Iniciando Geração Sintética de LIBRAS (Varredura Contínua 3D) ---")
    
    states_raw = list(itertools.product([0, 1, 2, 3], repeat=5))
    states = []
    
    # Filtro Connexus Intertendinei (Exclusão Biológica)
    for st in states_raw:
        thumb, index, middle, ring, pinky = st
        # Bloqueia a extensão do anelar apenas se ambos os vizinhos estiverem 100% flexionados
        if middle == 2 and pinky == 2 and ring == 0: continue
        # Impede isolamento perfeito do anelar flexionado enquanto a mão está totalmente aberta
        if ring == 2 and index == 0 and middle == 0 and pinky == 0: continue
        states.append(st)
        
    # Calcula número total de classes reais após podas anatômicas
    total_classes = 0
    for st in states:
        if st[1] != 2 and st[2] != 2: total_classes += 3
        else: total_classes += 1
        
    SAMPLES_PER_STATE = 1800
    total_generated = 0
    discarded_collisions = 0
    
    # Os dados serão salvos em pastas particionadas por label para economizar memória
    
    for state_tuple in states:
        # PODA ANATÔMICA: Ligamentos Colaterais
        # Se os dedos estiverem completamente flexionados (2), é mecanicamente impossível ter Gap.
        # Também, por oclusão de visão computacional, o gap interno de um soco é irrelevante.
        if state_tuple[1] != 2 and state_tuple[2] != 2:
            current_gap_states = [0, 1, 2] # Variamos o Gap para permitir diferenciação de letras (ex: U e V)
        else:
            current_gap_states = [1] # Força estado Neutro (Colado/Relaxado) e evita multiplicador inútil
            
        for g_im in current_gap_states:
            # Nova Nomenclatura com Sufixo de Gap (ex: S_20022_0 = U, S_20022_2 = V)
            label = f"S_{state_tuple[0]}{state_tuple[1]}{state_tuple[2]}{state_tuple[3]}{state_tuple[4]}_{g_im}"
            
            # Geração Padrão da Mão para esta classe
            lms_3d = generate_hand_3d(state_tuple, g_im)
            
            # Validação Antecipada de Colisão Biomecânica da Classe
            if check_self_collision(lms_3d):
                discarded_collisions += SAMPLES_PER_STATE
                continue
            
            label_dataset = {
                "metadata": {
                    "label": label,
                    "samples": SAMPLES_PER_STATE
                },
                "frames": []
            }
            
            valid_samples = 0
            while valid_samples < SAMPLES_PER_STATE:
                progress = valid_samples / float(SAMPLES_PER_STATE)
                
                # Varredura Contínua 3D (Onda Linear / Triangle Wave)
                # O limite não é mais 90 graus (perfil perfeito), pois na LIBRAS raramente 
                # a mão fica em 90 graus absolutos (esconde os dedos).
                
                # Restringe o perfil a 65 graus para evitar colapso e ambiguidade tridimensional na projeção 2D
                target_pitch = bounce_wave(progress, 1) * 65.0
                target_yaw = bounce_wave(progress, 2) * 65.0
                
                # Roll (Giro do Pulso): Rotação contínua para cobrir 2 voltas na esfera
                target_roll = progress * 360.0 * 2
                
                lms_2d = apply_global_transform(lms_3d, target_pitch, target_yaw, target_roll)
                lms_final = normalize_and_add_noise(lms_2d)
                
                label_dataset["frames"].append({
                    "label": label,
                    "landmarks": lms_final
                })
                valid_samples += 1
                total_generated += 1
                
            label_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)
            label_file = os.path.join(label_dir, "data.json")
            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump(label_dataset, f, separators=(',', ':'))
                
    logging.info(f"Geração concluída! Total gerado: {total_generated}. Classes rejeitadas (Colisão): {discarded_collisions}")
    logging.info(f"Datasets particionados salvos em subpastas dentro de: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
