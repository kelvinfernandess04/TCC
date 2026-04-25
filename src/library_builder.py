import cv2
import mediapipe as mp
import numpy as np
import os
import json
import glob

# Configurações Anatômicas (v10.31)
FINGER_JOINTS = [
    (1, 2, 3, 4),    # Thumb
    (5, 6, 7, 8),    # Index
    (9, 10, 11, 12), # Middle
    (13, 14, 15, 16),# Ring
    (17, 18, 19, 20)  # Pinky
]

# Travas Fisiológicas (Limites da Anatomia Humana)
LIMITS = {
    "articular": (0.0, 120.0), # Flexão máxima natural
    "abduction": (0.0, 65.0)   # Abertura lateral máxima
}

def calculate_angle_3d(a, b, c):
    """Calcula o ângulo de flexão (0 = Reto, 90+ = Flexionado)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6: return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    # Alinhamento v10.31: 0 é reto
    return abs(180.0 - angle)

def extract_hand_angles(image_path, landmarker):
    """Processa uma imagem e retorna o vetor de 19 ângulos ou None se falhar/inválido"""
    image = cv2.imread(image_path)
    if image is None: return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = landmarker.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Pegamos a primeira mão detectada
    hlms = results.multi_hand_landmarks[0]
    h_arr = np.array([[lm.x, lm.y, lm.z] for lm in hlms.landmark])
    
    angles = []
    # 1. Articulares (15)
    for finger_idx, (mcp, pip, dip, tip) in enumerate(FINGER_JOINTS):
        pts = [h_arr[0], h_arr[mcp], h_arr[pip], h_arr[dip], h_arr[tip]]
        a1 = calculate_angle_3d(pts[0], pts[1], pts[2]) # MCP
        a2 = calculate_angle_3d(pts[1], pts[2], pts[3]) # PIP
        a3 = calculate_angle_3d(pts[2], pts[3], pts[4]) # DIP
        
        # Aplicação da Trava Fisiológica Individual
        for a in [a1, a2, a3]:
            if a < LIMITS["articular"][0] - 10 or a > LIMITS["articular"][1] + 10:
                print(f"  [VETO] Ângulo articular irreal detectado: {a:.1f}°")
                return None
            angles.append(float(np.clip(a, LIMITS["articular"][0], LIMITS["articular"][1])))

    # 2. Abdução (4)
    for i in range(4):
        v1 = h_arr[FINGER_JOINTS[i][0]] - h_arr[0]
        v2 = h_arr[FINGER_JOINTS[i+1][0]] - h_arr[0]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            abd = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
        else:
            abd = 0.0
            
        if abd > LIMITS["abduction"][1] + 15:
            print(f"  [VETO] Abdução irreal detectada: {abd:.1f}°")
            return None
        angles.append(float(np.clip(abd, LIMITS["abduction"][0], LIMITS["abduction"][1])))
        
    return angles

def build_library(root_dir):
    print(f"=== LIBRARY BUILDER v1.0 (Phase 6) ===")
    print(f"Diretório Raiz: {root_dir}")
    
    mp_hands = mp.solutions.hands
    new_library = {}
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        # Cada subpasta é uma classe
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for cls_name in classes:
            print(f"\nProcessando Classe: {cls_name}")
            cls_path = os.path.join(root_dir, cls_name)
            images = glob.glob(os.path.join(cls_path, "*.*"))
            
            vectors = []
            for img_p in images:
                vec = extract_hand_angles(img_p, hands)
                if vec:
                    vectors.append(vec)
                    print(f"  [OK] {os.path.basename(img_p)}")
                else:
                    print(f"  [SKIP] {os.path.basename(img_p)}")
            
            if vectors:
                centroid = np.mean(vectors, axis=0)
                new_library[cls_name] = {
                    "articular": [round(float(x), 1) for x in centroid[:15]],
                    "abduction": [round(float(x), 1) for x in centroid[15:]]
                }
                print(f"  => Centroide gerado com {len(vectors)} amostras.")
            else:
                print(f"  => ERRO: Nenhuma amostra válida para {cls_name}")
                
    return new_library

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="images/library_source", help="Pasta com subpastas por classe")
    parser.add_argument("--out", default="new_library.json", help="Arquivo de saída")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
        print(f"Diretório '{args.dir}' criado. Adicione subpastas com imagens e rode novamente.")
    else:
        lib = build_library(args.dir)
        if lib:
            with open(args.out, "w", encoding='utf-8') as f:
                json.dump(lib, f, indent=4)
            print(f"\nBIBLIOTECA GERADA: {args.out}")
            print("\nCopie os valores para o comparador_v10.py conforme necessário.")
