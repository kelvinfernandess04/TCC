import os
import json
import glob
import numpy as np

BASE_DIR = r"C:\DevTools\Faculdade\TCC\Treinamento IA"
CAPTURES_DIR = os.path.join(BASE_DIR, 'data', 'captured_gestures')

def main():
    pattern = os.path.join(CAPTURES_DIR, 'captured_gestures_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print("Nenhuma captura encontrada!")
        return

    with open(files[0], 'r', encoding='utf-8') as f:
        frames = json.load(f)
    
    # Pegar o primeiro frame
    pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in frames[0]])
    rel = pts - pts[0]
    
    print("=== COORDENADAS RELATIVAS AO PULSO DO PRIMEIRO FRAME CAPTURADO ===")
    fingers = {
        'Wrist': [0],
        'Thumb': [1, 2, 3, 4],
        'Index': [5, 6, 7, 8],
        'Middle': [9, 10, 11, 12],
        'Ring': [13, 14, 15, 16],
        'Pinky': [17, 18, 19, 20]
    }
    for name, indices in fingers.items():
        print(f"\n{name}:")
        for idx in indices:
            print(f"  Landmark {idx}: [{rel[idx,0]:.4f}, {rel[idx,1]:.4f}, {rel[idx,2]:.4f}]")

if __name__ == "__main__":
    main()
