import os
import json
import glob
import numpy as np

BASE_DIR = r"C:\DevTools\Faculdade\TCC\Treinamento IA"
CAPTURES_DIR = os.path.join(BASE_DIR, 'data', 'captured_gestures')

def vec_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

def main():
    pattern = os.path.join(CAPTURES_DIR, 'captured_gestures_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print("Nenhuma captura encontrada!")
        return

    frames_open = []
    frames_closed = []

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            frames = json.load(f)
        for frame in frames:
            if len(frame) == 21:
                pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in frame])
                # Calcular flexão do polegar no MCP (triplet 1, 2, 3)
                v1 = pts[1] - pts[2]
                v2 = pts[3] - pts[2]
                flex = 180.0 - vec_angle(v1, v2)
                
                if flex < 20.0:
                    frames_open.append(pts)
                elif flex > 50.0:
                    frames_closed.append(pts)

    print(f"Frames com Polegar Aberto (flex < 20°): {len(frames_open)}")
    print(f"Frames com Polegar Fechado (flex > 50°): {len(frames_closed)}")

    if frames_open:
        avg_open = np.mean(np.array(frames_open) - np.array(frames_open)[:, 0:1, :], axis=0)
        print("\n=== COORDENADAS REAIS MÉDIAS - POLEGAR ABERTO ===")
        for i in range(5):
            print(f"Landmark {i}: [{avg_open[i,0]:.4f}, {avg_open[i,1]:.4f}, {avg_open[i,2]:.4f}]")

    if frames_closed:
        avg_closed = np.mean(np.array(frames_closed) - np.array(frames_closed)[:, 0:1, :], axis=0)
        print("\n=== COORDENADAS REAIS MÉDIAS - POLEGAR FECHADO ===")
        for i in range(5):
            print(f"Landmark {i}: [{avg_closed[i,0]:.4f}, {avg_closed[i,1]:.4f}, {avg_closed[i,2]:.4f}]")

if __name__ == "__main__":
    main()
