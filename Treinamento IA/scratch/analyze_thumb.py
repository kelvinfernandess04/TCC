import os
import json
import glob
import numpy as np

BASE_DIR = r"C:\DevTools\Faculdade\TCC\Treinamento IA"
CAPTURES_DIR = os.path.join(BASE_DIR, 'data', 'captured_gestures')
SEEDS_FILE = os.path.join(BASE_DIR, 'data', 'seeds', 'seeds.json')

def main():
    # 1. Load some frames from raw captures
    pattern = os.path.join(CAPTURES_DIR, 'captured_gestures_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print("Nenhuma captura encontrada!")
        return

    print(f"Lendo {len(files)} arquivos de captura...")
    captured_thumbs = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            frames = json.load(f)
        for frame in frames:
            if len(frame) == 21:
                pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in frame])
                captured_thumbs.append(pts[0:5]) # Wrist + thumb joints

    captured_thumbs = np.array(captured_thumbs)
    print(f"Total de frames capturados analisados: {len(captured_thumbs)}")

    # Calcular médias das posições relativas ao wrist (0)
    rel_captured = captured_thumbs - captured_thumbs[:, 0:1, :]
    avg_captured = np.mean(rel_captured, axis=0)

    print("\n=== POSIÇÕES MÉDIAS DO POLEGAR NAS CAPTURAS (Relativo ao Pulso) ===")
    for i in range(5):
        print(f"Landmark {i}: [{avg_captured[i,0]:.4f}, {avg_captured[i,1]:.4f}, {avg_captured[i,2]:.4f}]")

    # 2. Load generated seeds
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
            seeds = json.load(f)
        
        # Vamos pegar algumas labels de polegar aberto/fechado
        labels = list(seeds.keys())
        print(f"\nTotal de labels no seeds.json: {len(labels)}")
        
        # Exemplo de polegar aberto (Pol:Aberto, ex: final 00)
        open_labels = [l for l in labels if l.endswith("00")]
        # Exemplo de polegar fechado (Pol:Fechado, ex: final 13)
        closed_labels = [l for l in labels if l.endswith("13")]

        if open_labels:
            lbl = open_labels[0]
            pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in seeds[lbl]])
            pts_rel = pts - pts[0]
            print(f"\n=== POLEGAR GERADO SEMENTE (ABERTO: {lbl}) ===")
            for i in range(5):
                print(f"Landmark {i}: [{pts_rel[i,0]:.4f}, {pts_rel[i,1]:.4f}, {pts_rel[i,2]:.4f}]")

        if closed_labels:
            lbl = closed_labels[0]
            pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in seeds[lbl]])
            pts_rel = pts - pts[0]
            print(f"\n=== POLEGAR GERADO SEMENTE (FECHADO: {lbl}) ===")
            for i in range(5):
                print(f"Landmark {i}: [{pts_rel[i,0]:.4f}, {pts_rel[i,1]:.4f}, {pts_rel[i,2]:.4f}]")

if __name__ == "__main__":
    main()
