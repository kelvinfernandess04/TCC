import os
import json
import cv2
import numpy as np

# Configurações de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_FILE = os.path.join(BASE_DIR, 'data', 'datasets', 'synthetic_dataset', 'synthetic_data.json')

# Definição das conexões da mão (MediaPipe padrão)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (0, 17), (17, 18), (18, 19), (20, 19), # Mínimo
    (5, 9), (9, 13), (13, 17)              # Palma
]

def draw_hand(img, landmarks):
    h, w, _ = img.shape
    # Converter normalizado para pixels
    pts = []
    for lm in landmarks:
        pts.append((int(lm[0] * w), int(lm[1] * h)))
    
    # Desenhar conexões
    for start, end in CONNECTIONS:
        cv2.line(img, pts[start], pts[end], (0, 255, 0), 2)
    
    # Desenhar pontos
    for pt in pts:
        cv2.circle(img, pt, 4, (0, 0, 255), -1)

def main():
    if not os.path.exists(SYNTHETIC_FILE):
        print(f"Erro: Arquivo não encontrado em {SYNTHETIC_FILE}")
        return

    print("Carregando base de dados... (isso pode levar alguns segundos)")
    with open(SYNTHETIC_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get("frames", [])
    if not frames:
        print("Base de dados vazia.")
        return

    # Agrupar frames por label para navegação inteligente
    grouped_data = {}
    for frame in frames:
        lbl = frame['label']
        if lbl not in grouped_data:
            grouped_data[lbl] = []
        grouped_data[lbl].append(frame['landmarks'])
    
    labels = sorted(list(grouped_data.keys()))
    current_label_idx = 0
    current_sample_idx = 0

    cv2.namedWindow("Visualizador Interativo TCC", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visualizador Interativo TCC", 600, 600)

    print("\n--- CONTROLES ---")
    print("[D] ou [Seta Direita]: Proxima Amostra")
    print("[A] ou [Seta Esquerda]: Amostra Anterior")
    print("[W] ou [Seta Cima]: Proxima Label (Configuração)")
    print("[S] ou [Seta Baixo]: Label Anterior")
    print("[Q]: Sair")

    while True:
        label = labels[current_label_idx]
        samples = grouped_data[label]
        landmarks = samples[current_sample_idx]

        # Criar tela preta
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        
        # Info de Texto
        cv2.putText(canvas, f"Label: {label} ({current_label_idx + 1}/{len(labels)})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Amostra: {current_sample_idx + 1}/{len(samples)}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Desenhar esqueleto
        draw_hand(canvas, landmarks)

        cv2.imshow("Visualizador Interativo TCC", canvas)
        
        key = cv2.waitKeyEx(0)

        # Tecla Q ou ESC para sair
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        
        # Navegação de Amostras (A/D ou Setas)
        elif key == ord('d') or key == ord('D') or key == 2555904: # 2555904 é seta direita no Windows
            current_sample_idx = (current_sample_idx + 1) % len(samples)
        elif key == ord('a') or key == ord('A') or key == 2424832: # 2424832 é seta esquerda
            current_sample_idx = (current_sample_idx - 1) % len(samples)
            
        # Navegação de Labels (W/S ou Setas)
        elif key == ord('w') or key == ord('W') or key == 2490368: # Seta Cima
            current_label_idx = (current_label_idx + 1) % len(labels)
            current_sample_idx = 0
        elif key == ord('s') or key == ord('S') or key == 2621440: # Seta Baixo
            current_label_idx = (current_label_idx - 1) % len(labels)
            current_sample_idx = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
