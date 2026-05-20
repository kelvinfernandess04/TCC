import os
import json
import cv2
import numpy as np

# Configurações de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_FILE = os.path.join(BASE_DIR, 'data', 'seeds', 'seeds.json')
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'data', 'datasets', 'synthetic_dataset')

# Definição das conexões da mão (MediaPipe padrão)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # Mínimo
    (5, 9), (9, 13), (13, 17)              # Palma
]

# Cores por dedo (BGR)
FINGER_COLORS = {
    'Thumb':  (0, 200, 255),   # Laranja
    'Index':  (0, 255, 0),     # Verde
    'Middle': (255, 200, 0),   # Azul claro
    'Ring':   (255, 0, 150),   # Roxo
    'Pinky':  (100, 100, 255), # Vermelho claro
    'Palm':   (150, 150, 150)  # Cinza
}

def get_connection_color(start, end):
    """Retorna a cor baseada no dedo da conexão."""
    if start in [0,1,2,3] and end in [0,1,2,3,4]:
        return FINGER_COLORS['Thumb']
    elif start in [0,5,6,7] and end in [0,5,6,7,8]:
        return FINGER_COLORS['Index']
    elif start in [0,9,10,11] and end in [0,9,10,11,12]:
        return FINGER_COLORS['Middle']
    elif start in [0,13,14,15] and end in [0,13,14,15,16]:
        return FINGER_COLORS['Ring']
    elif start in [0,17,18,19] and end in [0,17,18,19,20]:
        return FINGER_COLORS['Pinky']
    return FINGER_COLORS['Palm']

def normalize_3d_to_2d(landmarks_3d):
    """Projeta landmarks 3D em 2D normalizado [0,1] para visualização."""
    pts = np.array([[l['x'], l['y'], l['z']] for l in landmarks_3d])
    # Centralizar no pulso
    pts = pts - pts[0]
    # Projeção ortográfica simples (X, Y)
    xs = pts[:, 0]
    ys = pts[:, 1]
    # Normalizar para [0.1, 0.9] com margem
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    size = max(max_x - min_x, max_y - min_y, 1e-6)
    norm = []
    for x, y in zip(xs, ys):
        nx = 0.1 + 0.8 * (x - min_x) / size
        ny = 0.1 + 0.8 * (y - min_y) / size
        norm.append([nx, ny])
    return norm

def decode_label(label):
    """Decodifica um label XAXAXAXAAX em descrição legível."""
    if len(label) != 10:
        return label

    state_names = {0: 'Aberto', 1: 'Meio', 2: 'Garra', 3: 'Fechado'}
    spread_names = {0: 'Junto', 1: 'Aberto'}

    pinky  = int(label[0])
    pr_spr = int(label[1])
    ring   = int(label[2])
    rm_spr = int(label[3])
    middle = int(label[4])
    mi_spr = int(label[5])
    index  = int(label[6])
    it_spr = int(label[7])
    th_opp = int(label[8])
    thumb  = int(label[9])

    parts = [
        f"Min:{state_names.get(pinky, '?')}",
        f"An:{state_names.get(ring, '?')}",
        f"Med:{state_names.get(middle, '?')}",
        f"Ind:{state_names.get(index, '?')}",
        f"Pol:{state_names.get(thumb, '?')}",
    ]
    spreads = []
    if pr_spr: spreads.append("Min-An")
    if rm_spr: spreads.append("An-Med")
    if mi_spr: spreads.append("Med-Ind")
    if it_spr: spreads.append("Ind-Pol")
    if th_opp: spreads.append("Oposicao")

    desc = " | ".join(parts)
    if spreads:
        desc += f"  [Aberturas: {', '.join(spreads)}]"
    return desc

def draw_hand(img, landmarks_2d):
    """Desenha o esqueleto da mão no canvas."""
    h, w, _ = img.shape
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks_2d]

    # Desenhar conexões com cores por dedo
    for start, end in CONNECTIONS:
        color = get_connection_color(start, end)
        cv2.line(img, pts[start], pts[end], color, 2, cv2.LINE_AA)

    # Desenhar pontos
    for i, pt in enumerate(pts):
        color = (255, 255, 255)
        if i == 0:
            color = (0, 255, 255)  # Pulso em amarelo
        cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)

def load_synthetic_samples(label):
    """Carrega dinamicamente os 1800 frames sintéticos do label."""
    path = os.path.join(SYNTHETIC_DIR, label, 'data.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("frames", [])
        except Exception as e:
            print(f"Erro ao ler {path}: {e}")
    return []

def main():
    if not os.path.exists(SEEDS_FILE):
        print(f"Erro: seeds.json não encontrado em {SEEDS_FILE}")
        print("Execute seed_extractor.py primeiro.")
        return

    print("Carregando sementes...")
    with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
        seeds = json.load(f)

    labels = sorted([k for k in seeds.keys() if not k.startswith("__")])
    total = len(labels)
    current_idx = 0

    # Estados do visualizador
    view_mode = "samples"  # "seed" para semente estática 3D, "samples" para rotações sintéticas
    current_sample_idx = 0
    loaded_samples = []
    last_loaded_label = ""

    auto_play = False
    auto_delay = 5  # Autoplay rápido para rotações (em milissegundos)

    WINDOW_NAME = "Visualizador de Sementes e Rotacoes LIBRAS"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 700, 700)

    print(f"\nCarregadas {total} classes de sementes.")
    print("\n" + "=" * 60)
    print("  CONTROLES GERAIS")
    print("=" * 60)
    print("[M]                  : Alterna modo (SEMENTE ESTATICA / ROTACOES SINTETICAS)")
    print("[Espaco]             : Iniciar/Pausar autoplay de rotações")
    print("[G]                  : Ir para label específica")
    print("[Q] / [ESC]          : Sair")
    print("\n" + "-" * 60)
    print("  CONTROLES DE SELECAO DE CLASSE (LABEL)")
    print("-" * 60)
    print("[W] / [Seta Cima]    : Avança classe (+1)")
    print("[S] / [Seta Baixo]   : Recua classe (-1)")
    print("[Home] / [End]       : Primeira / última classe")
    print("\n" + "-" * 60)
    print("  CONTROLES DE AMOSTRA (ROTACAO E PERSPECTIVA)")
    print("-" * 60)
    print("[D] / [Seta Direita] : Avança rotação (+1)")
    print("[A] / [Seta Esquerda]: Recua rotação (-1)")
    print("[Page Up] / [Page Dn]: Salta 100 rotações à frente / atrás")
    print("[+] / [-]            : Ajusta velocidade do autoplay")
    print("=" * 60)

    while True:
        label = labels[current_idx]

        # Carregar dinamicamente as amostras sintéticas da classe se mudou de label
        if view_mode == "samples" and last_loaded_label != label:
            loaded_samples = load_synthetic_samples(label)
            last_loaded_label = label
            # Garantir índice dentro dos limites
            if loaded_samples:
                current_sample_idx = min(current_sample_idx, len(loaded_samples) - 1)
            else:
                current_sample_idx = 0

        # Selecionar o landmark 2D correspondente ao modo ativo
        if view_mode == "samples" and loaded_samples:
            frame = loaded_samples[current_sample_idx]
            lms_2d = frame["landmarks"]
            mode_text = f"ROTACAO SINTETICA (Amostra {current_sample_idx + 1}/{len(loaded_samples)})"
            text_color = (0, 165, 255)  # Laranja
        else:
            # Fallback para visualização da semente estática
            lms_3d = seeds[label]
            lms_2d = normalize_3d_to_2d(lms_3d)
            mode_text = "SEMENTE ORIGINAL ESTATICA (Base)"
            text_color = (0, 255, 200)  # Verde Claro

        # Canvas de fundo
        canvas = np.zeros((700, 700, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Fundo cinza escuro

        # Header
        header_y = 35
        cv2.putText(canvas, f"Label: {label}", (20, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Classe: {current_idx + 1} / {total}", (500, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Modo ativo
        cv2.putText(canvas, mode_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Descrição decodificada anatômica
        desc = decode_label(label)
        cv2.putText(canvas, desc, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

        # Desenhar status do Autoplay
        if auto_play and view_mode == "samples":
            cv2.putText(canvas, f"PLAYING ({auto_delay}ms)", (560, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Barra de progresso da amostra (se estiver no modo de rotações)
        if view_mode == "samples" and loaded_samples:
            bar_y = 680
            bar_w = int(660 * (current_sample_idx / max(len(loaded_samples) - 1, 1)))
            cv2.rectangle(canvas, (20, bar_y), (680, bar_y + 8), (60, 60, 60), -1)
            cv2.rectangle(canvas, (20, bar_y), (20 + bar_w, bar_y + 8), (0, 165, 255), -1)

        # Desenhar a mão centralizada
        hand_roi = canvas[95:670, 20:680]
        draw_hand(hand_roi, lms_2d)

        cv2.imshow(WINDOW_NAME, canvas)

        wait_time = auto_delay if (auto_play and view_mode == "samples" and loaded_samples) else 0
        key = cv2.waitKeyEx(wait_time)

        # Avanço do autoplay
        if auto_play and view_mode == "samples" and loaded_samples and key == -1:
            current_sample_idx = (current_sample_idx + 1) % len(loaded_samples)
            continue

        # Sair
        if key == ord('q') or key == ord('Q') or key == 27:
            break

        # Alternar modo (Semente Estática / Amostras)
        elif key == ord('m') or key == ord('M'):
            if view_mode == "seed":
                view_mode = "samples"
                last_loaded_label = ""  # Forçar recarga
            else:
                view_mode = "seed"
            auto_play = False

        # Navegação de classes (W/S ou Cima/Baixo)
        elif key == ord('w') or key == ord('W') or key == 2490368:
            current_idx = (current_idx + 1) % total
            current_sample_idx = 0
        elif key == ord('s') or key == ord('S') or key == 2621440:
            current_idx = (current_idx - 1) % total
            current_sample_idx = 0

        # Navegação de amostras / rotações (D/A ou Direita/Esquerda)
        elif key == ord('d') or key == ord('D') or key == 2555904:
            if view_mode == "samples" and loaded_samples:
                current_sample_idx = (current_sample_idx + 1) % len(loaded_samples)
            else:
                current_idx = (current_idx + 1) % total
        elif key == ord('a') or key == ord('A') or key == 2424832:
            if view_mode == "samples" and loaded_samples:
                current_sample_idx = (current_sample_idx - 1) % len(loaded_samples)
            else:
                current_idx = (current_idx - 1) % total

        # Salto de amostras / Page Up e Down (100 frames)
        elif key == 2162688:  # Page Up
            if view_mode == "samples" and loaded_samples:
                current_sample_idx = min(current_sample_idx + 100, len(loaded_samples) - 1)
        elif key == 2228224:  # Page Down
            if view_mode == "samples" and loaded_samples:
                current_sample_idx = max(current_sample_idx - 100, 0)

        # Home / End (Primeira/Última classe)
        elif key == 2359296:  # Home
            current_idx = 0
            current_sample_idx = 0
        elif key == 2293760:  # End
            current_idx = total - 1
            current_sample_idx = 0

        # Iniciar/Pausar autoplay de rotações (Espaço)
        elif key == 32:
            if view_mode == "samples" and loaded_samples:
                auto_play = not auto_play

        # Ajuste de velocidade do autoplay (+ / -)
        elif key == ord('+') or key == ord('='):
            auto_delay = max(1, auto_delay - 5)
        elif key == ord('-') or key == ord('_'):
            auto_delay = min(500, auto_delay + 5)

        # Ir para label específica
        elif key == ord('g') or key == ord('G'):
            auto_play = False
            target = input("Digite a label (ex: 0010010000): ").strip()
            if target in seeds:
                current_idx = labels.index(target)
                current_sample_idx = 0
                last_loaded_label = ""  # Forçar recarga
                print(f"Navegado para classe: {target}")
            else:
                print(f"Classe '{target}' não encontrada no banco.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
