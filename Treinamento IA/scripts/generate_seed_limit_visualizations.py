import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEEDS_DIR = os.path.join(DATA_DIR, 'seeds')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'seed_verification')

os.makedirs(REPORTS_DIR, exist_ok=True)

# Color palette per finger segment
FINGER_COLORS = {
    'Thumb':  '#FF5722',  # Orange-Red
    'Index':  '#FFEB3B',  # Yellow
    'Middle': '#4CAF50',  # Green
    'Ring':   '#00BCD4',  # Cyan
    'Pinky':  '#9C27B0'   # Purple
}

def project_seed_to_2d(pts_seed):
    """Project 3D landmark points (dict list or array) to normalized 2D camera view."""
    if isinstance(pts_seed[0], dict):
        pts_3d = np.array([[p['x'], p['y'], p['z']] for p in pts_seed])
    else:
        pts_3d = np.array(pts_seed)

    wrist = pts_3d[0]
    pts_rel = pts_3d - wrist
    middle_mcp = pts_rel[9]
    palm_len = np.linalg.norm(middle_mcp)
    total_len = palm_len * 1.85 if palm_len > 1e-6 else 1.0

    y_dir = -1.0 if middle_mcp[1] < 0 else 1.0

    pts_2d = []
    for p in pts_rel:
        u = 0.5 + (p[0] / total_len) * 0.65
        v = 0.82 - (y_dir * p[1] / total_len) * 0.65
        pts_2d.append([u, v])
    return np.array(pts_2d)

def draw_hand_skeleton(ax, pts_2d, title, subtitle=""):
    """Draw hand skeleton with clean colored bones and landmark nodes."""
    segment_indices = {
        'Thumb':  [(0,1),(1,2),(2,3),(3,4)],
        'Index':  [(0,5),(5,6),(6,7),(7,8)],
        'Middle': [(0,9),(9,10),(10,11),(11,12)],
        'Ring':   [(0,13),(13,14),(14,15),(15,16)],
        'Pinky':  [(0,17),(17,18),(18,19),(19,20)]
    }

    # Draw palm bones (wrist to MCPs)
    palm_conns = [(0,1), (0,5), (0,9), (0,13), (0,17), (5,9), (9,13), (13,17)]
    for start, end in palm_conns:
        ax.plot([pts_2d[start, 0], pts_2d[end, 0]],
                [pts_2d[start, 1], pts_2d[end, 1]],
                color='#555577', linewidth=1.5, linestyle=':', alpha=0.6)

    # Draw finger segments
    for finger, segs in segment_indices.items():
        color = FINGER_COLORS[finger]
        for start, end in segs:
            ax.plot([pts_2d[start, 0], pts_2d[end, 0]],
                    [pts_2d[start, 1], pts_2d[end, 1]],
                    color=color, linewidth=2.8, alpha=0.95)

    # Draw landmark nodes
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], color='#FFFFFF', edgecolors='#111111', s=35, zorder=5)

    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight='bold', pad=8)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.9, 0.1])  # Camera view format (top points UP)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

def find_best_matching_key(seeds_keys, pattern_states):
    """Find a seed label in seeds.json matching specified finger states."""
    for key in seeds_keys:
        if key.startswith('__'): continue
        if len(key) == 10:
            # key format: D4 A3 D3 A2 D2 A1 D1 A0 F P
            d4, a3, d3, a2, d2, a1, d1, a0, f, p = key
            matches = True
            if 'd4' in pattern_states and d4 != str(pattern_states['d4']): matches = False
            if 'a3' in pattern_states and a3 != str(pattern_states['a3']): matches = False
            if 'd3' in pattern_states and d3 != str(pattern_states['d3']): matches = False
            if 'a2' in pattern_states and a2 != str(pattern_states['a2']): matches = False
            if 'd2' in pattern_states and d2 != str(pattern_states['d2']): matches = False
            if 'a1' in pattern_states and a1 != str(pattern_states['a1']): matches = False
            if 'd1' in pattern_states and d1 != str(pattern_states['d1']): matches = False
            if 'a0' in pattern_states and a0 != str(pattern_states['a0']): matches = False
            if 'f' in pattern_states and f != str(pattern_states['f']): matches = False
            if 'p' in pattern_states and p != str(pattern_states['p']): matches = False
            if matches:
                return key
    return None

def main():
    print("=========================================================")
    print("  GERANDO RELATÓRIO VISUAL DAS SEMENTES (SEEDS.JSON)     ")
    print("=========================================================")

    seeds_path = os.path.join(SEEDS_DIR, 'seeds.json')
    if not os.path.exists(seeds_path):
        print(f"[ERRO] {seeds_path} não encontrado.")
        return

    with open(seeds_path, 'r', encoding='utf-8') as f:
        seeds = json.load(f)

    seed_keys = [k for k in seeds.keys() if not k.startswith('__')]
    print(f"Total de {len(seed_keys)} sementes encontradas no dataset.")

    # ---------------------------------------------------------
    # FIGURA 1: ESTÁGIOS DOS DEDOS (ESTÁGIOS 0 A 4)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 5, figsize=(18, 4.5))
    fig1.suptitle("1. SEMENTES DOS ESTÁGIOS DOS DEDOS (0: Reto | 1: Concha | 2: Gancho | 3: Mesa | 4: Punho)", fontsize=12, fontweight='bold', y=1.02)

    stage_patterns = [
        ({'d4': 0, 'd3': 0, 'd2': 0, 'd1': 0, 'f': 0, 'p': 0}, "Estágio 0: Reto", "Dedos 100% estendidos"),
        ({'d4': 1, 'd3': 1, 'd2': 1, 'd1': 1, 'f': 0, 'p': 0}, "Estágio 1: Concha", "Arco suave MCP/PIP"),
        ({'d4': 2, 'd3': 2, 'd2': 2, 'd1': 2, 'f': 0, 'p': 0}, "Estágio 2: Gancho", "Base reta, pontas dobradas"),
        ({'d4': 3, 'd3': 3, 'd2': 3, 'd1': 3, 'f': 0, 'p': 0}, "Estágio 3: Mesa", "MCP 90°, falanges retas"),
        ({'d4': 4, 'd3': 4, 'd2': 4, 'd1': 4, 'f': 1, 'p': 0}, "Estágio 4: Punho Fechado", "Dedos colados na palma")
    ]

    for idx, (pattern, title, sub) in enumerate(stage_patterns):
        k = find_best_matching_key(seed_keys, pattern) or seed_keys[min(idx, len(seed_keys)-1)]
        pts_2d = project_seed_to_2d(seeds[k])
        draw_hand_skeleton(axes1[idx], pts_2d, title, f"({k})\n{sub}")

    path1 = os.path.join(REPORTS_DIR, "01_limitacoes_estagios_dedos.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGEM GERADA 1/4] Estágios dos Dedos: {path1}")

    # ---------------------------------------------------------
    # FIGURA 2: ESTADOS DO POLEGAR (3 ESTADOS SIMPLIFICADOS)
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.suptitle("2. SEMENTES ANATÔMICAS SIMPLIFICADAS DO POLEGAR (3 Estados | IP Desconsiderado)", fontsize=12, fontweight='bold', y=1.02)

    thumb_patterns = [
        ({'a0': 0, 'f': 0, 'p': 0}, "1. Aberto Esticado (A0=0, F=0)", "Totalmente estendido na palma aberta"),
        ({'a0': 1, 'f': 0, 'p': 0}, "2. Junto aos Dedos (A0=1, F=0)", "Aduzido lateralmente"),
        ({'a0': 1, 'f': 1, 'p': 0}, "3. Na Transversal (A0=1, F=1)", "Oposição cruzando a palma")
    ]

    for idx, (pattern, title, sub) in enumerate(thumb_patterns):
        k = find_best_matching_key(seed_keys, pattern) or seed_keys[min(idx*10, len(seed_keys)-1)]
        pts_2d = project_seed_to_2d(seeds[k])
        draw_hand_skeleton(axes2[idx], pts_2d, title, f"({k})\n{sub}")

    path2 = os.path.join(REPORTS_DIR, "02_limitacoes_polegar_F_P.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGEM GERADA 2/4] Estados do Polegar: {path2}")

    # ---------------------------------------------------------
    # FIGURA 3: LIMITAÇÕES DE ABERTURAS (SPREADS)
    # ---------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4.5))
    fig3.suptitle("3. SEMENTES DE ABERTURA LATERAL (SPREADS A0, A1, A2, A3)", fontsize=12, fontweight='bold', y=1.02)

    spread_keys_demo = [
        find_best_matching_key(seed_keys, {'a3': 1, 'a2': 1, 'a1': 1, 'a0': 1, 'd4': 0, 'd3': 0, 'd2': 0, 'd1': 0}) or seed_keys[0],
        find_best_matching_key(seed_keys, {'a3': 1, 'a2': 1, 'a1': 1, 'a0': 0, 'd4': 0, 'd3': 0, 'd2': 0, 'd1': 0}) or seed_keys[1],
        find_best_matching_key(seed_keys, {'a3': 1, 'a2': 1, 'a1': 0, 'a0': 1, 'd4': 0, 'd3': 0, 'd2': 0, 'd1': 0}) or seed_keys[2],
        find_best_matching_key(seed_keys, {'a3': 0, 'a2': 0, 'a1': 0, 'a0': 0, 'd4': 0, 'd3': 0, 'd2': 0, 'd1': 0}) or seed_keys[3]
    ]

    spread_titles = [
        ("Dedos Juntos (Sem Spread)", "Paralelos colados"),
        ("Abertura Polegar (A0=0)", "Polegar afastado"),
        ("Abertura Indicador-Médio (A1=0)", "Afastamento lateral"),
        ("Leque Completo (Spreads=0)", "Abertura máxima total")
    ]

    for idx, k in enumerate(spread_keys_demo):
        title, sub = spread_titles[idx]
        pts_2d = project_seed_to_2d(seeds[k])
        draw_hand_skeleton(axes3[idx], pts_2d, title, f"({k})\n{sub}")

    path3 = os.path.join(REPORTS_DIR, "03_limitacoes_aberturas_spread.png")
    plt.tight_layout()
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGEM GERADA 3/4] Aberturas Laterais: {path3}")

    # ---------------------------------------------------------
    # FIGURA 4: EXEMPLOS DE SINAIS DO ALFABETO LIBRAS
    # ---------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 4, figsize=(16, 4.5))
    fig4.suptitle("4. SEMENTES DE SINAIS EXEMPLOS DE LIBRAS", fontsize=12, fontweight='bold', y=1.02)

    libras_patterns = [
        ({'d4': 4, 'd3': 4, 'd2': 4, 'd1': 4, 'a0': 1, 'f': 1, 'p': 0}, "Sinal 'A'", "Punho fechado, polegar oposto"),
        ({'d4': 0, 'd3': 4, 'd2': 4, 'd1': 4, 'a0': 1, 'f': 1, 'p': 0}, "Sinal 'I'", "Apenas mindinho levantado"),
        ({'d4': 4, 'd3': 4, 'd2': 0, 'd1': 0, 'a1': 0, 'a0': 1, 'f': 1, 'p': 0}, "Sinal 'V'", "Indicador e Médio em V"),
        ({'d4': 4, 'd3': 0, 'd2': 0, 'd1': 0, 'a2': 0, 'a1': 0, 'a0': 1, 'f': 1, 'p': 0}, "Sinal 'W'", "Indicador, Médio e Anelar estendidos")
    ]

    for idx, (pattern, title, sub) in enumerate(libras_patterns):
        k = find_best_matching_key(seed_keys, pattern) or seed_keys[min(idx*15, len(seed_keys)-1)]
        pts_2d = project_seed_to_2d(seeds[k])
        draw_hand_skeleton(axes4[idx], pts_2d, title, f"({k})\n{sub}")

    path4 = os.path.join(REPORTS_DIR, "04_sementes_exemplos_libras.png")
    plt.tight_layout()
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGEM GERADA 4/4] Exemplos Libras: {path4}")

    print("\n=========================================================")
    print(" [CONCLUÍDO] 4 PAINÉIS DE IMAGENS GERADOS COM SUCESSO!")
    print(" Pasta: Treinamento IA/reports/seed_verification/")
    print("=========================================================")

if __name__ == "__main__":
    main()
