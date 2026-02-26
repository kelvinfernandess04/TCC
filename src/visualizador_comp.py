import json
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.widgets import Slider, Button
import numpy as np

# Definir conexões
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
    (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20),  # Mindinho
    (5, 9), (9, 13), (13, 17)  # Palma
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Rosto
    (0, 4), (4, 5), (5, 6), (6, 8),  # Rosto
    (9, 10),  # Boca
    (11, 12),  # Ombros
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Braço esquerdo
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Braço direito
    (11, 23), (12, 24), (23, 24),  # Tronco
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Perna esquerda
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)   # Perna direita
]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

print("Selecione um arquivo de Match JSON (results/json/)...")
root = tk.Tk()
root.withdraw()
match_file = filedialog.askopenfilename(
    title="Selecione um arquivo de Match JSON",
    filetypes=[("Arquivos JSON", "*.json"), ("Todos os arquivos", "*.*")]
)

if not match_file:
    print("Nenhum arquivo selecionado. Encerrando...")
    exit()

try:
    match_data = load_json(match_file)
    base_file = match_data['base']
    target_file = match_data['target']
    matches = match_data['matches']
except KeyError as e:
    print("\n[ERRO FATAL] Formato de arquivo Incorreto!")
    print(f"O arquivo que você selecionou não possui a chave {e}.")
    print("Você provavelmente selecionou um arquivo de vídeo bruto da pasta 'data/'.")
    print("Por favor, rode o script novamente e selecione um arquivo de MATCH dentro da pasta 'results/json/'.")
    print("Exemplo: 'results/json/lm_abacate_base_boa.mp4_norm.json_vs_lm_abacate_teste1.mp4.json'")
    input("\nPressione Enter para sair...")
    exit(1)

# Garantir que a lista de matches está ordenada
matches.sort(key=lambda x: x.get('target_frame', 0))

base_data = load_json(base_file)
target_data = load_json(target_file)
b_frames = base_data.get('frames', [])
t_frames = target_data.get('frames', [])

t_frames = target_data.get('frames', [])

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
ax_base = fig.add_subplot(gs[0, 0])
ax_target = fig.add_subplot(gs[0, 1])
ax_timeline = fig.add_subplot(gs[1, :])
plt.subplots_adjust(bottom=0.20, hspace=0.3)

hand_colors = {'Left': 'blue', 'Right': 'red'}
pose_color = 'green'
pose_point_color = 'darkgreen'

def draw_connections(ax, x_coords, y_coords, connections, color, linewidth=2, alpha=0.6):
    for start_idx, end_idx in connections:
        if start_idx < len(x_coords) and end_idx < len(y_coords):
            ax.plot([x_coords[start_idx], x_coords[end_idx]], 
                   [y_coords[start_idx], y_coords[end_idx]], 
                   color=color, linewidth=linewidth, alpha=alpha)

def get_global_bounds(frames):
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    
    for f in frames:
        for pose in f.get('pose', []):
            for lm in pose.get('landmarks', []):
                min_x, max_x = min(min_x, lm['x']), max(max_x, lm['x'])
                min_y, max_y = min(min_y, lm['y']), max(max_y, lm['y'])
        for hand in f.get('hands', []):
            for lm in hand.get('landmarks', []):
                min_x, max_x = min(min_x, lm['x']), max(max_x, lm['x'])
                min_y, max_y = min(min_y, lm['y']), max(max_y, lm['y'])
                
    if min_x == float('inf'):
        return 0, 1, 0, 1
        
    pad_x = (max_x - min_x) * 0.1 if max_x != min_x else 0.5
    pad_y = (max_y - min_y) * 0.1 if max_y != min_y else 0.5
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y

b_bounds = get_global_bounds(b_frames)
t_bounds = get_global_bounds(t_frames)

def draw_frame(ax, frame_data, title, bounds, bg_color='white'):
    ax.clear()
    ax.set_facecolor(bg_color)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if not frame_data:
        ax.text(0.5, 0.5, 'Sem dados detectados', ha='center', va='center', fontsize=14, color='gray')
        return

    hands_data = frame_data.get('hands', [])
    pose_data = frame_data.get('pose', [])
    
    has_detection = False
    
    if pose_data:
        has_detection = True
        for pose in pose_data:
            landmarks = pose.get('landmarks', [])
            x_coords = [lm['x'] for lm in landmarks]
            y_coords = [lm['y'] for lm in landmarks]
            draw_connections(ax, x_coords, y_coords, POSE_CONNECTIONS, pose_color)
            ax.scatter(x_coords, y_coords, c=pose_point_color, s=80, alpha=0.8, edgecolors='black', zorder=5)
            
    if hands_data:
        has_detection = True
        for hand in hands_data:
            handedness = hand.get('handedness', 'Left')
            landmarks = hand.get('landmarks', [])
            color = hand_colors.get(handedness, 'orange')
            x_coords = [lm['x'] for lm in landmarks]
            y_coords = [lm['y'] for lm in landmarks]
            draw_connections(ax, x_coords, y_coords, HAND_CONNECTIONS, color)
            ax.scatter(x_coords, y_coords, c=color, s=100, alpha=0.8, edgecolors='black', zorder=10)

    if not has_detection:
        ax.text(0.5, 0.5, 'Nenhuma detecção neste frame', ha='center', va='center', fontsize=14, color='gray')

def update_view(t_idx):
    current_match = None
    for m in matches:
        if m["target_frame"] == t_idx:
            current_match = m
            break
        elif m["target_frame"] < t_idx:
            current_match = m
            
    t_title = f"TARGET\nFrame {t_idx} - Time: {t_frames[t_idx].get('timestamp_ms', 0)}ms" if t_idx < len(t_frames) else "TARGET"
    draw_frame(ax_target, t_frames[t_idx] if t_idx < len(t_frames) else None, t_title, t_bounds)

    # TIMELINE DRAW
    ax_timeline.clear()
    ax_timeline.set_title("Alinhamento Otimizado (Base -> Target)")
    ax_timeline.set_xlabel("Target Timeline (Frames)")
    ax_timeline.set_yticks([]) # Hide Y
    ax_timeline.set_xlim(-5, len(t_frames) + 5)
    ax_timeline.plot([0, len(t_frames)], [0, 0], color='gray', alpha=0.3, linewidth=2)
    
    # Navegação Atual
    ax_timeline.scatter([t_idx], [0], color='purple', s=150, zorder=10, marker='X', label='Momento Atual')
    
    # Tentar ler todos os keyframes da base pra ver os perdidos
    all_b_keys = match_data.get('base_keyframes', [])
    matched_b_keys = [m["base_frame"] for m in matches]
    
    # Mapear posições fictícias na timeline pros vermelhos caírem ordenados
    # ou só listamos num text box. Como não sabemos em que "tempo" do target cai uma falha,
    # vamos empilhar eles como Status.
    
    for m in matches:
        t_f = m["target_frame"]
        b_f = m["base_frame"]
        ax_timeline.scatter([t_f], [0], color='green', s=60, zorder=5)
        ax_timeline.plot([t_f, t_f], [0, 1], color='green', alpha=0.5)
        ax_timeline.text(t_f, 1.1, f"B{b_f}", ha='center', fontsize=8, color='darkgreen', rotation=90)
        
    ax_timeline.legend(loc='upper right')
    
    # Exibir painel de Keyframes Rejeitados na Esquerda
    dropped_keys = [k for k in all_b_keys if k not in matched_b_keys]
    if dropped_keys:
        drop_text = "Deduções de Keyframes (Perdidos/Dropados):\n" + ", ".join([f"B{k}" for k in dropped_keys])
        ax_timeline.text(0.01, 0.90, drop_text, transform=ax_timeline.transAxes, va='top', ha='left', fontsize=9, color='darkred', bbox=dict(facecolor='#ffe6e6', alpha=0.8))
    else:
        ax_timeline.text(0.01, 0.90, "100% dos Keyframes Mapeados com Sucesso.", transform=ax_timeline.transAxes, va='top', ha='left', fontsize=9, color='darkgreen', bbox=dict(facecolor='#e6ffe6', alpha=0.8))

    if current_match:
        b_idx_real = current_match["base_frame"]
        is_exact = current_match["target_frame"] == t_idx
        
        b_title = f"BASE (Keyframe Localizado)\nFrame {b_idx_real}"
        if is_exact:
            b_bg = '#e6ffe6'
            b_title += f" [MATCH ATIVO!]"
        else:
            b_bg = '#ffffff'
            b_title += f" (Passou / Aguardando)"
            
        draw_frame(ax_base, b_frames[b_idx_real] if b_idx_real < len(b_frames) else None, b_title, b_bounds, bg_color=b_bg)
        
        if is_exact:
            info = f"Score Parcial: {current_match['net_score']:.1f}/100\n" \
                   f"Forma: {current_match['shape_score']:.1f}\n" \
                   f"Posição: {current_match['position_score']:.1f}\n" \
                   f"Mov: {current_match['movement_score']:.1f}\n" \
                   f"Penalty Gap: -{current_match['gap_penalty']:.1f}"
            ax_base.text(0.05, 0.95, info, transform=ax_base.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.9))
            
    else:
        draw_frame(ax_base, None, "BASE\nAguardando início da sequência...", b_bounds)

    status_color = "red" if match_data['status'] == "FAIL" else "green"
    fig.suptitle(f"COMPARADOR - Score Final: {match_data['score']:.1f}/100\n", fontsize=16, fontweight='bold', color='black')
    fig.text(0.5, 0.95, match_data['status'], fontsize=16, fontweight='bold', color=status_color, ha='center')
    
    fig.canvas.draw_idle()

ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
slider = Slider(ax_slider, 'Target Frame', 0, len(t_frames)-1, valinit=0, valstep=1)

def on_slider_change(val):
    update_view(int(val))
    
slider.on_changed(on_slider_change)

ax_prev = plt.axes([0.2, 0.03, 0.1, 0.04])
ax_next = plt.axes([0.7, 0.03, 0.1, 0.04])
btn_prev = Button(ax_prev, 'Anterior')
btn_next = Button(ax_next, 'Próximo')

def prev_frame(event):
    curr = int(slider.val)
    if curr > 0: slider.set_val(curr - 1)

def next_frame(event):
    curr = int(slider.val)
    if curr < len(t_frames)-1: slider.set_val(curr + 1)

btn_prev.on_clicked(prev_frame)
btn_next.on_clicked(next_frame)

info_text = "Use as setas ← → do teclado ou os botões para navegar"
fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, style='italic', color='gray')

def on_key(event):
    if event.key == 'left': prev_frame(None)
    elif event.key == 'right': next_frame(None)

fig.canvas.mpl_connect('key_press_event', on_key)

update_view(0)
plt.show()
