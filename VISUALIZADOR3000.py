import json
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.widgets import Slider, Button
import numpy as np

# Definir conexões para desenhar o esqueleto
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

# Carregar o arquivo JSON
print("Selecione um arquivo JSON...")
root = tk.Tk()
root.withdraw()

json_file = filedialog.askopenfilename(
    title="Selecione um arquivo JSON",
    filetypes=[
        ("Arquivos JSON", "*.json"),
        ("Todos os arquivos", "*.*")
    ]
)

if not json_file:
    print("Nenhum arquivo selecionado. Encerrando...")
    exit()

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

frames = data['frames']

# Configurar a figura
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(bottom=0.25)

# Cores
hand_colors = {'Left': 'blue', 'Right': 'red'}
pose_color = 'green'
pose_point_color = 'darkgreen'

def draw_connections(x_coords, y_coords, connections, color, linewidth=2, alpha=0.6):
    """Desenha conexões entre pontos"""
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(x_coords) and end_idx < len(y_coords):
            ax.plot([x_coords[start_idx], x_coords[end_idx]], 
                   [y_coords[start_idx], y_coords[end_idx]], 
                   color=color, linewidth=linewidth, alpha=alpha)

def draw_frame(frame_idx):
    """Desenha um frame específico com mãos e pose"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Frame {frames[frame_idx]['frame']} - Timestamp: {frames[frame_idx]['timestamp_ms']}ms")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    hands_data = frames[frame_idx]['hands']
    pose_data = frames[frame_idx]['pose']
    
    has_detection = False
    
    # Desenhar POSE
    if pose_data:
        has_detection = True
        for pose in pose_data:
            landmarks = pose['landmarks']
            
            # Extrair coordenadas
            x_coords = [lm['x'] for lm in landmarks]
            y_coords = [lm['y'] for lm in landmarks]
            
            # Desenhar conexões da pose
            draw_connections(x_coords, y_coords, POSE_CONNECTIONS, 
                           pose_color, linewidth=2, alpha=0.6)
            
            # Desenhar pontos da pose
            ax.scatter(x_coords, y_coords, c=pose_point_color, s=80, 
                      alpha=0.8, edgecolors='black', linewidths=1.5,
                      label='Pose', zorder=5)
            
            # Adicionar números aos pontos principais (opcional)
            # for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            #     ax.annotate(str(i), (x, y), fontsize=6, ha='center', va='center')
    
    # Desenhar MÃOS
    if hands_data:
        has_detection = True
        for hand in hands_data:
            handedness = hand['handedness']
            landmarks = hand['landmarks']
            color = hand_colors.get(handedness, 'orange')
            
            # Extrair coordenadas
            x_coords = [lm['x'] for lm in landmarks]
            y_coords = [lm['y'] for lm in landmarks]
            
            # Desenhar conexões da mão
            draw_connections(x_coords, y_coords, HAND_CONNECTIONS, 
                           color, linewidth=2, alpha=0.7)
            
            # Desenhar pontos da mão
            ax.scatter(x_coords, y_coords, c=color, s=100, alpha=0.8, 
                      label=f"{handedness} (conf: {hand['confidence']:.2f})", 
                      edgecolors='black', linewidths=1.5, zorder=10)
            
            # Adicionar números aos pontos
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax.annotate(str(i), (x, y), fontsize=8, ha='center', 
                          va='center', color='white', weight='bold')
    
    if not has_detection:
        ax.text(0.5, 0.5, 'Nenhuma detecção neste frame', 
                ha='center', va='center', fontsize=14, color='gray')
    else:
        ax.legend(loc='upper right')

# Desenhar o primeiro frame
draw_frame(0)

# Criar slider para navegar entre frames
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(frames)-1, valinit=0, valstep=1)

def update_slider(val):
    frame_idx = int(slider.val)
    draw_frame(frame_idx)
    fig.canvas.draw_idle()

slider.on_changed(update_slider)

# Botões de navegação
ax_prev = plt.axes([0.2, 0.05, 0.1, 0.04])
ax_next = plt.axes([0.7, 0.05, 0.1, 0.04])
btn_prev = Button(ax_prev, 'Anterior')
btn_next = Button(ax_next, 'Próximo')

def prev_frame(event):
    current = int(slider.val)
    if current > 0:
        slider.set_val(current - 1)

def next_frame(event):
    current = int(slider.val)
    if current < len(frames) - 1:
        slider.set_val(current + 1)

btn_prev.on_clicked(prev_frame)
btn_next.on_clicked(next_frame)

# Adicionar informações de atalhos
info_text = "Use as setas ← → do teclado ou os botões para navegar"
fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic', color='gray')

# Suporte para teclas de seta
def on_key(event):
    if event.key == 'left':
        prev_frame(None)
    elif event.key == 'right':
        next_frame(None)

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()