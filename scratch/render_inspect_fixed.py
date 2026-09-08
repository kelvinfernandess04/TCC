import os, cv2, json, numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Treinamento IA', 'scripts'))
from inspect_seeds import UnicodeHUD, render_skeleton_viewport

SEEDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'Treinamento IA', 'data', 'seeds', 'seeds.json')
seeds = json.load(open(SEEDS_PATH, 'r', encoding='utf-8'))
seed_keys = list(seeds.keys())
key_code = '0001013110'
current_idx = seed_keys.index(key_code)
total_seeds = len(seed_keys)
pts_3d = np.array([[p['x'], p['y'], p['z']] for p in seeds[key_code]], dtype=np.float64)

frame = np.zeros((720, 1280, 3), dtype=np.uint8)
frame[:] = (17, 16, 24)
cv2.rectangle(frame, (0, 0), (1280, 72), (24, 24, 37), -1)
cv2.line(frame, (0, 72), (1280, 72), (166, 227, 161), 2)

vp_w, vp_h, vp_y = 460, 490, 90
vp1_x = 330
vp2_x = 330 + vp_w + 20
card_x, card_w, card_h = 22, 290, vp_h
leg_y = vp_y + card_h - 105

vp_front = render_skeleton_viewport(pts_3d, vp_w, vp_h, 15.0, -12.0, 'VISÃO 1: PERSPECTIVA FRONTAL / ISOMÉTRICA')
frame[vp_y:vp_y + vp_h, vp1_x:vp1_x + vp_w] = vp_front

vp_side = render_skeleton_viewport(pts_3d, vp_w, vp_h, 90.0, -5.0, 'VISÃO 2: PERFIL LATERAL 90° (PROFUNDIDADE Z)')
frame[vp_y:vp_y + vp_h, vp2_x:vp2_x + vp_w] = vp_side

cv2.rectangle(frame, (card_x, vp_y), (card_x + card_w, vp_y + card_h), (24, 24, 37), -1)
cv2.rectangle(frame, (card_x, vp_y), (card_x + card_w, vp_y + card_h), (69, 71, 90), 1)
cv2.line(frame, (card_x + 10, leg_y), (card_x + card_w - 10, leg_y), (69, 71, 90), 1)

cv2.rectangle(frame, (0, 645), (1280, 720), (17, 17, 27), -1)
cv2.line(frame, (0, 645), (1280, 645), (69, 71, 90), 1)

stage_names = {'0': 'Estendido (0°)', '1': 'Curvado (Concha)', '2': 'Gancho (Hook)', '3': 'Plataforma (Tabletop)', '4': 'Fechado (Punho)'}
d4, a3, d3, a2, d2, a1, d1, a0, f, p = key_code

hud = UnicodeHUD()
text_batch = [
    ('INSPETOR E PLAYER DE SEMENTES CINEMÁTICAS 3D (SEEDS.JSON)', (22, 14), 18, (205, 214, 244), True),
    (f'Seed [{current_idx + 1:,} / {total_seeds:,}]  |  Código DADADADAFP: {key_code}  |  Estado: ❚❚ PAUSADO', (22, 42), 13, (249, 226, 175), False),
    ('VISÃO 1: PERSPECTIVA FRONTAL / ISOMÉTRICA', (vp1_x + 12, vp_y + 4), 12, (137, 180, 250), True),
    ('Yaw: +15° | Pitch: -12°', (vp1_x + 12, vp_y + vp_h - 20), 11, (140, 145, 165), False),
    ('VISÃO 2: PERFIL LATERAL 90° (PROFUNDIDADE Z)', (vp2_x + 12, vp_y + 4), 12, (137, 180, 250), True),
    ('Yaw: +90° | Pitch: -5°', (vp2_x + 12, vp_y + vp_h - 20), 11, (140, 145, 165), False),
    ('ANÁLISE TAXONÔMICA:', (card_x + 14, vp_y + 14), 14, (249, 226, 175), True),
    (f'• Mindinho (D4): {stage_names.get(d4, d4)}', (card_x + 14, vp_y + 46), 12, (225, 120, 215), True),
    (f'  Spread Min-Ane: {"Aberto" if a3=="0" else "Fechado"}', (card_x + 14, vp_y + 68), 11, (180, 180, 190), False),
    (f'• Anelar (D3):   {stage_names.get(d3, d3)}', (card_x + 14, vp_y + 92), 12, (240, 210, 80), True),
    (f'  Spread Ane-Med: {"Aberto" if a2=="0" else "Fechado"}', (card_x + 14, vp_y + 114), 11, (180, 180, 190), False),
    (f'• Médio (D2):    {stage_names.get(d2, d2)}', (card_x + 14, vp_y + 138), 12, (110, 230, 130), True),
    (f'  Spread Med-Ind: {"Aberto" if a1=="0" else "Fechado"}', (card_x + 14, vp_y + 160), 11, (180, 180, 190), False),
    (f'• Indicador (D1):{stage_names.get(d1, d1)}', (card_x + 14, vp_y + 184), 12, (70, 225, 255), True),
    (f'  Spread Ind-Pol: {"Aberto" if a0=="0" else "Fechado"}', (card_x + 14, vp_y + 206), 11, (180, 180, 190), False),
    (f'• Polegar (F):   {"Oposição Transv." if f=="1" else "No Plano da Palma"}', (card_x + 14, vp_y + 230), 12, (40, 140, 255), True),
    (f'• Ponta Pol.(P): {"Flexionada (IP)" if p=="1" else "Estendida"}', (card_x + 14, vp_y + 254), 12, (40, 140, 255), True),
    ('Cores das Articulações:', (card_x + 14, leg_y + 8), 11, (205, 214, 244), True),
    ('• Polegar: Laranja  • Indicador: Amarelo', (card_x + 14, leg_y + 28), 10, (249, 226, 175), False),
    ('• Médio: Verde      • Anelar: Ciano', (card_x + 14, leg_y + 46), 10, (166, 227, 161), False),
    ('• Mínimo: Magenta   • Pontas: Branco/Verde', (card_x + 14, leg_y + 64), 10, (245, 194, 231), False),
    ('[ESPAÇO]: Play/Pause Sequencial  |  [D]/[->]: Próxima  |  [A]/[<-]: Anterior  |  [W]/[S]: +/- 50 Seeds', (25, 658), 12, (166, 227, 161), False),
    ('[L]: Exemplos Libras (A, B, C, V, W...)  |  [1]..[5]: Filtrar Estágio Flexão  |  [+/-]: Velocidade  |  [Q]: Sair', (25, 684), 11, (205, 214, 244), False)
]
frame = hud.render_batch(frame, text_batch)
out_path = os.path.join(os.path.dirname(__file__), 'rendered_inspect_seed_fixed.png')
cv2.imwrite(out_path, frame)
print('Saved successfully to:', out_path)
