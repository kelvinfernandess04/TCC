import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


# -----------------------------
# Helpers numéricos
# -----------------------------

EPS = 1e-9

def safe_norm(v):
    n = np.linalg.norm(v)
    return n if n > EPS else EPS

def unit(v):
    return v / safe_norm(v)

def angle_cos(a, b, c):
    """cos do ângulo ABC (no ponto B)"""
    v1 = a - b
    v2 = c - b
    return float(np.dot(v1, v2) / (safe_norm(v1) * safe_norm(v2)))

def lerp(a, b, t):
    return (1 - t) * a + t * b

def fill_nan_linear(arr):
    """
    Interpola NaNs linearmente por coluna.
    arr: (T, D)
    """
    T, D = arr.shape
    out = arr.copy()
    for d in range(D):
        col = out[:, d]
        nan = np.isnan(col)
        if nan.all():
            continue
        idx = np.arange(T)
        good = ~nan
        out[nan, d] = np.interp(idx[nan], idx[good], col[good])
    return out

def robust_zscore(seq):
    """
    Normalização robusta por dimensão (mediana e MAD).
    Ajuda quando tem ruído ou outliers.
    """
    med = np.nanmedian(seq, axis=0)
    mad = np.nanmedian(np.abs(seq - med), axis=0)
    mad = np.where(mad < 1e-6, 1.0, mad)
    return (seq - med) / mad


# -----------------------------
# DTW com máscara de validade
# -----------------------------

def masked_frame_distance(a, ma, b, mb, min_valid_dims=8, missing_penalty=2.0):
    """
    Distância entre dois frames com máscara booleana de dimensões válidas.
    - Considera só dims válidas em ambos.
    - Se poucas dims válidas, penaliza (pra evitar "parece igual pq falta dado").
    """
    valid = ma & mb
    k = int(valid.sum())
    if k == 0:
        return 1e6
    diff = a[valid] - b[valid]
    dist = float(np.linalg.norm(diff) / np.sqrt(k))  # normaliza por k
    if k < min_valid_dims:
        dist *= (1.0 + missing_penalty * (min_valid_dims - k) / min_valid_dims)
    return dist

def dtw_masked(seqA, maskA, seqB, maskB, window=None):
    """
    DTW (Sakoe-Chiba) para sequências com máscara de validade por frame.
    Retorna: custo_total, caminho[(i,j)], custos_por_passo
    """
    n = len(seqA)
    m = len(seqB)
    if n == 0 or m == 0:
        return np.inf, [], []

    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    P = np.full((n + 1, m + 1, 2), -1, dtype=np.int32)

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = masked_frame_distance(seqA[i - 1], maskA[i - 1], seqB[j - 1], maskB[j - 1])

            choices = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]
            k = int(np.argmin(choices))
            if k == 0:
                prev = (i - 1, j)
            elif k == 1:
                prev = (i, j - 1)
            else:
                prev = (i - 1, j - 1)

            D[i, j] = cost + choices[k]
            P[i, j] = prev

    # reconstrói caminho
    i, j = n, m
    path = []
    step_costs = []
    while i > 0 and j > 0:
        ii, jj = i - 1, j - 1
        path.append((ii, jj))
        step_costs.append(masked_frame_distance(seqA[ii], maskA[ii], seqB[jj], maskB[jj]))
        pi, pj = P[i, j]
        if pi < 0:
            break
        i, j = int(pi), int(pj)

    path.reverse()
    step_costs.reverse()
    return float(D[n, m]), path, step_costs


# -----------------------------
# Comparador 2.0 (corpo -> base -> features)
# -----------------------------

class GestureComparator2:
    def __init__(self, hand_weight=0.7, pose_weight=0.3, dtw_window_ratio=0.2):
        total = hand_weight + pose_weight
        self.hand_weight = hand_weight / total if total > 0 else 1.0
        self.pose_weight = pose_weight / total if total > 0 else 0.0
        self.dtw_window_ratio = dtw_window_ratio

        print("⚙️ GestureComparator 2.0")
        print(f"   • Pesos: mãos={self.hand_weight*100:.1f}%, pose={self.pose_weight*100:.1f}%")
        print(f"   • DTW window ratio: {self.dtw_window_ratio:.2f}")
        print()

    # ---- parsing landmarks ----
    def _to_xyz(self, lms):
        # lms: lista de dicts com x,y,z
        return np.array([[p["x"], p["y"], p["z"]] for p in lms], dtype=np.float64)

    # ---- base corporal ----
    def body_basis(self, pose33):
        """
        Cria uma base ortonormal do corpo:
        origem = centro do quadril (23,24)
        X = ombro_esq->ombro_dir (11->12) (lateral)
        Y = quadril->ombros (vertical do tronco)
        Z = X x Y (profundidade)
        Escala = largura dos ombros
        """
        # índices MediaPipe Pose
        L_SH, R_SH = 11, 12
        L_HIP, R_HIP = 23, 24

        hip_c = (pose33[L_HIP] + pose33[R_HIP]) / 2.0
        sh_c = (pose33[L_SH] + pose33[R_SH]) / 2.0

        x_axis = unit(pose33[R_SH] - pose33[L_SH])
        y_axis = unit(sh_c - hip_c)

        z_axis = np.cross(x_axis, y_axis)
        if np.linalg.norm(z_axis) < 1e-6:
            # fallback: usa um eixo z padrão
            z_axis = np.array([0, 0, 1], dtype=np.float64)
        z_axis = unit(z_axis)

        # re-ortogonaliza y (pra ficar bonitinho de verdade)
        y_axis = unit(np.cross(z_axis, x_axis))

        scale = np.linalg.norm(pose33[R_SH] - pose33[L_SH])
        if scale < 1e-6:
            scale = 1.0

        return hip_c, x_axis, y_axis, z_axis, scale

    def to_body_coords(self, pts, origin, x_axis, y_axis, z_axis, scale):
        """
        Converte pontos do mundo para coordenadas do corpo.
        """
        rel = pts - origin
        xb = rel @ x_axis
        yb = rel @ y_axis
        zb = rel @ z_axis
        out = np.stack([xb, yb, zb], axis=1) / scale
        return out

    # ---- hands: achar left/right com robustez ----
    def split_hands(self, hands_list):
        """
        Retorna (left_hand, right_hand) onde cada um é dict com landmarks/conf.
        Tenta:
        - campo 'handedness'/'label'
        - campo 'category_name' (MediaPipe tasks)
        - fallback: None
        """
        left = None
        right = None
        for h in hands_list or []:
            label = None
            conf = None

            # vários formatos possíveis
            if "handedness" in h:
                # pode ser string, dict, lista
                hd = h["handedness"]
                if isinstance(hd, str):
                    label = hd
                elif isinstance(hd, dict):
                    label = hd.get("label") or hd.get("category_name")
                    conf = hd.get("score") or hd.get("confidence")
                elif isinstance(hd, list) and len(hd) > 0:
                    label = hd[0].get("label") or hd[0].get("category_name")
                    conf = hd[0].get("score") or hd[0].get("confidence")

            if label is None:
                label = h.get("label") or h.get("category_name")

            if conf is None:
                conf = h.get("confidence") or h.get("score")

            label_norm = (label or "").lower()
            if "left" in label_norm or "esq" in label_norm:
                left = {"landmarks": h.get("landmarks"), "conf": conf}
            elif "right" in label_norm or "dir" in label_norm:
                right = {"landmarks": h.get("landmarks"), "conf": conf}

        return left, right

    # ---- features: pose ----
    def pose_features(self, pose33_body):
        """
        Features de pose que preservam sentido:
        - ângulos (cos) cotovelos e ombros
        - posições dos punhos em coords do corpo (x,y,z)
        - posições relativas punho-nariz (mão perto do rosto vs barriga)
        """
        # índices
        NOSE = 0
        L_SH, R_SH = 11, 12
        L_EL, R_EL = 13, 14
        L_WR, R_WR = 15, 16
        L_HIP, R_HIP = 23, 24

        feats = []
        valid = []

        def add_val(v, ok=True):
            if ok:
                feats.append(float(v))
                valid.append(True)
            else:
                feats.append(np.nan)
                valid.append(False)

        # ângulos
        try:
            add_val(angle_cos(pose33_body[L_SH], pose33_body[L_EL], pose33_body[L_WR]), True)  # cotovelo esq
        except:
            add_val(0, False)

        try:
            add_val(angle_cos(pose33_body[R_SH], pose33_body[R_EL], pose33_body[R_WR]), True)  # cotovelo dir
        except:
            add_val(0, False)

        try:
            add_val(angle_cos(pose33_body[L_EL], pose33_body[L_SH], pose33_body[L_HIP]), True)  # ombro esq
        except:
            add_val(0, False)

        try:
            add_val(angle_cos(pose33_body[R_EL], pose33_body[R_SH], pose33_body[R_HIP]), True)  # ombro dir
        except:
            add_val(0, False)

        # punhos (posição no corpo)
        for idx in [L_WR, R_WR]:
            for c in range(3):
                v = pose33_body[idx, c]
                add_val(v, True)

        # punho relativo ao nariz (mão perto do rosto)
        for idx in [L_WR, R_WR]:
            v = pose33_body[idx] - pose33_body[NOSE]
            for c in range(3):
                add_val(v[c], True)

        return np.array(feats, dtype=np.float64), np.array(valid, dtype=bool)

    # ---- features: hand ----
    def hand_features(self, hand21_body, wrist_body=None):
        """
        Features de mão em coords do corpo:
        - posição do punho (se vier)
        - abertura: distâncias entre pontas
        - direção dos dedos: vetores unitários base->ponta
        """
        feats = []
        valid = []

        def add(v, ok=True):
            if ok:
                feats.append(float(v))
                valid.append(True)
            else:
                feats.append(np.nan)
                valid.append(False)

        # Se tiver um punho do pose, usar como referência extra (posição da mão no corpo)
        if wrist_body is not None:
            for c in range(3):
                add(wrist_body[c], True)
        else:
            for _ in range(3):
                add(0, False)

        # normaliza mão localmente (tirando o pulso e escalando pela palma)
        try:
            pts = hand21_body.copy()
            wrist = pts[0]
            centered = pts - wrist

            palm_scale = np.linalg.norm(pts[5] - pts[17])
            if palm_scale < 1e-6:
                palm_scale = np.linalg.norm(pts[0] - pts[9])
            palm_scale = palm_scale if palm_scale > 1e-6 else 1.0
            hn = centered / palm_scale
        except:
            # não deu: marca inválido
            hn = None

        if hn is None:
            # reserva espaço: 5+10+15 = 30 features (abaixo) + 3 do punho
            for _ in range(30):
                add(0, False)
            return np.array(feats, dtype=np.float64), np.array(valid, dtype=bool)

        fingertips = [4, 8, 12, 16, 20]
        mcp = [2, 5, 9, 13, 17]

        # dist ponta-base (5)
        for t, b in zip(fingertips, mcp):
            add(np.linalg.norm(hn[t] - hn[b]), True)

        # dist entre pontas (10)
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                add(np.linalg.norm(hn[fingertips[i]] - hn[fingertips[j]]), True)

        # direções base->ponta (5 vetores * 3 = 15)
        for t, b in zip(fingertips, mcp):
            v = hn[t] - hn[b]
            v = v / safe_norm(v)
            add(v[0], True)
            add(v[1], True)
            add(v[2], True)

        return np.array(feats, dtype=np.float64), np.array(valid, dtype=bool)

    # ---- extrair sequência completa ----
    def extract_sequences(self, frames, use_pose=True, use_hands=True):
        pose_seq = []
        pose_mask = []

        hand_seq = []
        hand_mask = []

        for fr in frames:
            pose_items = fr.get("pose", [])
            hands_items = fr.get("hands", [])

            pose33 = None
            if use_pose and pose_items and "landmarks" in pose_items[0]:
                pose33 = self._to_xyz(pose_items[0]["landmarks"])
                if len(pose33) >= 33:
                    origin, x, y, z, scale = self.body_basis(pose33)
                    pose_body = self.to_body_coords(pose33, origin, x, y, z, scale)
                    pf, pm = self.pose_features(pose_body)
                else:
                    pf, pm = None, None
            else:
                pf, pm = None, None

            if use_pose:
                if pf is None:
                    # tamanho fixo: 4 ang + 6 punhos + 6 (punho-nariz) = 16
                    pose_seq.append(np.full((16,), np.nan, dtype=np.float64))
                    pose_mask.append(np.zeros((16,), dtype=bool))
                else:
                    pose_seq.append(pf)
                    pose_mask.append(pm)

            # mãos
            if use_hands:
                # tenta split L/R
                left, right = self.split_hands(hands_items)

                # se houver pose, pegamos punho do pose como âncora (melhor que nada)
                wristL_body = None
                wristR_body = None
                if pose33 is not None and len(pose33) >= 33:
                    origin, x, y, z, scale = self.body_basis(pose33)
                    pose_body = self.to_body_coords(pose33, origin, x, y, z, scale)
                    wristL_body = pose_body[15]  # L_WR
                    wristR_body = pose_body[16]  # R_WR

                # monta vetor de mão total = [features_left || features_right]
                # cada mão: 3 (pos punho) + 30 = 33
                total_feats = []
                total_mask = []

                # LEFT
                if left and left.get("landmarks"):
                    h21 = self._to_xyz(left["landmarks"])
                    if pose33 is not None and len(pose33) >= 33:
                        origin, x, y, z, scale = self.body_basis(pose33)
                        h_body = self.to_body_coords(h21, origin, x, y, z, scale)
                    else:
                        # sem pose, usa a própria mão "crua" (pior, mas dá)
                        h_body = h21.copy()

                    hf, hm = self.hand_features(h_body, wrist_body=wristL_body)
                else:
                    hf = np.full((33,), np.nan, dtype=np.float64)
                    hm = np.zeros((33,), dtype=bool)

                total_feats.append(hf)
                total_mask.append(hm)

                # RIGHT
                if right and right.get("landmarks"):
                    h21 = self._to_xyz(right["landmarks"])
                    if pose33 is not None and len(pose33) >= 33:
                        origin, x, y, z, scale = self.body_basis(pose33)
                        h_body = self.to_body_coords(h21, origin, x, y, z, scale)
                    else:
                        h_body = h21.copy()

                    hf, hm = self.hand_features(h_body, wrist_body=wristR_body)
                else:
                    hf = np.full((33,), np.nan, dtype=np.float64)
                    hm = np.zeros((33,), dtype=bool)

                total_feats.append(hf)
                total_mask.append(hm)

                hand_seq.append(np.concatenate(total_feats, axis=0))
                hand_mask.append(np.concatenate(total_mask, axis=0))

        pose_seq = np.array(pose_seq, dtype=np.float64) if use_pose else None
        pose_mask = np.array(pose_mask, dtype=bool) if use_pose else None
        hand_seq = np.array(hand_seq, dtype=np.float64) if use_hands else None
        hand_mask = np.array(hand_mask, dtype=bool) if use_hands else None

        return pose_seq, pose_mask, hand_seq, hand_mask

    def compare(self, json1, json2, use_pose=True, use_hands=True):
        with open(json1, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        with open(json2, "r", encoding="utf-8") as f:
            d2 = json.load(f)

        frames1 = d1.get("frames", [])
        frames2 = d2.get("frames", [])

        print(f"  • Vídeo 1: {len(frames1)} frames")
        print(f"  • Vídeo 2: {len(frames2)} frames")

        p1, pm1, h1, hm1 = self.extract_sequences(frames1, use_pose, use_hands)
        p2, pm2, h2, hm2 = self.extract_sequences(frames2, use_pose, use_hands)

        results = {}

        def prep(seq, mask):
            # seq tem NaN onde inválido. Vamos:
            # 1) preencher NaN por interpolação (só onde tem alguns pontos)
            # 2) normalização robusta
            if seq is None:
                return None, None
            filled = fill_nan_linear(seq)
            normed = robust_zscore(filled)
            # máscara permanece, mas depois do fill usamos a máscara original para distância
            return normed, mask

        p1n, pm1n = prep(p1, pm1)
        p2n, pm2n = prep(p2, pm2)
        h1n, hm1n = prep(h1, hm1)
        h2n, hm2n = prep(h2, hm2)

        # DTW window
        def win(n, m):
            return int(max(n, m) * self.dtw_window_ratio)

        # pose DTW
        pose_cost = None
        pose_path = []
        pose_steps = []
        pose_sim = 0.0
        if use_pose and p1n is not None and p2n is not None:
            w = win(len(p1n), len(p2n))
            pose_cost, pose_path, pose_steps = dtw_masked(p1n, pm1n, p2n, pm2n, window=w)
            avg = pose_cost / max(1, len(pose_path))
            # similarity: mapeia distância -> [0,1]
            pose_sim = float(np.exp(-avg))

        # hands DTW
        hand_cost = None
        hand_path = []
        hand_steps = []
        hand_sim = 0.0
        if use_hands and h1n is not None and h2n is not None:
            w = win(len(h1n), len(h2n))
            hand_cost, hand_path, hand_steps = dtw_masked(h1n, hm1n, h2n, hm2n, window=w)
            avg = hand_cost / max(1, len(hand_path))
            hand_sim = float(np.exp(-avg))

        # ponderado
        weighted = 0.0
        used = 0.0
        if use_hands and h1n is not None and h2n is not None:
            weighted += hand_sim * self.hand_weight
            used += self.hand_weight
        if use_pose and p1n is not None and p2n is not None:
            weighted += pose_sim * self.pose_weight
            used += self.pose_weight
        if used > 0:
            weighted /= used

        # diffs combinados (no caminho)
        combined_steps = []
        if hand_steps and pose_steps:
            L = min(len(hand_steps), len(pose_steps))
            for k in range(L):
                combined_steps.append(hand_steps[k] * self.hand_weight + pose_steps[k] * self.pose_weight)
        elif hand_steps:
            combined_steps = hand_steps
        elif pose_steps:
            combined_steps = pose_steps

        results = {
            "similarity": float(weighted),
            "hand_similarity": float(hand_sim),
            "pose_similarity": float(pose_sim),
            "hand_weight": float(self.hand_weight),
            "pose_weight": float(self.pose_weight),
            "hand_avg_cost": None if hand_cost is None else float(hand_cost / max(1, len(hand_path))),
            "pose_avg_cost": None if pose_cost is None else float(pose_cost / max(1, len(pose_path))),
            "combined_step_costs": combined_steps,
            "hand_step_costs": hand_steps,
            "pose_step_costs": pose_steps,
            "pairs_compared": int(len(combined_steps)),
            "hand_pairs": int(len(hand_path)),
            "pose_pairs": int(len(pose_path)),
        }
        return results

    def visualize(self, results):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        if results["combined_step_costs"]:
            y = results["combined_step_costs"]
            axes[0].plot(y, linewidth=2, label="Combinado (DTW)")
            avg = float(np.mean(y))
            axes[0].axhline(avg, linestyle="--", linewidth=2, label=f"Média: {avg:.3f}")
            axes[0].set_title("Custo por passo no caminho DTW (combinado)")
            axes[0].set_xlabel("Índice no caminho DTW")
            axes[0].set_ylabel("Custo")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        if results["hand_step_costs"] or results["pose_step_costs"]:
            if results["hand_step_costs"]:
                axes[1].plot(results["hand_step_costs"], linewidth=2,
                             label=f"Mãos ({results['hand_weight']*100:.0f}%)")
            if results["pose_step_costs"]:
                axes[1].plot(results["pose_step_costs"], linewidth=2,
                             label=f"Pose ({results['pose_weight']*100:.0f}%)")
            axes[1].set_title("Custo DTW: mãos vs pose")
            axes[1].set_xlabel("Índice no caminho DTW")
            axes[1].set_ylabel("Custo")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        if results["combined_step_costs"]:
            axes[2].hist(results["combined_step_costs"], bins=30, edgecolor="black", alpha=0.7)
            avg = float(np.mean(results["combined_step_costs"]))
            axes[2].axvline(avg, linestyle="--", linewidth=2, label="Média")
            axes[2].set_title("Distribuição do custo (DTW)")
            axes[2].set_xlabel("Custo")
            axes[2].set_ylabel("Frequência")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

        plt.tight_layout()
        plt.show()


# -----------------------------
# UI e main
# -----------------------------

def pick_json(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Arquivos JSON", "*.json"), ("Todos os arquivos", "*.*")]
    )
    try:
        root.destroy()
    except:
        pass
    return path

def interpret(sim):
    # thresholds mais honestos pra esse tipo de problema
    if sim > 0.92:
        return "PRATICAMENTE IDÊNTICOS"
    if sim > 0.80:
        return "MUITO SIMILARES"
    if sim > 0.65:
        return "SIMILARES"
    if sim > 0.50:
        return "ALGUMA SEMELHANÇA"
    return "DIFERENTES"

def main():
    print("=" * 60)
    print("      COMPARADOR DE GESTOS 2.0 (Corpo-base + DTW)")
    print("=" * 60)
    print()

    print("Configuração de pesos:")
    print("1. Padrão (Mãos: 70%, Pose: 30%)")
    print("2. Mãos dominantes (Mãos: 85%, Pose: 15%)")
    print("3. Apenas mãos (100/0)")
    print("4. Apenas pose (0/100)")
    print("5. Personalizado")

    choice = input("\nEscolha (1-5) [padrão: 1]: ").strip()

    if choice == "2":
        comp = GestureComparator2(hand_weight=0.85, pose_weight=0.15, dtw_window_ratio=0.2)
    elif choice == "3":
        comp = GestureComparator2(hand_weight=1.0, pose_weight=0.0, dtw_window_ratio=0.2)
    elif choice == "4":
        comp = GestureComparator2(hand_weight=0.0, pose_weight=1.0, dtw_window_ratio=0.2)
    elif choice == "5":
        try:
            hw = float(input("Peso mãos (0-1): "))
            pw = float(input("Peso pose (0-1): "))
            wr = float(input("DTW window ratio (0.05 a 0.4) [0.2]: ") or "0.2")
            comp = GestureComparator2(hand_weight=hw, pose_weight=pw, dtw_window_ratio=wr)
        except:
            comp = GestureComparator2()
    else:
        comp = GestureComparator2()

    print("\nSelecione o PRIMEIRO JSON...")
    j1 = pick_json("Selecione o primeiro JSON")
    if not j1:
        print("Nenhum arquivo selecionado.")
        return
    print(f"✓ Arquivo 1: {j1.split('/')[-1]}")

    print("\nSelecione o SEGUNDO JSON...")
    j2 = pick_json("Selecione o segundo JSON")
    if not j2:
        print("Nenhum arquivo selecionado.")
        return
    print(f"✓ Arquivo 2: {j2.split('/')[-1]}")

    print("\n" + "=" * 60)
    print("Comparando (DTW + base corporal)...")
    print("=" * 60)

    results = comp.compare(j1, j2, use_pose=True, use_hands=True)

    print("\n" + "─" * 60)
    print("RESULTADOS")
    print("─" * 60)
    print(f"📊 Similaridade geral (ponderada): {results['similarity']:.4f}  -> {interpret(results['similarity'])}")
    print(f"   • Mãos ({results['hand_weight']*100:.0f}%): {results['hand_similarity']:.4f} | avg cost: {results['hand_avg_cost']}")
    print(f"   • Pose ({results['pose_weight']*100:.0f}%): {results['pose_similarity']:.4f} | avg cost: {results['pose_avg_cost']}")
    print(f"🔗 Pares comparados (caminho): {results['pairs_compared']} | mãos: {results['hand_pairs']} | pose: {results['pose_pairs']}")
    print("─" * 60)

    comp.visualize(results)


if __name__ == "__main__":
    main()
