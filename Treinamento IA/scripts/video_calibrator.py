import os
import json
import math
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'calibration_settings.json')

def vec_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))

def joint_flexion(p_prev, p_joint, p_next):
    v1 = p_prev - p_joint
    v2 = p_next - p_joint
    return 180.0 - vec_angle(v1, v2)

class VideoRangeCalibrator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process_video(self, video_path):
        if not os.path.exists(video_path):
            return False, f"Arquivo de vídeo não encontrado: {video_path}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Não foi possível abrir o arquivo de vídeo: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[ANÁLISE DE VÍDEO] Processando {total_frames} quadros de: {video_path}")
        
        json_path = video_path.replace('.mp4', '_landmarks.json')
        video_landmarks = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    video_landmarks = json.load(f)
                print(f"[LOG] Carregados {len(video_landmarks)} quadros de landmarks do JSON.")
            except Exception as e:
                print(f"[ERRO] Erro ao ler arquivo JSON: {e}. Usando fallback do MediaPipe.")
        else:
            print(f"[AVISO] Arquivo JSON de landmarks não encontrado ({json_path}). Usando fallback do MediaPipe.")

        frame_data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            has_landmarks = False
            pts_raw = None
            if video_landmarks and frame_idx < len(video_landmarks):
                if video_landmarks[frame_idx] is not None:
                    has_landmarks = True
                    pts_raw = np.array([[lm['x'] * w, lm['y'] * h, lm['z'] * w] for lm in video_landmarks[frame_idx]])
            else:
                # Fallback to MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    has_landmarks = True
                    pts_raw = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in res.multi_hand_landmarks[0].landmark])

            if has_landmarks:
                wrist = pts_raw[0]
                palm_len = np.linalg.norm(pts_raw[9] - wrist)
                if palm_len > 1e-6:
                    pts_norm = (pts_raw - wrist) / palm_len

                    # Compute finger flexions
                    # Index
                    idx_mcp = joint_flexion(pts_norm[0], pts_norm[5], pts_norm[6])
                    idx_pip = joint_flexion(pts_norm[5], pts_norm[6], pts_norm[7])
                    idx_dip = joint_flexion(pts_norm[6], pts_norm[7], pts_norm[8])
                    idx_tot = idx_mcp + idx_pip + idx_dip

                    # Middle
                    mid_mcp = joint_flexion(pts_norm[0], pts_norm[9], pts_norm[10])
                    mid_pip = joint_flexion(pts_norm[9], pts_norm[10], pts_norm[11])
                    mid_dip = joint_flexion(pts_norm[10], pts_norm[11], pts_norm[12])
                    mid_tot = mid_mcp + mid_pip + mid_dip

                    # Ring
                    rng_mcp = joint_flexion(pts_norm[0], pts_norm[13], pts_norm[14])
                    rng_pip = joint_flexion(pts_norm[13], pts_norm[14], pts_norm[15])
                    rng_dip = joint_flexion(pts_norm[14], pts_norm[15], pts_norm[16])
                    rng_tot = rng_mcp + rng_pip + rng_dip

                    # Pinky
                    pnk_mcp = joint_flexion(pts_norm[0], pts_norm[17], pts_norm[18])
                    pnk_pip = joint_flexion(pts_norm[17], pts_norm[18], pts_norm[19])
                    pnk_dip = joint_flexion(pts_norm[18], pts_norm[19], pts_norm[20])
                    pnk_tot = pnk_mcp + pnk_pip + pnk_dip

                    # Thumb
                    thm_cmc = joint_flexion(pts_norm[0], pts_norm[1], pts_norm[2])
                    thm_mcp = joint_flexion(pts_norm[1], pts_norm[2], pts_norm[3])
                    thm_ip  = joint_flexion(pts_norm[2], pts_norm[3], pts_norm[4])

                    # Spreads
                    sp_pnk_rng = vec_angle(pts_norm[17] - pts_norm[0], pts_norm[13] - pts_norm[0])
                    sp_rng_mid = vec_angle(pts_norm[13] - pts_norm[0], pts_norm[9] - pts_norm[0])
                    sp_mid_idx = vec_angle(pts_norm[9] - pts_norm[0], pts_norm[5] - pts_norm[0])
                    sp_idx_thm = vec_angle(pts_norm[5] - pts_norm[0], pts_norm[1] - pts_norm[0])
                    total_spread = sp_pnk_rng + sp_rng_mid + sp_mid_idx + sp_idx_thm

                    # Thumb Opposition (Distance from Thumb Tip to Middle MCP)
                    dist_opp = np.linalg.norm(pts_norm[4] - pts_norm[9])

                    frame_data.append({
                        'frame_idx': frame_idx,
                        'pts': pts_norm,
                        'idx_tot': idx_tot, 'mid_tot': mid_tot, 'rng_tot': rng_tot, 'pnk_tot': pnk_tot,
                        'avg_finger_tot': (idx_tot + mid_tot + rng_tot + pnk_tot) / 4.0,
                        'idx_mcp': idx_mcp, 'idx_pip': idx_pip,
                        'mid_mcp': mid_mcp, 'mid_pip': mid_pip,
                        'rng_mcp': rng_mcp, 'rng_pip': rng_pip,
                        'pnk_mcp': pnk_mcp, 'pnk_pip': pnk_pip,
                        'thm_cmc': thm_cmc, 'thm_mcp': thm_mcp, 'thm_ip': thm_ip,
                        'total_spread': total_spread,
                        'sp_mid_idx': sp_mid_idx,
                        'dist_opp': dist_opp
                    })

            frame_idx += 1

        cap.release()

        if not frame_data:
            return False, "Nenhuma mão foi detectada no vídeo gravado. Certifique-se de boa iluminação e mão visível."

        print(f"[ANÁLISE DE VÍDEO] {len(frame_data)} quadros válidos com landmarks detectados.")

        # Identify Keyframes
        # 1. Stage 0 Spread (Mão aberta em leque máximo: baixa flexão, alto spread)
        open_frames = [f for f in frame_data if f['avg_finger_tot'] < 60.0]
        if not open_frames:
            open_frames = sorted(frame_data, key=lambda x: x['avg_finger_tot'])[:10]
        kf_stage_0_spread = max(open_frames, key=lambda x: x['total_spread'])

        # 2. Stage 0 Closed (Mão aberta com dedos juntos: baixa flexão, menor spread)
        kf_stage_0_closed = min(open_frames, key=lambda x: x['total_spread'])

        # 3. Stage 3 (Punho fechado: flexão máxima total)
        kf_stage_3 = max(frame_data, key=lambda x: x['avg_finger_tot'])

        # 4. Stage 1 (Garra leve: flexão média ~100-140)
        target_claw = 120.0
        kf_stage_1 = min(frame_data, key=lambda x: abs(x['avg_finger_tot'] - target_claw))

        # 5. Stage 2 (Plataforma/Gancho: MCP reto, PIP/DIP dobrado)
        hook_candidates = [f for f in frame_data if f['mid_mcp'] < 40.0]
        if hook_candidates:
            kf_stage_2 = max(hook_candidates, key=lambda x: x['mid_pip'])
        else:
            kf_stage_2 = min(frame_data, key=lambda x: abs(x['avg_finger_tot'] - 180.0))

        # 6. Thumb Opposition (Menor distância entre ponta do polegar e centro da palma)
        kf_thumb_opp = min(frame_data, key=lambda x: x['dist_opp'])

        # 7. Thumb IP flexed (Maior flexão do IP com MCP estendido)
        thumb_candidates = [f for f in frame_data if f['thm_mcp'] < 35.0]
        if thumb_candidates:
            kf_thumb_ip = max(thumb_candidates, key=lambda x: x['thm_ip'])
        else:
            kf_thumb_ip = max(frame_data, key=lambda x: x['thm_ip'])

        # Extract Proportions from best open frame
        ref_pts = kf_stage_0_spread['pts']
        avg_lengths = {
            'Thumb':  [float(np.linalg.norm(ref_pts[2] - ref_pts[1])),
                       float(np.linalg.norm(ref_pts[3] - ref_pts[2])),
                       float(np.linalg.norm(ref_pts[4] - ref_pts[3]))],
            'Index':  [float(np.linalg.norm(ref_pts[6] - ref_pts[5])),
                       float(np.linalg.norm(ref_pts[7] - ref_pts[6])),
                       float(np.linalg.norm(ref_pts[8] - ref_pts[7]))],
            'Middle': [float(np.linalg.norm(ref_pts[10] - ref_pts[9])),
                       float(np.linalg.norm(ref_pts[11] - ref_pts[10])),
                       float(np.linalg.norm(ref_pts[12] - ref_pts[11]))],
            'Ring':   [float(np.linalg.norm(ref_pts[14] - ref_pts[13])),
                       float(np.linalg.norm(ref_pts[15] - ref_pts[14])),
                       float(np.linalg.norm(ref_pts[16] - ref_pts[15]))],
            'Pinky':  [float(np.linalg.norm(ref_pts[18] - ref_pts[17])),
                       float(np.linalg.norm(ref_pts[19] - ref_pts[18])),
                       float(np.linalg.norm(ref_pts[20] - ref_pts[19]))]
        }
        avg_palm = {
            'Thumb':  float(np.linalg.norm(ref_pts[1] - ref_pts[0])),
            'Index':  float(np.linalg.norm(ref_pts[5] - ref_pts[0])),
            'Middle': float(np.linalg.norm(ref_pts[9] - ref_pts[0])),
            'Ring':   float(np.linalg.norm(ref_pts[13] - ref_pts[0])),
            'Pinky':  float(np.linalg.norm(ref_pts[17] - ref_pts[0]))
        }

        # Build Calibration Package
        calib_data = {
            "captured_poses": {
                "stage_0_spread": {"front": kf_stage_0_spread['pts'].tolist(), "profile": None},
                "stage_0_closed": {"front": kf_stage_0_closed['pts'].tolist(), "profile": None},
                "stage_1":         {"front": kf_stage_1['pts'].tolist(), "profile": None},
                "stage_2":         {"front": kf_stage_2['pts'].tolist(), "profile": None},
                "stage_3":         {"front": kf_stage_3['pts'].tolist(), "profile": None},
                "thumb_opposition":{"front": kf_thumb_opp['pts'].tolist(), "profile": None},
                "thumb_ip_flexed": {"front": kf_thumb_ip['pts'].tolist(), "profile": None}
            },
            "avg_lengths": avg_lengths,
            "avg_palm": avg_palm,
            "video_metadata": {
                "source_video": video_path,
                "total_frames_analyzed": len(frame_data),
                "keyframe_indices": {
                    "stage_0_spread": kf_stage_0_spread['frame_idx'],
                    "stage_0_closed": kf_stage_0_closed['frame_idx'],
                    "stage_1": kf_stage_1['frame_idx'],
                    "stage_2": kf_stage_2['frame_idx'],
                    "stage_3": kf_stage_3['frame_idx'],
                    "thumb_opposition": kf_thumb_opp['frame_idx'],
                    "thumb_ip_flexed": kf_thumb_ip['frame_idx']
                }
            }
        }

        # Save to JSON
        with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(calib_data, f, indent=2)

        summary = (
            f"• Quadros analisados: {len(frame_data)}\n"
            f"• Keyframe Mão Espalmada Aberta: Quadro #{kf_stage_0_spread['frame_idx']}\n"
            f"• Keyframe Mão Dedos Juntos: Quadro #{kf_stage_0_closed['frame_idx']}\n"
            f"• Keyframe Garra (Estágio 1): Quadro #{kf_stage_1['frame_idx']}\n"
            f"• Keyframe Gancho (Estágio 2): Quadro #{kf_stage_2['frame_idx']}\n"
            f"• Keyframe Punho (Estágio 3): Quadro #{kf_stage_3['frame_idx']}\n"
            f"• Keyframe Oposição do Polegar: Quadro #{kf_thumb_opp['frame_idx']}\n"
            f"• Proporções e Landmarks salvos em: calibration_settings.json"
        )
        print(f"[SUCESSO]\n{summary}")
        return True, summary

if __name__ == "__main__":
    import sys
    calibrator = VideoRangeCalibrator()
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
    else:
        # Check if there is any recording in recordings folder
        recordings = sorted([
            os.path.join(RECORDINGS_DIR, f) for f in os.listdir(RECORDINGS_DIR) if f.endswith('.mp4')
        ])
        if recordings:
            v_path = recordings[-1]
            print(f"[AUTO] Selecionando vídeo mais recente: {v_path}")
        else:
            print("Uso: python video_calibrator.py <caminho_do_video.mp4>")
            sys.exit(1)

    ok, msg = calibrator.process_video(v_path)
    print(msg)
