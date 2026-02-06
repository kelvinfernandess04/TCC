import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.distance import cdist

# Pure Numpy implementation to avoid pandas dependency

class LibrasValidator:
    def __init__(self):
        # Weights (The Jury)
        # 60% Shape (5 dims), 40% Motion (3 dims)
        w_s = 0.12
        w_m = 0.1333
        self.weights = np.array([w_s, w_s, w_s, w_s, w_s, w_m, w_m, w_m])

    def load_data(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return None, None

        frames = []
        sorted_keys = sorted(data.keys(), key=lambda x: int(x))
        for k in sorted_keys:
            frames.append(data[k])
        return frames

    def get_global_scalars(self, frames):
        """
        Calculates Global Median for Shoulder Width and Palm Size.
        """
        shoulder_widths = []
        palm_sizes_l = []
        palm_sizes_r = []
        
        for frame in frames:
            pose = frame.get('pose')
            if pose:
                p11 = np.array([pose[11]['x'], pose[11]['y']])
                p12 = np.array([pose[12]['x'], pose[12]['y']])
                sw = np.linalg.norm(p11 - p12)
                if sw > 1e-6: shoulder_widths.append(sw)
                
            lh = frame.get('left_hand')
            if lh:
                lw = np.array([lh[0]['x'], lh[0]['y']])
                lm = np.array([lh[9]['x'], lh[9]['y']])
                ps = np.linalg.norm(lw - lm)
                if ps > 1e-6: palm_sizes_l.append(ps)
                
            rh = frame.get('right_hand')
            if rh:
                rw = np.array([rh[0]['x'], rh[0]['y']])
                rm = np.array([rh[9]['x'], rh[9]['y']])
                ps = np.linalg.norm(rw - rm)
                if ps > 1e-6: palm_sizes_r.append(ps)
                
        global_sw = np.median(shoulder_widths) if shoulder_widths else 1.0
        global_pl = np.median(palm_sizes_l) if palm_sizes_l else 1.0
        global_pr = np.median(palm_sizes_r) if palm_sizes_r else 1.0
        
        return global_sw, global_pl, global_pr

    def get_hand_features(self, hand_lms, pose_lms, shoulder_width, palm_size, prev_hand_lms=None):
        if not hand_lms: return None 

        h = np.array([[lm['x'], lm['y']] for lm in hand_lms])
        wrist = h[0]
        
        # 1. SHAPE (Topology)
        topo_feats = []
        relevant_tips = [4, 8, 12, 16, 20]
        for idx in relevant_tips:
            dist = np.linalg.norm(wrist - h[idx])
            topo_feats.append(dist / palm_size)
            
        # 2. MOTION (Location + Direction)
        nose = np.array([pose_lms[0]['x'], pose_lms[0]['y']])
        loc_dist = np.linalg.norm(wrist - nose) / shoulder_width
        
        direction = np.array([0.0, 0.0])
        if prev_hand_lms:
            prev_wrist = np.array([prev_hand_lms[0]['x'], prev_hand_lms[0]['y']])
            diff = wrist - prev_wrist
            magnitude = np.linalg.norm(diff)
            # Stabilization threshold
            if magnitude > (0.01 * shoulder_width):
                direction = diff / magnitude
            
        return np.concatenate([topo_feats, [loc_dist], direction])

    def numpy_interpolate(self, arr):
        """
        Linear interpolation for 2D numpy array (column-wise).
        Handles NaNs.
        """
        out = arr.copy()
        for col_idx in range(out.shape[1]):
            col = out[:, col_idx]
            nans = np.isnan(col)
            # Use X indices for valid values
            x = lambda z: z.nonzero()[0]
            
            if np.sum(~nans) > 0 and np.sum(nans) > 0:
                col[nans] = np.interp(x(nans), x(~nans), col[~nans])
                out[:, col_idx] = col
                
        # Fill edges with nearest (ffill/bfill behavior equivalent)
        # Verify any remaining NaNs (leading/trailing if interp range is limited?)
        # np.interp does extrapolate or clamp? Default is constant extrapolation? 
        # Actually np.interp clamps to edge values by default. So ffill/bfill is automatic.
        return out

    def numpy_sma(self, arr, window=3):
        """
        Simple Moving Average on columns.
        """
        if len(arr) < window: return arr
        
        kernel = np.ones(window) / window
        out = arr.copy()
        
        for col_idx in range(out.shape[1]):
            # mode='same' keeps size, check boundary effects
            out[:, col_idx] = np.convolve(arr[:, col_idx], kernel, mode='same')
            
            # Convolve 'same' has edge artifacts (zero padding). 
            # Ideally we want valid padding or mirror.
            # For simplicity in this context, acceptable.
            
        return out

    def process_sequence(self, frames):
        g_sw, g_pl, g_pr = self.get_global_scalars(frames)
        
        has_left = any(f.get('left_hand') for f in frames)
        has_right = any(f.get('right_hand') for f in frames)
        
        raw_seq = []
        
        for i, frame in enumerate(frames):
            pose = frame.get('pose')
            if not pose: 
                row_dim = (8 if has_left else 0) + (8 if has_right else 0)
                raw_seq.append(np.full(row_dim, np.nan))
                continue
                
            lh = frame.get('left_hand')
            rh = frame.get('right_hand')
            
            prev_frame = frames[i-1] if i>0 else None
            prev_lh = prev_frame.get('left_hand') if prev_frame else None
            prev_rh = prev_frame.get('right_hand') if prev_frame else None
            
            row = []
            
            if has_left:
                feats = self.get_hand_features(lh, pose, g_sw, g_pl, prev_lh)
                if feats is not None:
                    row.append(feats)
                else:
                    row.append(np.full(8, np.nan))
                    
            if has_right:
                feats = self.get_hand_features(rh, pose, g_sw, g_pr, prev_rh)
                if feats is not None:
                    row.append(feats)
                else:
                    row.append(np.full(8, np.nan))
            
            if not row:
                raw_seq.append(np.full((8 if has_left else 0) + (8 if has_right else 0), np.nan))
                continue

            raw_seq.append(np.concatenate(row))
            
        # To Numpy
        data_matrix = np.array(raw_seq)
        
        # 3. Interpolation
        if np.isnan(data_matrix).all():
            return np.array([]), (has_left, has_right)
            
        data_matrix = self.numpy_interpolate(data_matrix)
        # Safety fill for all-nan columns if any
        data_matrix = np.nan_to_num(data_matrix) 
        
        # 4. Smoothing
        data_matrix = self.numpy_sma(data_matrix, window=3)
        
        # 5. Zero-Delta First Frame
        if len(data_matrix) > 0:
            if has_left:
                data_matrix[0, 6:8] = 0.0
            if has_right:
                offset = 8 if has_left else 0
                data_matrix[0, offset+6:offset+8] = 0.0
                
        return data_matrix, (has_left, has_right)

    def subsequence_dtw(self, ref, target, config):
        N, D_dim = ref.shape
        M, _ = target.shape
        
        w_vec = []
        if config[0]: w_vec.append(self.weights)
        if config[1]: w_vec.append(self.weights)
        full_weights = np.concatenate(w_vec)
        
        if len(full_weights) != D_dim: full_weights = np.ones(D_dim) 

        w_sqrt = np.sqrt(full_weights)
        ref_w = ref * w_sqrt
        tgt_w = target * w_sqrt
        
        dist_mat = cdist(ref_w, tgt_w, metric='euclidean')
        
        D = np.full((N + 1, M + 1), np.inf)
        D[0, :] = 0 
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                cost = dist_mat[i-1, j-1]
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
                
        best_end = np.argmin(D[N, :])
        best_cost = D[N, best_end]
        norm_score = best_cost / N
        
        path = []
        i, j = N, best_end
        if j == 0: return norm_score, 0, 0
        
        while i > 0:
            path.append((i-1, j-1))
            candidates = [
                (i-1, j-1, D[i-1, j-1]),
                (i-1, j,   D[i-1, j]),
                (i,   j-1, D[i, j-1])
            ]
            candidates.sort(key=lambda x: x[2])
            i, j = candidates[0][0], candidates[0][1]
            if j < 0: break
            
        path.reverse()
        start = path[0][1] if path else 0
        end = best_end
        
        return norm_score, start, end

    def plot_validation(self, ref_name, tgt_name, feat_seq, s, e, score, out_dir):
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        plt.figure(figsize=(10, 5))
        plt.plot(feat_seq[:, 0], color='lightgray', label='Start Stream')
        if e > s:
            plt.plot(range(s, e), feat_seq[s:e, 0], color='green', linewidth=2, label='Matched')
        plt.title(f"{ref_name} vs {tgt_name} | Score: {score:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/{ref_name}_VS_{tgt_name}.png")
        plt.close()

    def validate(self, ref_path, tgt_path, out_dir):
        ref_f = self.load_data(ref_path)
        tgt_f = self.load_data(tgt_path)
        
        if not ref_f or not tgt_f: return 999.0
        
        r_feat, r_cfg = self.process_sequence(ref_f)
        t_feat, t_cfg = self.process_sequence(tgt_f)
        
        if len(r_feat) == 0 or len(t_feat) == 0: return 999.0
        
        if r_feat.shape[1] != t_feat.shape[1]:
            return 888.0 
            
        score, s, e = self.subsequence_dtw(r_feat, t_feat, r_cfg)
        
        ref_name = os.path.basename(ref_path).replace('.json', '').replace('_landmarks', '')
        tgt_name = os.path.basename(tgt_path).replace('.json', '').replace('_landmarks', '')
        
        self.plot_validation(ref_name, tgt_name, t_feat, s, e, score, out_dir)
        
        return score

if __name__ == "__main__":
    validator = LibrasValidator()
    files = glob.glob(os.path.join("JSONs", "*.json"))
    refs = [f for f in files if "_base_" in f]
    targets = files 
    
    # Pre-clean output dir? No, overwrite is fine.
    
    full_report = {}
    
    print("=== LIBRAS VALIDATOR: ABSOLUTE STABILITY (NP) ===")
    
    for ref in refs:
        ref_name = os.path.basename(ref).replace('.json', '').replace('_landmarks', '')
        full_report[ref_name] = []
        print(f"Processing Reference: {ref_name}...")
        
        self_score = validator.validate(ref, ref, "comparison_plots")
        print(f"[{self_score:.4f}] {ref_name} vs {ref_name}")
        
        for target in targets:
            target_name = os.path.basename(target).replace('.json', '').replace('_landmarks', '')
            try:
                score = validator.validate(ref, target, "comparison_plots")
                status = "APROVADO" if score < 0.6 else "REPROVADO"
                full_report[ref_name].append((target_name, score, status))
            except Exception as e:
                full_report[ref_name].append((target_name, 999.0, f"Error"))

    print("\n" + "="*50)
    print("FINAL BATCH REPORT")
    print("="*50)
    for ref_name, results in full_report.items():
        print(f"\n=== RESULTADOS PARA: {ref_name} ===")
        for target_name, score, status in sorted(results, key=lambda x: x[1]):
            print(f"[{score:.4f}] {target_name} ({status})")
