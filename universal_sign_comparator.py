import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

class UniversalSignComparator:
    def __init__(self, motion_threshold=0.005):
        self.motion_threshold = motion_threshold
        # Weights for DTW Features
        # 0-4: Thumb-Finger Dists (Shape)
        # 5: Hand Width (Shape)
        # 6: Wrist Velocity (Motion)
        # Total 7 features (indices 0-6).
        # Actually in code below:
        # Feat 0-4: Topology (5 items)
        # Feat 5: Velocity (1 item) -> This seems to be the logic in the current file (6 dims total)
        # Let's verify strict alignment with the "approved core".
        # Current file (Step 102/113):
        # feat = np.array([d_ti, d_tm, d_tr, d_tp, d_ip, velocity]) -> 6 dims
        # weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.30]) -> 6 dims
        self.weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.30])

    def load_data(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return None, None

        frames = []
        sorted_keys = sorted(data.keys(), key=int)
        for k in sorted_keys:
            frames.append(data[k])
            
        return frames, sorted_keys

    def detect_active_segment(self, frames):
        """
        Segments the video to keep only the active signing phase.
        Based on Wrist Velocity.
        """
        velocities = []
        # Calculate velocity profile
        for i in range(1, len(frames)):
            curr = frames[i].get('right_hand')
            prev = frames[i-1].get('right_hand')
            
            # Check for None or Empty list
            if not curr or not prev:
                velocities.append(0.0)
                continue
                
            # Wrist is index 0
            c_wrist = np.array([curr[0]['x'], curr[0]['y']])
            p_wrist = np.array([prev[0]['x'], prev[0]['y']])
            
            # Simple 2D velocity (ignoring depth for segmentation robustness)
            vel = np.linalg.norm(c_wrist - p_wrist)
            velocities.append(vel)
            
        # Pad velocities to match frames length
        velocities = [0.0] + velocities
        
        # Smooth velocity (Moving Average)
        window_size = 5
        smoothed_vel = np.convolve(velocities, np.ones(window_size)/window_size, mode='same')
        
        # Thresholding
        active_indices = [i for i, v in enumerate(smoothed_vel) if v > self.motion_threshold]
        
        if not active_indices:
            # print("Warning: No active motion detected. Using full video.")
            return 0, len(frames)-1, smoothed_vel
            
        start = max(0, active_indices[0] - 5) # Pad 5 frames
        end = min(len(frames)-1, active_indices[-1] + 5)
        
        return start, end, smoothed_vel

    def calculate_palm_scale(self, hand_landmarks):
        """
        Calculates local scale: Distance from Wrist (0) to Middle MCP (9).
        Used to normalize all other distances.
        """
        if not hand_landmarks or len(hand_landmarks) < 21: return 1.0
        
        wrist = np.array([hand_landmarks[0]['x'], hand_landmarks[0]['y'], hand_landmarks[0]['z']])
        middle_mcp = np.array([hand_landmarks[9]['x'], hand_landmarks[9]['y'], hand_landmarks[9]['z']])
        
        dist = np.linalg.norm(wrist - middle_mcp)
        return dist if dist > 1e-6 else 1.0

    def extract_features(self, frames):
        features_seq = []
        
        for i, frame in enumerate(frames):
            rh = frame.get('right_hand')
            
            # Check for validity and length
            if not rh or len(rh) < 21: 
                # Handle missing hand? Interpolate or skip. 
                # For now, replicate previous or zeros.
                if features_seq: features_seq.append(features_seq[-1])
                else: features_seq.append(np.zeros(6))
                continue

            # Convert to numpy
            lms = np.array([[p['x'], p['y'], p['z']] for p in rh])
            
            # 1. Palm Scale
            scale = self.calculate_palm_scale(rh)
            
            # 2. Shape Features (Topology) - Local & Scaled
            # Tips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
            thumb_tip = lms[4]
            index_tip = lms[8]
            middle_tip = lms[12]
            ring_tip = lms[16]
            pinky_tip = lms[20]
            
            # Distances from Thumb Tip to others (The "Claw" signature)
            d_ti = np.linalg.norm(thumb_tip - index_tip) / scale
            d_tm = np.linalg.norm(thumb_tip - middle_tip) / scale
            d_tr = np.linalg.norm(thumb_tip - ring_tip) / scale
            d_tp = np.linalg.norm(thumb_tip - pinky_tip) / scale
            
            # Hand Width (Index to Pinky)
            d_ip = np.linalg.norm(index_tip - pinky_tip) / scale
            
            # 3. Motion Feature (Velocity)
            # Velocity relative to palm scale (Zoom invariant speed)
            if i > 0 and frames[i-1].get('right_hand'):
                prev_rh = frames[i-1]['right_hand']
                prev_wrist = np.array([prev_rh[0]['x'], prev_rh[0]['y'], prev_rh[0]['z']])
                curr_wrist = lms[0]
                velocity = np.linalg.norm(curr_wrist - prev_wrist) / scale
            else:
                velocity = 0.0
                
            # Feature Vector [6 dims]
            feat = np.array([d_ti, d_tm, d_tr, d_tp, d_ip, velocity])
            
            features_seq.append(feat)
            
        return np.array(features_seq)

    def dtw_distance(self, seq1, seq2):
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0: return float('inf')
        
        # Weights for the 6 features
        weights = self.weights
        
        dtw = np.full((n + 1, m + 1), float('inf'))
        dtw[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diff = seq1[i-1] - seq2[j-1]
                # Weighted Euclidean Distance
                cost = np.sqrt(np.sum(weights * (diff ** 2)))
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
                
        return dtw[n, m] / max(n, m) # Normalize by path length

    def plot_comparison(self, ref_name, target_name, ref_vel, target_vel):
        """
        Plots both velocity profiles on the same chart, normalized by time (0% to 100%).
        Saves the plot to disk.
        """
        plt.figure(figsize=(10, 6))
        
        # Normalize time axis
        ref_time = np.linspace(0, 1, len(ref_vel))
        target_time = np.linspace(0, 1, len(target_vel))
        
        plt.plot(ref_time, ref_vel, label=f'Ref: {ref_name}', color='blue', linewidth=2)
        plt.plot(target_time, target_vel, label=f'Target: {target_name}', color='orange', linestyle='--', linewidth=2)
        
        plt.title(f"Comparison: {ref_name} vs {target_name}")
        plt.xlabel("Normalized Time (0-100%)")
        plt.ylabel("Velocity (Normalized by Palm)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        filename = f"comparison_plots/{ref_name}_VS_{target_name}.png"
        plt.savefig(filename)
        plt.close() # Free memory

    def compare(self, ref_file, test_file):
        # Load
        ref_frames, _ = self.load_data(ref_file)
        test_frames, _ = self.load_data(test_file)
        
        if not ref_frames or not test_frames: return float('inf')
        
        # Segment
        r_start, r_end, r_vel = self.detect_active_segment(ref_frames)
        t_start, t_end, t_vel = self.detect_active_segment(test_frames)
        
        # Slice Active Segments
        ref_active = ref_frames[r_start:r_end+1]
        test_active = test_frames[t_start:t_end+1]
        
        if not ref_active or not test_active:
            return float('inf')

        # Extract features
        ref_feats = self.extract_features(ref_active)
        test_feats = self.extract_features(test_active)
        
        # DTW
        score = self.dtw_distance(ref_feats, test_feats)
        
        # Plot Comparison (using the active segment velocities for visualization)
        # Using index 5 (Velocity) from processed features to show what DTW saw
        r_vel_plot = ref_feats[:, 5]
        t_vel_plot = test_feats[:, 5]
        
        self.plot_comparison(
            os.path.basename(ref_file).split('_landmarks')[0],
            os.path.basename(test_file).split('_landmarks')[0],
            r_vel_plot,
            t_vel_plot
        )
        
        return score

def scan_directory(json_dir):
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    references = []
    targets = []
    
    for f in json_files:
        filename = os.path.basename(f)
        if "_base_" in filename:
            references.append(f)
        # All files are targets
        targets.append(f)
        
    return references, targets

if __name__ == "__main__":
    comparator = UniversalSignComparator(motion_threshold=0.005)
    
    # 1. Setup Directories
    json_dir = "JSONs"
    plot_dir = "comparison_plots"
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # 2. Scan Files
    refs, targets = scan_directory(json_dir)
    
    print(f"Found {len(refs)} References and {len(targets)} Targets.")
    print("-" * 50)
    
    # 3. Batch Compare
    full_report = {}
    
    for ref in refs:
        ref_name = os.path.basename(ref).split('_landmarks')[0]
        full_report[ref_name] = []
        print(f"Processing Reference: {ref_name}...")
        
        for target in targets:
            target_name = os.path.basename(target).split('_landmarks')[0]
            
            # Optional: Skip self-compare vs strict requirements.
            # User said: "compare against all targets"
            # It enables checking baseline = 0.0
            
            score = comparator.compare(ref, target)
            full_report[ref_name].append((target_name, score))

    # 4. Final Report
    print("\n" + "="*50)
    print("FINAL BATCH REPORT")
    print("="*50)
    
    for ref_name, results in full_report.items():
        print(f"\n=== RESULTADOS PARA: {ref_name} ===")
        # Sort by score (Lower is better)
        sorted_results = sorted(results, key=lambda x: x[1])
        
        for target_name, score in sorted_results:
            # Heuristic for Print
            status = "APROVADO" if score < 0.3 else "REPROVADO"
            print(f"[{score:.4f}] {target_name} ({status})")
