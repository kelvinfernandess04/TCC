import json
import numpy as np
import matplotlib.pyplot as plt
import os

class UniversalSignComparator:
    def __init__(self, motion_threshold=0.005):
        self.motion_threshold = motion_threshold
        # Weights for DTW Features
        # 0-4: Thumb-Finger Dists (Shape)
        # 5: Hand Width (Shape)
        # 6: Wrist Velocity (Motion)
        # Total 7 features.
        # Shape (6 features) = 70% weight -> 0.116 each
        # Motion (1 feature) = 30% weight -> 0.30
        self.weights = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.1, 0.3])

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
            print("Warning: No active motion detected. Using full video.")
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
                else: features_seq.append(np.zeros(6)) # Adjusted to size 6
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
                
            # Feature Vector [7 dims]
            feat = np.array([d_ti, d_tm, d_tr, d_tp, d_ip, velocity])
            
            # Padding to match weights length if needed, or adjust weights
            # Wait, I defined 7 weights but calculated 6 features?
            # 0: TI, 1: TM, 2: TR, 3: TP, 4: IP. Total 5 Topo.
            # 5: Velocity. Total 6.
            # Let's adjust weights to size 6.
            # Shape (5) = 70% -> 0.14 each.
            # Motion (1) = 30%.
            
            features_seq.append(feat)
            
        return np.array(features_seq)

    def dtw_distance(self, seq1, seq2):
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0: return float('inf')
        
        # Adjust weights
        weights = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.30])
        
        dtw = np.full((n + 1, m + 1), float('inf'))
        dtw[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diff = seq1[i-1] - seq2[j-1]
                # Weighted Euclidean Distance
                cost = np.sqrt(np.sum(weights * (diff ** 2)))
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
                
        return dtw[n, m] / max(n, m) # Normalize by path length

    def plot_segmentation(self, name, vel, start, end):
        plt.figure(figsize=(10, 4))
        plt.plot(vel, label='Velocity')
        plt.axvspan(start, end, color='green', alpha=0.3, label='Active Segment')
        plt.axhline(self.motion_threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f"Temporal Segmentation: {name}")
        plt.legend()
        plt.savefig(f"segmentation_{name}.png")
        plt.close()

    def compare(self, ref_file, test_file):
        print(f"Comparing {ref_file} vs {test_file}...")
        
        # Load
        ref_frames, _ = self.load_data(ref_file)
        test_frames, _ = self.load_data(test_file)
        
        if not ref_frames or not test_frames: return
        
        # Segment
        r_start, r_end, r_vel = self.detect_active_segment(ref_frames)
        t_start, t_end, t_vel = self.detect_active_segment(test_frames)
        
        # Plot Segmentation
        self.plot_segmentation(os.path.basename(ref_file).split('.')[0], r_vel, r_start, r_end)
        self.plot_segmentation(os.path.basename(test_file).split('.')[0], t_vel, t_start, t_end)
        
        # Slice
        ref_active = ref_frames[r_start:r_end+1]
        test_active = test_frames[t_start:t_end+1]
        
        if not ref_active or not test_active:
            print("Error: Empty active segment.")
            return

        # Extract features
        ref_feats = self.extract_features(ref_active)
        test_feats = self.extract_features(test_active)
        
        # DTW
        score = self.dtw_distance(ref_feats, test_feats)
        
        print(f"  > Active Segment Ref : Frames {r_start} to {r_end} ({len(ref_active)} frames)")
        print(f"  > Active Segment Test: Frames {t_start} to {t_end} ({len(test_active)} frames)")
        print(f"  > Similarity Score   : {score:.4f}")
        
        # Interpretation
        # 0.0 is identical.
        # "Abacate" vs "Abacaxi" (Claw vs C) -> Thumb dists will differ significantly.
        return score

if __name__ == "__main__":
    comparator = UniversalSignComparator(motion_threshold=0.005)
    
    ref = "JSONs/abacate_base_boa_landmarks.json"
    targets = [
        "JSONs/abacate_certo1_landmarks.json",
        "JSONs/abacate_certo2_landmarks.json",
        "JSONs/abacaxi_base_landmarks.json"
    ]
    
    print("=== UNIVERSAL COMPARATOR RUN ===")
    results = {}
    for t in targets:
        score = comparator.compare(ref, t)
        results[t] = score
        print("-" * 30)

    print("\nFINAL RANKING (Lower is Better):")
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{score:.4f} : {name}")
