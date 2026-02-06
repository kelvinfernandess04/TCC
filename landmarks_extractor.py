import cv2
import mediapipe as mp
import json
import argparse
import os
import sys
import glob

def extract_landmarks(video_path, output_dir="JSONs"):
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Derive output filename from input filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_landmarks.json")

    mp_holistic = mp.solutions.holistic
    
    # Use Holistic model
    # smooth_landmarks=False for stateless extraction (Absolute Stability)
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print(f"Processing {video_path} -> {output_path}...")
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_idx = 0
    data = {}

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        frame_data = {
            "left_hand": [],
            "right_hand": [],
            "pose": []
        }

        if results.left_hand_landmarks:
            frame_data["left_hand"] = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results.left_hand_landmarks.landmark]
            
        if results.right_hand_landmarks:
            frame_data["right_hand"] = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results.right_hand_landmarks.landmark]

        if results.pose_landmarks:
             frame_data["pose"] = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results.pose_landmarks.landmark]

        data[frame_idx] = frame_data
        frame_idx += 1

    holistic.close()
    vid.release()

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    # Scan MP4s in MP4 directory
    mp4_files = glob.glob(os.path.join("MP4", "*.mp4"))
    
    if not mp4_files:
        print("No MP4 files found in the current directory.")
    else:
        print(f"Found {len(mp4_files)} videos. Starting extraction...")
        for video_file in mp4_files:
            extract_landmarks(video_file, output_dir="JSONs")
        print("Batch extraction complete.")
