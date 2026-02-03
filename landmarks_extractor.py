import cv2
import mediapipe as mp
import json
import argparse
import os
import sys

def extract_landmarks(video_path):
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return

    # Derive output filename from input filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base_name}_landmarks.json"

    mp_holistic = mp.solutions.holistic
    
    # Use Holistic model
    # refine_face_landmarks=False for speed if face is not needed for hand signs (usually not needed for hand shape, but facial expression is part of Libras. Leaving false for now to focus on hands).
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print(f"Processing {video_path} using Holistic model...")
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
    print(f"Done! Holistic landmarks saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks from a video file.")
    parser.add_argument("video_path", nargs='?', help="Path to input video file")
    
    args = parser.parse_args()

    if args.video_path:
        extract_landmarks(args.video_path)
    else:
        # Fallback to interactive input if no argument provided
        video_path = input("Digite o nome do arquivo de vídeo (ex: video.mp4): ").strip()
        if video_path:
            extract_landmarks(video_path)
        else:
            print("Nenhum arquivo informado.")
