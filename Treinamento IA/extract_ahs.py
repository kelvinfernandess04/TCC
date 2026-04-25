import os
import cv2
import json

# MOCK TENSORFLOW to avoid protobuf clash on MediaPipe import
import sys
from unittest.mock import MagicMock
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.tools'] = MagicMock()
sys.modules['tensorflow.tools.docs'] = MagicMock()

import mediapipe as mp
import glob

# Config paths
BASE_DIR = r"C:\DevTools\Repositories\Faculdade\TCC\Treinamento IA\American Hand Signs Dataset\American Hand Signs Dataset\DataZip\Data"
OUTPUT_DIR = r"C:\DevTools\Repositories\Faculdade\TCC\Treinamento IA\AHS_json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mp_holistic = mp.solutions.holistic

def process_ahs_dataset():
    folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        
        for folder_name in folders:
            label = folder_name
            print(f"Processando classe AHS: {label}")
            
            output_data = {}
            folder_path = os.path.join(BASE_DIR, folder_name)
            
            images = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
            valid_images_count = 0
            
            for img_path in images:
                img_name = os.path.basename(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # Check for hands
                hand_landmarks = None
                if results.right_hand_landmarks:
                    hand_landmarks = results.right_hand_landmarks
                elif results.left_hand_landmarks:
                    hand_landmarks = results.left_hand_landmarks
                
                if hand_landmarks:
                    landmarks_list = []
                    for lm in hand_landmarks.landmark:
                        landmarks_list.append([lm.x, lm.y])
                    
                    item_id = img_name
                    output_data[item_id] = {
                        "labels": [label],
                        "landmarks": [landmarks_list]
                    }
                    valid_images_count += 1
            
            print(f"Classe {label}: {len(images)} imagens encontradas, {valid_images_count} mãos detectadas com sucesso.")
            
            if output_data:
                json_path = os.path.join(OUTPUT_DIR, f"{label}.json")
                with open(json_path, 'w') as f:
                    json.dump(output_data, f)

if __name__ == "__main__":
    process_ahs_dataset()
    print("Extração finalizada com sucesso!")
