import cv2
import mediapipe as mp
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

vid = cv2.VideoCapture('abacaxi.mp4')
frame_idx = 0
data = {}

while True:
    ret, frame = vid.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    frame_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
            label = handedness.classification[0].label
            frame_data.append({'handedness': label, 'landmarks': lm_list})

    data[frame_idx] = frame_data
    frame_idx += 1

hands.close()
vid.release()

with open('abacaxi_hand_landmarks.json', 'w') as f:
    json.dump(data, f, indent=2)
