import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Standard MediaPipe Hand Connections (Indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} frames.")
    return data

def update_graph(num, data_keys, data, sc_left, sc_right, lines_left, lines_right, ax):
    frame_idx = data_keys[num]
    frame_content = data[frame_idx]

    # Initialize empty data
    left_xs, left_ys = [], []
    right_xs, right_ys = [], []

    # Left Hand
    if frame_content.get("left_hand"):
        left_xs = [lm['x'] for lm in frame_content["left_hand"]]
        left_ys = [-lm['y'] for lm in frame_content["left_hand"]]
    
    # Right Hand
    if frame_content.get("right_hand"):
        right_xs = [lm['x'] for lm in frame_content["right_hand"]]
        right_ys = [-lm['y'] for lm in frame_content["right_hand"]]

    # Update Scatters
    if left_xs:
        sc_left.set_offsets(np.c_[left_xs, left_ys])
        for line, (start_idx, end_idx) in zip(lines_left, HAND_CONNECTIONS):
            line.set_data([left_xs[start_idx], left_xs[end_idx]], [left_ys[start_idx], left_ys[end_idx]])
    else:
        sc_left.set_offsets(np.empty((0, 2)))
        for line in lines_left:
            line.set_data([], [])

    if right_xs:
        sc_right.set_offsets(np.c_[right_xs, right_ys])
        for line, (start_idx, end_idx) in zip(lines_right, HAND_CONNECTIONS):
            line.set_data([right_xs[start_idx], right_xs[end_idx]], [right_ys[start_idx], right_ys[end_idx]])
    else:
        sc_right.set_offsets(np.empty((0, 2)))
        for line in lines_right:
             line.set_data([], [])

    ax.set_title(f"Frame {frame_idx}")
    return sc_left, sc_right, *lines_left, *lines_right

def visualize(json_path):
    data = load_data(json_path)
    sorted_keys = sorted(data.keys(), key=int)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Initialize objects
    # Left Hand (Blue)
    sc_left = ax.scatter([], [], c='b', s=20, label='Left Hand')
    lines_left = [ax.plot([], [], 'b-')[0] for _ in HAND_CONNECTIONS]
    
    # Right Hand (Red)
    sc_right = ax.scatter([], [], c='r', s=20, label='Right Hand')
    lines_right = [ax.plot([], [], 'r-')[0] for _ in HAND_CONNECTIONS]
    
    ax.legend()

    ani = animation.FuncAnimation(
        fig, 
        update_graph, 
        frames=len(sorted_keys), 
        fargs=(sorted_keys, data, sc_left, sc_right, lines_left, lines_right, ax),
        interval=50,
        blit=False
    )

    print("Showing animation... (Close window to exit)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand landmarks from JSON.")
    parser.add_argument("json_path", nargs='?', help="Path to JSON file")
    args = parser.parse_args()

    if args.json_path:
        visualize(args.json_path)
    else:
        path = input("Digite o caminho do JSON: ").strip()
        if path:
            visualize(path)
