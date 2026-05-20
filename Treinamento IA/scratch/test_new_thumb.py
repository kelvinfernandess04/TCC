import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

def test():
    # Real open/closed vectors
    u01_open = normalize(np.array([-0.0643, 0.0469, -0.0241]))
    u12_open = normalize(np.array([-0.0498, 0.0838, -0.0129]))
    u23_open = normalize(np.array([-0.0331, 0.0665, -0.0128]))
    u34_open = normalize(np.array([-0.0251, 0.0463, -0.0108]))

    u01_closed = normalize(np.array([-0.0500, 0.0197, -0.0436]))
    u12_closed = normalize(np.array([-0.0261, 0.0596, -0.0272]))
    u23_closed = normalize(np.array([0.0385, 0.0456, -0.0238]))
    u34_closed = normalize(np.array([0.0493, 0.0181, -0.0224]))

    # Mock user bone lengths (from print_frame output averages)
    avg_palm_thumb = 0.0831 # distance from 0 to 1
    lengths = [0.1135, 0.0572, 0.0427] # distances for 1->2, 2->3, 3->4

    for state, opp in [(0, 0), (3, 1)]:
        t = 0.0 if state == 0 else 1.0
        opp_t = float(opp)
        mix_t = np.clip(t * 0.8 + opp_t * 0.3, 0.0, 1.0)

        dir_01 = normalize(u01_open * (1.0 - mix_t) + u01_closed * mix_t)
        dir_12 = normalize(u12_open * (1.0 - mix_t) + u12_closed * mix_t)
        dir_23 = normalize(u23_open * (1.0 - mix_t) + u23_closed * mix_t)
        dir_34 = normalize(u34_open * (1.0 - mix_t) + u34_closed * mix_t)

        p1 = dir_01 * avg_palm_thumb
        p2 = p1 + dir_12 * lengths[0]
        p3 = p2 + dir_23 * lengths[1]
        p4 = p3 + dir_34 * lengths[2]

        print(f"\nState={state}, Opp={opp}")
        print(f"Landmark 0: [0, 0, 0]")
        print(f"Landmark 1: {p1}")
        print(f"Landmark 2: {p2}")
        print(f"Landmark 3: {p3}")
        print(f"Landmark 4: {p4}")

if __name__ == "__main__":
    test()
