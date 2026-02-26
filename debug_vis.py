import sys
sys.path.append('src')

import json
import matplotlib.pyplot as plt

def load_json(path):
    print("Loading:", path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    match_file = 'results/json/lm_abacate_base_boa.mp4_norm.json_vs_lm_abacate_teste1.mp4.json.json'
    match_data = load_json(match_file)
    base_file = match_data['base']
    target_file = match_data['target']
    print(f"Base: {base_file}")
    print(f"Target: {target_file}")
    matches = match_data['matches']
    matches.sort(key=lambda x: x['target_frame'])

    base_data = load_json(base_file)
    target_data = load_json(target_file)
    
except Exception as e:
    import traceback
    traceback.print_exc()
