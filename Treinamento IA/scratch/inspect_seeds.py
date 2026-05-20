import json
import os

BASE_DIR = r"c:\DevTools\Faculdade\TCC\Treinamento IA"
SEEDS_FILE = os.path.join(BASE_DIR, 'data', 'seeds', 'seeds.json')

if os.path.exists(SEEDS_FILE):
    with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
        seeds = json.load(f)
    keys = sorted([k for k in seeds.keys() if not k.startswith("__")])
    print(f"Total keys: {len(keys)}")
    print("First 20 keys:")
    for k in keys[:20]:
        print(k)
else:
    print("Seeds file not found!")
