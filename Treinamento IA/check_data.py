import os
import json
import numpy as np

DATA_PATH = r'C:\DevTools\Repositories\Faculdade\TCC\Treinamento IA\HaGRID\ann_subsample'

lengths = set()

for file_name in os.listdir(DATA_PATH):
    if file_name.endswith('.json'):
        label = file_name.replace('.json', '')
        with open(os.path.join(DATA_PATH, file_name), 'r') as f:
            data = json.load(f)
            
            for item_id, item_data in data.items():
                if 'landmarks' in item_data and item_data['landmarks'] is not None:
                    # we should find the correct label index
                    labels = item_data.get('labels', [])
                    for i, lbl in enumerate(labels):
                        if lbl == label:
                            if i < len(item_data['landmarks']):
                                lm = item_data['landmarks'][i]
                            else:
                                lm = item_data['landmarks'][0] # fallback
                            if lm is not None:
                                lengths.add(len(lm))
                    
print("Lengths found:", lengths)
