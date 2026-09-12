"""
Script para exportar os parâmetros de rede do modelo_gestos.h5 para o Godot (PENO).
Exporta pesos, bias, scale e offset (provenientes de BatchNormalization)
em um formato binário compacto (weights.bin) e gera um arquivo de teste de sanidade
para verificação automática dentro do Godot.
"""

import os
import sys
import json
import struct
import numpy as np
import tensorflow as tf

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_gestos.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "PENO", "assets", "models"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
WEIGHTS_BIN_PATH = os.path.join(OUTPUT_DIR, "weights.bin")
LABELS_JSON_PATH = os.path.join(OUTPUT_DIR, "labels.json")
SANITY_TEST_PATH = os.path.join(OUTPUT_DIR, "sanity_test.json")

def extract_bn_params(bn_layer):
    gamma, beta, mean, variance = bn_layer.get_weights()
    eps = bn_layer.epsilon
    scale = gamma / np.sqrt(variance + eps)
    offset = beta - mean * scale
    return scale.astype(np.float32), offset.astype(np.float32)

def main():
    print(f"Carregando modelo de: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    layers = model.layers
    # 0: dense (42->512)
    # 1: bn (512)
    # 2: dropout
    # 3: dense_1 (512->256)
    # 4: bn_1 (256)
    # 5: dropout
    # 6: dense_2 (256->128)
    # 7: bn_2 (128)
    # 8: dense_3 (128->2364)

    W0, b0 = layers[0].get_weights()
    s0, o0 = extract_bn_params(layers[1])

    W1, b1 = layers[3].get_weights()
    s1, o1 = extract_bn_params(layers[4])

    W2, b2 = layers[6].get_weights()
    s2, o2 = extract_bn_params(layers[7])

    W3, b3 = layers[8].get_weights()

    W0, b0 = W0.astype(np.float32), b0.astype(np.float32)
    W1, b1 = W1.astype(np.float32), b1.astype(np.float32)
    W2, b2 = W2.astype(np.float32), b2.astype(np.float32)
    W3, b3 = W3.astype(np.float32), b3.astype(np.float32)

    num_classes = W3.shape[1]
    print(f"Camadas: 42 -> 512 -> 256 -> 128 -> {num_classes}")

    # Teste de equivalência numérica com o Keras
    np.random.seed(42)
    test_x = np.random.randn(3, 42).astype(np.float32)
    pred_orig = model.predict(test_x, verbose=0)

    h0 = np.maximum(0, np.dot(test_x, W0) + b0) * s0 + o0
    h1 = np.maximum(0, np.dot(h0, W1) + b1) * s1 + o1
    h2 = np.maximum(0, np.dot(h1, W2) + b2) * s2 + o2
    z3 = np.dot(h2, W3) + b3
    exp_z = np.exp(z3 - np.max(z3, axis=-1, keepdims=True))
    pred_manual = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    diff = np.max(np.abs(pred_orig - pred_manual))
    print(f"Erro máximo de equivalência numérica: {diff:.2e}")
    assert diff < 1e-4, f"Diferença numérica excessiva: {diff}"
    print("✅ Teste de sanidade em Python validado com sucesso!")

    # Escrever weights.bin
    # Cabeçalho: dimensões das camadas: [42, 512, 256, 128, num_classes] (5 x uint32)
    # Em seguida, os arrays em ordem:
    # W0, b0, s0, o0
    # W1, b1, s1, o1
    # W2, b2, s2, o2
    # W3, b3
    dims = [42, 512, 256, 128, num_classes]
    with open(WEIGHTS_BIN_PATH, "wb") as f:
        f.write(struct.pack(f"<{len(dims)}I", *dims))
        for arr in [W0, b0, s0, o0, W1, b1, s1, o1, W2, b2, s2, o2, W3, b3]:
            f.write(arr.tobytes())

    file_size_mb = os.path.getsize(WEIGHTS_BIN_PATH) / (1024 * 1024)
    print(f"✅ Pesos exportados: {WEIGHTS_BIN_PATH} ({file_size_mb:.2f} MB)")

    # Exportar labels
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    with open(LABELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"✅ Labels exportadas: {LABELS_JSON_PATH} ({len(labels)} classes)")

    # Exportar arquivo de teste de sanidade com 1 vetor e resultado esperado
    sample_vec = test_x[0].tolist()
    sample_pred = pred_orig[0].tolist()
    sample_top_idx = int(np.argmax(sample_pred))
    sanity_data = {
        "input_features": sample_vec,
        "expected_top_index": sample_top_idx,
        "expected_top_label": labels[sample_top_idx] if sample_top_idx < len(labels) else "",
        "expected_confidence": float(sample_pred[sample_top_idx])
    }
    with open(SANITY_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(sanity_data, f, indent=2)
    print(f"✅ Arquivo de teste de sanidade exportado: {SANITY_TEST_PATH}")

if __name__ == "__main__":
    main()
