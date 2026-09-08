import os
import json
import glob
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Treinamento IA root
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'modelo_gestos.h5')
TFLITE_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'modelo_gestos.tflite')
LABELS_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'labels.txt')
SYNTHETIC_JSON_DIR = os.path.join(BASE_DIR, 'data', 'datasets', 'synthetic_dataset')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'unified_cache')

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, _ = divmod(remainder, 60)
        return f"{h}h{m:02d}m"

# ----------------------------------------------------------
# FASE 1: Conversão incremental JSON -> NPZ (uma vez só)
# ----------------------------------------------------------

def convert_json_to_npz():
    """
    Verifica o cache .npz ou converte cada data.json em .npz compacto.
    Retorna a lista de labels encontradas.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    if len(npz_files) >= 2000:
        logging.info(f"--- [FASE 1] Encontrados {len(npz_files):,} arquivos .npz prontos no cache ---")
        labels_found = set()
        for p in npz_files:
            lbl = os.path.splitext(os.path.basename(p))[0].upper()
            labels_found.add(lbl)
        logging.info(f"Cache pronto: {len(labels_found)} classes prontas para carregamento.")
        return sorted(labels_found)

    json_files = sorted(glob.glob(os.path.join(SYNTHETIC_JSON_DIR, "**", "*.json"), recursive=True))

    if not json_files:
        logging.error("Nenhum arquivo sintético (.npz ou .json) encontrado.")
        return []

    total = len(json_files)
    labels_found = set()
    converted = 0
    skipped = 0
    start = time.time()

    logging.info(f"--- [FASE 1] Conversão JSON → NPZ ({total} arquivos) ---")

    for idx, path in enumerate(json_files):
        label_dir = os.path.basename(os.path.dirname(path))
        npz_path = os.path.join(CACHE_DIR, f"{label_dir}.npz")

        # Incremental: pula se já convertido e o JSON não foi modificado
        if os.path.exists(npz_path) and os.path.getmtime(npz_path) >= os.path.getmtime(path):
            # Ler apenas a label do nome da pasta (sem abrir o arquivo)
            labels_found.add(label_dir.upper())
            skipped += 1
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "metadata" not in data or "frames" not in data:
                    continue

                label = data["metadata"].get("label", label_dir).upper()
                labels_found.add(label)

                # Processar frames desta classe: JSON → numpy flat (N, 42) in-place
                frames = data["frames"]
                n = len(frames)
                X_class = np.empty((n, 42), dtype=np.float32)

                for i, frame in enumerate(frames):
                    lms = frame["landmarks"]
                    # Flatten 21x2 → 42, subtraindo pulso (landmark 0)
                    wrist_x, wrist_y = lms[0][0], lms[0][1]
                    for j in range(21):
                        X_class[i, j*2]     = lms[j][0] - wrist_x
                        X_class[i, j*2 + 1] = lms[j][1] - wrist_y

                # Salvar como NPZ comprimido
                np.savez_compressed(npz_path, X=X_class, label=label)
                converted += 1

                # Liberar memória imediatamente
                del data, frames, X_class

            except Exception as e:
                logging.warning(f"Erro ao converter {path}: {e}")

        # Progresso
        if (idx + 1) % 100 == 0 or idx == total - 1:
            elapsed = time.time() - start
            eta = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0
            print(f"\r  [CACHE] {idx+1}/{total} "
                  f"({converted} novos, {skipped} cache) "
                  f"| Tempo: {format_time(elapsed)} | ETA: {format_time(eta)}",
                  end="", flush=True)

    print()
    logging.info(f"Cache pronto: {converted} convertidos, {skipped} do cache. "
                 f"Labels únicas: {len(labels_found)}")
    return sorted(labels_found)

# ----------------------------------------------------------
# FASE 2: Carregamento classe-por-classe em array pré-alocada
# ----------------------------------------------------------

def load_dataset_from_cache(label_encoder):
    """
    Carrega os NPZ classe-por-classe numa array pré-alocada.
    Sem cópias intermediárias — apenas 1 array X e 1 array y na RAM.
    Aplica espelhamento horizontal via operação vetorizada.
    Retorna X_data (N*2, 42) e y_data (N*2,) já com espelho incluso.
    """
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

    # 1ª passada rápida: contar total de amostras
    logging.info("  Contando amostras nos arquivos de cache...")
    count_start = time.time()
    total_samples = 0
    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        total_samples += len(data['X'])
        del data
        if (i + 1) % 500 == 0 or i == len(npz_files) - 1:
            print(f"\r    {i+1}/{len(npz_files)} arquivos | {total_samples:,} amostras",
                  end="", flush=True)
    print()
    logging.info(f"  Contagem concluída em {format_time(time.time() - count_start)}")

    # Pré-alocar arrays (original + espelhado = 2x)
    total_with_mirror = total_samples * 2
    logging.info(f"  Pré-alocando arrays: {total_with_mirror:,} × 42 floats "
                 f"(~{total_with_mirror * 42 * 4 / (1024**3):.2f} GB)")
    X_data = np.empty((total_with_mirror, 42), dtype=np.float32)
    y_data = np.empty(total_with_mirror, dtype=np.int32)

    # 2ª passada: preencher classe-por-classe (sem cópias intermediárias)
    logging.info("  Carregando dados classe-por-classe...")
    load_start = time.time()
    offset = 0  # posição de escrita na metade original
    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        X_class = data['X']  # (N_class, 42) — já relativizado no cache
        label_str = str(data['label'])
        label_enc = label_encoder.transform([label_str])[0]
        n = len(X_class)

        # Preencher metade original
        X_data[offset:offset + n] = X_class
        y_data[offset:offset + n] = label_enc

        # Preencher metade espelhada (na segunda metade do array)
        mirror_offset = total_samples + offset
        X_data[mirror_offset:mirror_offset + n] = X_class
        X_data[mirror_offset:mirror_offset + n, 0::2] *= -1  # Inverte eixo X

        y_data[mirror_offset:mirror_offset + n] = label_enc

        offset += n
        del data, X_class

        if (i + 1) % 200 == 0 or i == len(npz_files) - 1:
            elapsed = time.time() - load_start
            pct = (i + 1) / len(npz_files) * 100
            eta = (elapsed / (i + 1)) * (len(npz_files) - i - 1) if i > 0 else 0
            print(f"\r    [CARGA] {i+1}/{len(npz_files)} classes ({pct:.0f}%) | "
                  f"{offset:,} amostras | "
                  f"Tempo: {format_time(elapsed)} | ETA: {format_time(eta)}",
                  end="", flush=True)
    print()
    logging.info(f"  Carga concluída em {format_time(time.time() - load_start)}")

    return X_data, y_data, total_samples

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run_neural_engine():
    total_start = time.time()

    logging.info("=" * 65)
    logging.info("  NEURAL ENGINE — Pipeline de Treinamento LIBRAS")
    logging.info("=" * 65)
    logging.info(f"Início: {time.strftime('%H:%M:%S')}")
    logging.info("")

    # ============================================================
    # FASE 1: Cache incremental JSON → NPZ
    # ============================================================
    fase1_start = time.time()
    labels_list = convert_json_to_npz()
    fase1_elapsed = time.time() - fase1_start
    logging.info(f"FASE 1 concluída em {format_time(fase1_elapsed)} (acumulado: {format_time(time.time() - total_start)})")
    logging.info("")

    if not labels_list:
        logging.error("Nenhuma label encontrada. Abortando.")
        return

    # Encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_list)
    num_classes = len(label_encoder.classes_)
    logging.info(f"Classes detectadas: {num_classes}")

    # ============================================================
    # FASE 2: Carregar dataset (classe-por-classe, array pré-alocada)
    # ============================================================
    fase2_start = time.time()
    logging.info("--- [FASE 2] Montando Matriz Neural do Dataset Sintético ---")

    X_data, y_data, total_original = load_dataset_from_cache(label_encoder)

    val_split = 0.15
    train_samples = int(total_original * (1.0 - val_split)) * 2
    val_samples = int(total_original * val_split)

    logging.info(f"  Amostras originais: {total_original:,}")
    logging.info(f"  Total com espelho: {len(X_data):,}")
    logging.info(f"  RAM estimada: ~{X_data.nbytes / (1024**3):.2f} GB (X) + ~{y_data.nbytes / (1024**2):.0f} MB (y)")

    # Split treino/validação
    logging.info("  Separando treino/validação (train_test_split)...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.15, random_state=42
    )
    # Liberar arrays originais (train_test_split já copiou)
    del X_data, y_data
    logging.info(f"  Treino: {len(X_train):,} | Validação: {len(X_test):,}")

    # Montar tf.data com from_tensor_slices (Keras sabe o tamanho → barra de progresso)
    BATCH_SIZE = 2048
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    fase2_elapsed = time.time() - fase2_start
    logging.info(f"FASE 2 concluída em {format_time(fase2_elapsed)} (acumulado: {format_time(time.time() - total_start)})")
    logging.info("")

    # ============================================================
    # FASE 3: Construção do Modelo
    # ============================================================
    fase3_start = time.time()
    logging.info("--- [FASE 3] Construindo Modelo Neural ---")
    logging.info(f"  Entrada: 42 features | Saída: {num_classes} classes")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    fase3_elapsed = time.time() - fase3_start
    logging.info(f"FASE 3 concluída em {format_time(fase3_elapsed)} (acumulado: {format_time(time.time() - total_start)})")
    logging.info("")

    # ============================================================
    # FASE 4: Treinamento
    # ============================================================
    logging.info("--- [FASE 4] Treinamento da Neural Engine ---")
    logging.info(f"  Batch size: {BATCH_SIZE} | Épocas máx: 150 | Early stopping: patience=15")
    logging.info(f"  Início do treino: {time.strftime('%H:%M:%S')}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    start_train = time.time()
    history = model.fit(
        train_dataset,
        epochs=150,
        validation_data=val_dataset,
        callbacks=[early_stopping, checkpoint_callback]
    )
    elapsed_train = time.time() - start_train
    logging.info(f"FASE 4 concluída em {format_time(elapsed_train)} (acumulado: {format_time(time.time() - total_start)})")
    logging.info("")

    # Relatório
    training_report_path = os.path.join(BASE_DIR, "reports", "training_report.json")
    os.makedirs(os.path.dirname(training_report_path), exist_ok=True)
    training_summary = {
        "total_original_samples": total_original,
        "total_augmented_samples": train_samples + val_samples,
        "num_classes": num_classes,
        "final_train_accuracy": float(history.history.get('accuracy', [0])[-1]),
        "final_train_loss": float(history.history.get('loss', [0])[-1]),
        "final_val_accuracy": float(history.history.get('val_accuracy', [0])[-1]),
        "final_val_loss": float(history.history.get('val_loss', [0])[-1]),
        "epochs_trained": len(history.history.get('accuracy', [])),
        "training_time": format_time(elapsed_train)
    }
    with open(training_report_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    logging.info(f"Relatório salvo em: {training_report_path}")

    # ============================================================
    # FASE 5: Compilando TFLite e Labels
    # ============================================================
    fase5_start = time.time()
    logging.info("--- [FASE 5] Compilando TFLite e Labels ---")
    logging.info("  Salvando modelo H5...")
    model.save(MODEL_SAVE_PATH)
    logging.info(f"  H5 salvo: {MODEL_SAVE_PATH}")

    logging.info("  Convertendo para TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    tflite_size_mb = os.path.getsize(TFLITE_SAVE_PATH) / (1024 * 1024)
    logging.info(f"  TFLite salvo: {TFLITE_SAVE_PATH} ({tflite_size_mb:.1f} MB)")

    logging.info("  Salvando labels...")
    with open(LABELS_SAVE_PATH, 'w') as f:
        for lbl in label_encoder.classes_:
            f.write(f"{lbl}\n")
    logging.info(f"  Labels salvas: {LABELS_SAVE_PATH} ({num_classes} classes)")

    fase5_elapsed = time.time() - fase5_start
    logging.info(f"FASE 5 concluída em {format_time(fase5_elapsed)} (acumulado: {format_time(time.time() - total_start)})")
    logging.info("")

    # ============================================================
    # FASE 6: Atualizando POC
    # ============================================================
    fase6_start = time.time()
    logging.info("--- [FASE 6] Atualizando Front-end da POC ---")
    import update_poc
    update_poc.update_poc_files()
    fase6_elapsed = time.time() - fase6_start
    logging.info(f"POC Atualizada. FASE 6 concluída em {format_time(fase6_elapsed)}")
    logging.info("")

    # ============================================================
    # RESUMO FINAL
    # ============================================================
    total_elapsed = time.time() - total_start
    logging.info("=" * 65)
    logging.info("  PIPELINE CONCLUÍDO")
    logging.info("=" * 65)
    logging.info(f"  Classes: {num_classes}")
    logging.info(f"  Amostras originais: {total_original:,}")
    logging.info(f"  Amostras de treino (c/ espelho): ~{train_samples:,}")
    logging.info(f"  Épocas treinadas: {len(history.history.get('accuracy', []))}")
    logging.info(f"  Acurácia final (treino): {history.history.get('accuracy', [0])[-1]:.4f}")
    logging.info(f"  Acurácia final (validação): {history.history.get('val_accuracy', [0])[-1]:.4f}")
    logging.info(f"  Tempo total: {format_time(total_elapsed)}")
    logging.info(f"  Fim: {time.strftime('%H:%M:%S')}")
    logging.info("=" * 65)

if __name__ == "__main__":
    run_neural_engine()
