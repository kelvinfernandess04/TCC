import os
import json
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Treinamento IA root
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'modelo_gestos.h5')
TFLITE_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'modelo_gestos.tflite')
LABELS_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'labels.txt')
SYNTHETIC_JSON_DIR = os.path.join(BASE_DIR, 'data', 'datasets', 'synthetic_dataset')

def run_neural_engine():
    logging.info("--- [FASE 2] Montando Matriz Neural do Dataset Sintético ---")
    X = []
    y = []
    
    # Puxar exclusivamente do Synthetic Directory
    json_files = []
    if os.path.exists(SYNTHETIC_JSON_DIR):
        json_files.extend(glob.glob(os.path.join(SYNTHETIC_JSON_DIR, "**", "*.json"), recursive=True))
            
    if json_files:
        logging.info(f"Localizados {len(json_files)} arquivos sintéticos.")
        
        for path in json_files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "metadata" in data and "frames" in data:
                        global_label = data["metadata"].get("label", "").upper()
                        for frame in data["frames"]:
                            frame_label = frame.get("label", global_label).upper()
                            lms = frame["landmarks"]
                            if len(lms) == 21 and frame_label:
                                X.append(lms)
                                y.append(frame_label)
                    else:
                        for item_id, item_data in data.items():
                            labels = item_data.get('labels', [])
                            lms_list = item_data.get('landmarks', [])
                            if labels and lms_list:
                                label = labels[0].upper()
                                lms = lms_list[0]
                                if len(lms) == 21:
                                    X.append(lms)
                                    y.append(label)
            except Exception as e:
                logging.warning(f"Erro ao ler json sintético: {path} -> {e}")

    logging.info(f"Total de amostras brutas garimpadas para treinamento (Sintéticas): {len(X)}")
    
    if len(X) == 0:
        logging.error("Nenhuma amostra validada para treinamento. Abortando pipeline.")
        return

    # FASE 3: Ambidestria (Espelhamento Horizontal)
    logging.info("--- [FASE 3] Aplicando Ambidestria (Flip X) ---")
    # Vetorização NumPy para altíssimo desempenho em datasets gigantes
    X_np = np.array(X, dtype=np.float32) # shape: (N, 21, 2)
    y_np = np.array(y)
    
    # Extrai o pulso (Landmark 0) e centraliza todos os pontos (Broadcast)
    wrists = X_np[:, 0:1, :] # shape: (N, 1, 2)
    relative_lms = X_np - wrists # shape: (N, 21, 2)
    
    # Flatten para (N, 42)
    flat_original = relative_lms.reshape(X_np.shape[0], 42)
    
    # Espelhamento Horizontal (Inversão do eixo X em relação ao pulso)
    mirrored_lms = relative_lms.copy()
    mirrored_lms[:, :, 0] = -mirrored_lms[:, :, 0] # Inverte os valores de X
    flat_mirrored = mirrored_lms.reshape(X_np.shape[0], 42)
    
    # Junta matrizes base e espelhadas
    X_data = np.vstack((flat_original, flat_mirrored))
    y_data = np.concatenate((y_np, y_np))
    
    logging.info(f"Total de amostras prontas para treino (Base + Espelho Esquerdo): {X_data.shape[0]}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)
    num_classes = len(label_encoder.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.15, random_state=42)

    # Construção Profunda 
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

    # ---------------------------
    # Treinamento da Neural Engine
    # ---------------------------
    import time
    # early stopping callback to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Checkpoint para salvar o melhor modelo ao longo das épocas
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    start_train = time.time()
    # Otimização do Pipeline com tf.data (Mega Batch + Prefetch)
    BATCH_SIZE = 2048
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train_test_split já embaralhou os dados perfeitamente. O shuffle_buffer massivo estava engasgando o CPU.
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Fit retorna History
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=[early_stopping, checkpoint_callback])
    elapsed_train = time.time() - start_train
    mins, secs = divmod(int(elapsed_train), 60)
    # Relatório resumido
    training_report_path = os.path.join(BASE_DIR, "reports", "training_report.json")
    os.makedirs(os.path.dirname(training_report_path), exist_ok=True)
    training_summary = {
        "total_original_samples": len(X),
        "total_augmented_samples": len(X_data),
        "num_classes": num_classes,
        "final_train_accuracy": float(history.history.get('accuracy', [0])[-1]),
        "final_train_loss": float(history.history.get('loss', [0])[-1]),
        "final_val_accuracy": float(history.history.get('val_accuracy', [0])[-1]),
        "final_val_loss": float(history.history.get('val_loss', [0])[-1]),
        "epochs_trained": len(history.history.get('accuracy', [])),
        "training_time": f"{mins}m{secs}s"
    }
    with open(training_report_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    print("\n[TRAINING] Relatório salvo em:", training_report_path)
    # ---------------------------
    # Compilando TFLite e Labels
    # ---------------------------
    logging.info("--- [FASE 5] Compilando TFLite e Labels ---")
    model.save(MODEL_SAVE_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    with open(LABELS_SAVE_PATH, 'w') as f:
        for lbl in label_encoder.classes_:
            f.write(f"{lbl}\n")

    logging.info(f"Modelo salvo. Classes listadas: {list(label_encoder.classes_)}")
    
    # ---------------------------
    # Atualizando a POC automaticamente
    # ---------------------------
    logging.info("--- [FASE 6] Atualizando Front-end da POC ---")
    import update_poc
    update_poc.update_poc_files()
    logging.info("POC Atualizada com o novo motor sintético.")

if __name__ == "__main__":
    run_neural_engine()
