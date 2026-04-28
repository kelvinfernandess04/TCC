import os
import json
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
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'extraction_cache.json')
CUSTOM_JSON_DIR = os.path.join(BASE_DIR, 'data', 'datasets', 'dataset_custom')

ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'I', 'L', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y']

def run_neural_engine():
    logging.info("--- [FASE 2] Montando Matriz Neural do Cache ---")
    X = []
    y = []
    
    # 2.1 Puxar do cache recém varrido 
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            for k, meta in cache.items():
                if meta.get("status") == "success":
                    label = meta["label"].upper()
                    if label in ALLOWED_LABELS:
                        X.append(meta["pts"])
                        y.append(label)
    
    # 2.2 Puxar do Custom Json Directory (Recursivo / Catálogo)
    if os.path.exists(CUSTOM_JSON_DIR):
        import glob
        json_files = glob.glob(os.path.join(CUSTOM_JSON_DIR, "**", "*.json"), recursive=True)
        logging.info(f"Localizados {len(json_files)} arquivos de catálogo customizado.")
        
        for path in json_files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Formato Novo: { metadata: {label: ...}, frames: [ {landmarks: ...}, ... ] }
                    if "metadata" in data and "frames" in data:
                        # Lógica Proativa: Catalogamos TUDO o que estiver no Custom (CONCHA, etc)
                        label = data["metadata"]["label"].upper()
                        for frame in data["frames"]:
                            lms = frame["landmarks"]
                            if len(lms) == 21:
                                X.append(lms)
                                y.append(label)
                    # Formato Antigo (Legado): { item_id: { labels: [...], landmarks: [...] } }
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
                logging.warning(f"Erro ao ler custom json: {path} -> {e}")

    logging.info(f"Total de amostras brutas garimpadas para treinamento: {len(X)}")
    
    if len(X) == 0:
        logging.error("Nenhuma amostra validada para treinamento. Abortando pipeline.")
        return

    # FASE 3: Aumento Computacional e Normalização 
    logging.info("--- [FASE 3] Data Augmentation e Normalização Bounding Box ---")
    augmented_X = []
    augmented_y = []
    
    AUGMENTATION_MULTIPLIER = 5
    ROTATION_ANGLES = [0, -15, 15, -30, 30]
    
    for idx, landmarks in enumerate(X):
        label = y[idx]
        for m in range(AUGMENTATION_MULTIPLIER):
            for angle in ROTATION_ANGLES:
                noise = np.random.normal(0, 0.005, size=(21, 2)) if m > 0 else np.zeros((21,2))
                scale = np.random.uniform(0.9, 1.1) if m > 0 else 1.0
                
                aug_landmarks = (np.array(landmarks) * scale) + noise
                
                if angle != 0:
                    angle_rad = np.radians(angle)
                    c, s = np.cos(angle_rad), np.sin(angle_rad)
                    R = np.array(((c, -s), (s, c)))
                    centroid = np.mean(aug_landmarks, axis=0)
                    aug_landmarks = np.dot(aug_landmarks - centroid, R) + centroid
                
                xs = aug_landmarks[:, 0]
                ys = aug_landmarks[:, 1]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                width = max(max_x - min_x, 1e-6)
                height = max(max_y - min_y, 1e-6)
                size = max(width, height)
                
                normalized = []
                for lx, ly in aug_landmarks:
                    nx = (lx - min_x) / size
                    ny = (ly - min_y) / size
                    normalized.append([nx, ny])
                    
                flat_original = np.array(normalized).flatten()
                augmented_X.append(flat_original)
                augmented_y.append(label)
                
                # Espelhamento Horizontal (Ambidestria): inverter eixo X
                mirrored = []
                for nx, ny in normalized:
                    mirrored.append(1.0 - nx)
                    mirrored.append(ny)
                augmented_X.append(np.array(mirrored, dtype=np.float32))
                augmented_y.append(label)
            
    logging.info(f"Total de amostras após Augmentation (x{AUGMENTATION_MULTIPLIER} ruído * x{len(ROTATION_ANGLES)} ângulos + espelho): {len(augmented_X)}")

    # Codificação
    X_data = np.array(augmented_X, dtype=np.float32)
    y_data = np.array(augmented_y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)
    num_classes = len(label_encoder.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.15, random_state=42)

    # Construção Profunda 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
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
    start_train = time.time()
    # Otimização do Pipeline com tf.data (Batch 128 + Prefetch)
    BATCH_SIZE = 128
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Fit retorna History
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=[early_stopping])
    elapsed_train = time.time() - start_train
    mins, secs = divmod(int(elapsed_train), 60)
    # Relatório resumido
    training_report_path = os.path.join(BASE_DIR, "reports", "training_report.json")
    training_summary = {
        "total_original_samples": len(X),
        "total_augmented_samples": len(augmented_X),
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

if __name__ == "__main__":
    run_neural_engine()
