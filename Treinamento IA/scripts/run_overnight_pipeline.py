#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Noturno Automatizado de Treinamento LIBRAS (Resiliente e Monitorado)
=============================================================================
Etapas executadas sequencialmente:
1. Geração Massiva de Dados Sintéticos baseados nas 2.568 sementes anatômicas (seeds.json).
2. Validação da integridade do cache de amostras geradas.
3. Treinamento da Neural Engine com Early Stopping e Checkpoint.
4. Exportação para TensorFlow Lite (.tflite) e sincronização automática com a POC.
5. Registro contínuo de logs cronometrados em reports/overnight_training.log.
"""

import os
import sys
import time
import traceback
import json
import glob
from datetime import datetime

# Garante saída UTF-8 no terminal Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CACHE_DIR = os.path.join(BASE_DIR, "data", "unified_cache")
SEEDS_FILE = os.path.join(BASE_DIR, "data", "seeds", "seeds.json")
LOG_FILE = os.path.join(REPORTS_DIR, "overnight_training.log")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


class Logger:
    """Escreve simultaneamente no console e no arquivo de log persistente."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "a", encoding="utf-8")

    def log(self, message: str, level: str = "INFO"):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{now_str}] [{level}] {message}"
        print(formatted, flush=True)
        self.file.write(formatted + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def run_pipeline():
    logger = Logger(LOG_FILE)
    pipeline_start = time.time()

    logger.log("=" * 70)
    logger.log("  INÍCIO DO PIPELINE NOTURNO AUTOMATIZADO DE TREINAMENTO LIBRAS")
    logger.log("=" * 70)
    logger.log(f"Diretório Base: {BASE_DIR}")
    logger.log(f"Arquivo de Sementes: {SEEDS_FILE}")
    logger.log(f"Arquivo de Log Persistente: {LOG_FILE}")

    # =========================================================================
    # ETAPA 1: GERAÇÃO MASSIVA DE DADOS SINTÉTICOS
    # =========================================================================
    logger.log("-" * 70)
    logger.log("ETAPA 1/2: GERAÇÃO MASSIVA DE DADOS SINTÉTICOS (SEEDS.JSON)")
    logger.log("-" * 70)

    if not os.path.exists(SEEDS_FILE):
        logger.log(f"ERRO CRÍTICO: Arquivo de sementes não encontrado ({SEEDS_FILE}).", "ERROR")
        return False

    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        seeds_data = json.load(f)
    expected_labels = sorted([k for k in seeds_data.keys() if not k.startswith("__")])
    total_expected = len(expected_labels)
    logger.log(f"Total de classes cinemáticas esperadas: {total_expected:,}")

    max_generation_retries = 3
    generation_success = False

    for attempt in range(1, max_generation_retries + 1):
        logger.log(f"Executando gerador sintético (Tentativa {attempt}/{max_generation_retries})...")
        t0 = time.time()
        try:
            import synthetic_generator
            synthetic_generator.main()
            elapsed_gen = time.time() - t0
            logger.log(f"Geração concluída em {elapsed_gen:.1f}s. Validando arquivos gerados...")

            # Validação da integridade dos arquivos .npz no cache
            valid_npz = 0
            empty_npz = []
            for lbl in expected_labels:
                npz_p = os.path.join(CACHE_DIR, f"{lbl}.npz")
                if os.path.exists(npz_p) and os.path.getsize(npz_p) > 1024:
                    valid_npz += 1
                else:
                    empty_npz.append(lbl)

            logger.log(f"Classes validadas com sucesso no cache: {valid_npz:,} / {total_expected:,}")

            if valid_npz == total_expected:
                logger.log("✓ ETAPA 1 VALIDADA: 100% das classes geradas com integridade!", "SUCCESS")
                generation_success = True
                break
            else:
                logger.log(f"Aviso: {len(empty_npz)} classes com arquivo ausente ou corrompido. Tentando novamente...", "WARNING")

        except Exception as e:
            logger.log(f"Erro durante a geração sintética: {e}\n{traceback.format_exc()}", "ERROR")
            time.sleep(2)

    if not generation_success:
        logger.log("ERRO FATAL: Não foi possível gerar a totalidade do dataset sintético.", "FATAL")
        return False

    # =========================================================================
    # ETAPA 2: TREINAMENTO DA REDE NEURAL E EXPORTAÇÃO TFLITE
    # =========================================================================
    logger.log("-" * 70)
    logger.log("ETAPA 2/2: TREINAMENTO DA NEURAL ENGINE (DEEP LEARNING)")
    logger.log("-" * 70)

    max_training_retries = 3
    training_success = False

    for attempt in range(1, max_training_retries + 1):
        logger.log(f"Iniciando treino da Neural Engine (Tentativa {attempt}/{max_training_retries})...")
        t0 = time.time()
        try:
            import neural_engine
            neural_engine.run_neural_engine()
            elapsed_train = time.time() - t0
            logger.log(f"Treinamento concluído em {elapsed_train / 60:.1f} minutos.")

            # Validação dos artefatos de saída do modelo
            h5_path = os.path.join(BASE_DIR, "models", "modelo_gestos.h5")
            tflite_path = os.path.join(BASE_DIR, "models", "modelo_gestos.tflite")
            labels_path = os.path.join(BASE_DIR, "models", "labels.txt")
            poc_model_js = os.path.join(PROJECT_ROOT, "POC", "modelBase64.js")
            poc_labels_js = os.path.join(PROJECT_ROOT, "POC", "labels.js")

            checks = [
                ("Modelo H5", h5_path, 100000),
                ("Modelo TFLite", tflite_path, 50000),
                ("Dicionário Labels", labels_path, 1000),
                ("POC Model Base64 JS", poc_model_js, 50000),
                ("POC Labels JS", poc_labels_js, 1000)
            ]

            all_ok = True
            for name, path, min_size in checks:
                if os.path.exists(path) and os.path.getsize(path) >= min_size:
                    logger.log(f"  ✓ {name}: {os.path.getsize(path) / 1024:.1f} KB")
                else:
                    logger.log(f"  ✗ {name}: Ausente ou menor que o esperado ({path})", "WARNING")
                    all_ok = False

            if all_ok:
                logger.log("✓ ETAPA 2 VALIDADA: Modelos e exportações concluídos com sucesso!", "SUCCESS")
                training_success = True
                break
            else:
                logger.log("Aviso: Alguns arquivos do modelo não passaram na verificação. Tentando novamente...", "WARNING")

        except Exception as e:
            logger.log(f"Erro durante o treinamento neural: {e}\n{traceback.format_exc()}", "ERROR")
            time.sleep(5)

    if not training_success:
        logger.log("ERRO FATAL: Falha no treinamento da rede neural após tentativas.", "FATAL")
        return False

    # =========================================================================
    # RELATÓRIO FINAL
    # =========================================================================
    total_elapsed = time.time() - pipeline_start
    logger.log("=" * 70)
    logger.log("  PIPELINE NOTURNO FINALIZADO COM ÊXITO TOTAL!")
    logger.log("=" * 70)
    logger.log(f"Tempo Total de Execução: {total_elapsed / 60:.1f} minutos ({total_elapsed:.0f}s)")
    logger.log(f"Horário de Conclusão: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("O modelo compilado e a POC estão prontos para o teste de reconhecimento!")
    logger.log("=" * 70)

    logger.close()
    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
