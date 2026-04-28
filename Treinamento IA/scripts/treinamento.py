import os
import sys
import logging
import subprocess

# Configuração Básica de Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Busca o interpretador virtual (Python 3.11) se existir
    venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    python_cmd = venv_python if os.path.exists(venv_python) else "python"

    print("="*60)
    print(" PIPELINE DE TREINAMENTO TOTAL - LIBRAS TCC ")
    print("="*60)
    
    # FASE 1: Extração e Crawler (Landmarks via MediaPipe)
    logging.info("--- [FASE 1] Invocando Data Crawler/Extractor Recursivo ---")
    ret_code_1 = subprocess.call([python_cmd, os.path.join(script_dir, "dataset_extractor.py")])
    if ret_code_1 != 0:
        logging.error("O Extrator encontrou problemas ou foi cancelado. Abortando treinamento.")
        return
    
    # FASE 2 a 5: Treinamento TensorFlow Neural Engine
    logging.info("--- [FASE 2] Ligando Motor de Machine Learning (TensorFlow) ---")
    ret_code_2 = subprocess.call([python_cmd, os.path.join(script_dir, "neural_engine.py")])
    if ret_code_2 != 0:
        logging.error("A Engine Neural encontrou problemas ou foi cancelada. Abortando deploy.")
        return

    # FASE 6: Delegação Automática Pro Front-End
    logging.info("--- [FASE 6] Enviando Base64 e Labels compilados pra POC Web ... ---")
    subprocess.call([python_cmd, os.path.join(script_dir, "update_poc.py")])
    
    print("\n" + "="*50)
    print(" SUCESSO ABSOLUTO! AMBIENTE TODO ATUALIZADO ")
    print("="*50)

if __name__ == "__main__":
    main()