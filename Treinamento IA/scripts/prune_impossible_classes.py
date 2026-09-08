"""
Script de Poda Biomecânica de Classes Impossíveis (TCC - LIBRAS)
===============================================================
Aplica a restrição anatômica decorrente das juncturae tendinum e acoplamento do
Flexor Digitorum Profundus:
Quando o dedo Anelar está estendido (D3 == 0), o dedo Mínimo não consegue
atingir a flexão total contra a palma (D4 in [3, 4]).

Remove as classes impossíveis de:
1. Treinamento IA/data/seeds/seeds.json
2. Treinamento IA/data/cache/npz_classes/*.npz
"""

import os
import sys
import json
import glob

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS_PATH = os.path.join(BASE_DIR, "data", "seeds", "seeds.json")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache", "npz_classes")

def is_anatomically_impossible(code: str) -> bool:
    """
    Retorna True se a configuração viola a cinesiologia humana:
    Código de 10 dígitos: D4 A3 D3 A2 D2 A1 D1 A0 F P
    D4 = code[0] (Mindinho)
    D3 = code[2] (Anelar)
    """
    if len(code) != 10 or not code.isdigit():
        return False
    d4 = int(code[0]) # Mindinho
    d3 = int(code[2]) # Anelar

    # Violação: Anelar esticado (0) com Mindinho fechado (3 ou 4)
    if d3 == 0 and d4 in (3, 4):
        return True
    return False

def main():
    print("=" * 60)
    print("   PODA BIOMECÂNICA DE CLASSES ANATOMICAMENTE IMPOSSÍVEIS")
    print("=" * 60)

    if not os.path.exists(SEEDS_PATH):
        print(f"[!] Erro: {SEEDS_PATH} não encontrado.")
        return

    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    total_original = len(seeds)
    pruned_seeds = {}
    impossible_classes = []

    for code, lms in seeds.items():
        if is_anatomically_impossible(code):
            impossible_classes.append(code)
        else:
            pruned_seeds[code] = lms

    total_pruned = len(impossible_classes)
    total_remaining = len(pruned_seeds)

    print(f"[*] Total original de classes: {total_original:,}")
    print(f"[*] Classes identificadas como impossíveis: {total_pruned:,}")
    print(f"[*] Classes válidas restantes: {total_remaining:,}")

    # Salva seeds.json podado
    with open(SEEDS_PATH, "w", encoding="utf-8") as f:
        json.dump(pruned_seeds, f)
    print(f"[OK] seeds.json atualizado com {total_remaining:,} classes.")

    # Remove arquivos .npz correspondentes do cache
    if os.path.exists(CACHE_DIR):
        removed_cache = 0
        for code in impossible_classes:
            npz_file = os.path.join(CACHE_DIR, f"{code}.npz")
            if os.path.exists(npz_file):
                try:
                    os.remove(npz_file)
                    removed_cache += 1
                except Exception as e:
                    print(f"[!] Erro ao remover {npz_file}: {e}")
        print(f"[OK] Removidos {removed_cache:,} arquivos .npz obsoletos do cache.")

    print("\n" + "=" * 60)
    print("  PODA CONCLUÍDA COM SUCESSO!")
    print("=" * 60)

if __name__ == "__main__":
    main()
