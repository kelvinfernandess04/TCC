import os, sys, glob, time, json

# Adiciona o diretório TCC vKV ao path para importar o COMPARADOR3000
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TCC vKV'))
try:
    from COMPARADOR3000 import analyze_similarity
except ImportError as e:
    print(f"[ERROR] Não foi possível importar COMPARADOR3000: {e}")
    sys.exit(1)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_class_from_filename(fn):
    fn = fn.lower()
    if 'abacate' in fn: return 'abacate'
    if 'alho' in fn: return 'alho'
    if 'abacaxi' in fn: return 'abacaxi'
    if 'abobrinha' in fn: return 'abobrinha'
    if 'amigo' in fn: return 'amigo'
    if 'aprender' in fn: return 'aprender'
    if 'trabalhar' in fn: return 'trabalhar'
    parts = fn.split('_')
    for p in parts:
        if p not in ['holistic', 'landmarker', 'lm', 'base', 'teste1', 'teste2', 'ruim', 'boa', 'upscaler-1080p', 'norm', 'json', 'mp4']:
            return p
    return 'unknown'

def main():
    print("Iniciando Benchmark do Método: TCC vKV (COMPARADOR3000)")
    
    all_files = glob.glob("data/*.json")
    bases = [f for f in all_files if f.endswith("_norm.json") and "_base" in os.path.basename(f)]
    targets = all_files
    
    if not bases or not targets:
        print("Bases ou Targets não encontrados.")
        return
        
    print(f"Bases: {len(bases)} | Targets: {len(targets)}")
    
    PASS_THRESHOLD = 75.0
    tp, fn_count, tn, fp = 0, 0, 0, 0
    total_comps = 0
    
    report_lines = []
    
    start_time = time.time()
    
    for base_path in bases:
        base_name = os.path.basename(base_path)
        base_data = load_json(base_path)
        base_class = extract_class_from_filename(base_name)
        
        report_lines.append("-" * 90)
        report_lines.append(f"BASE: {base_name}")
        report_lines.append("-" * 90)
        report_lines.append(f"{'STATUS':<10} | {'SCORE':<8} | TARGET")
        report_lines.append("-" * 90)
        
        for tgt_path in targets:
            tgt_name = os.path.basename(tgt_path)
            
            # Filtra norm_vs_norm e cru_vs_cru igual ao comparador V4
            is_base_norm = "_norm.json" in base_name
            is_tgt_norm = "_norm.json" in tgt_name
            if is_base_norm != is_tgt_norm: continue
            if tgt_name == base_name: continue # Ignore self
            
            tgt_data = load_json(tgt_path)
            tgt_class = extract_class_from_filename(tgt_name)
            
            # Executa o comparador do Colega
            try:
                results = analyze_similarity(base_data, tgt_data)
                score = results.get("_global_similarity_pct", 0.0)
            except Exception as e:
                print(f"[!] Erro ao comparar {base_name} e {tgt_name}: {e}")
                score = 0.0
                
            is_match = base_class == tgt_class
            passed = score >= PASS_THRESHOLD
            
            status_str = "PASS" if passed else "FAIL"
            report_lines.append(f"{status_str:<10} | {score:<8.1f} | {tgt_name}")
            
            if is_match and passed: tp += 1
            elif is_match and not passed: fn_count += 1
            elif not is_match and not passed: tn += 1
            elif not is_match and passed: fp += 1
            
            total_comps += 1

    exec_time = time.time() - start_time
    
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn_count) * 100) if (tp + fn_count) > 0 else 0
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0
    
    report_lines.append("\n" + "="*90)
    report_lines.append(" RESUMO ESTATÍSTICO DE CONFIABILIDADE (COMPARADOR3000) ".center(90))
    report_lines.append("="*90)
    report_lines.append(f"Tempo de Execução:        {exec_time:.2f}s")
    report_lines.append(f"Total de Comparações:     {total_comps}")
    report_lines.append("-" * 90)
    report_lines.append(f"Verdadeiros Posit. (Sinal Certo, Aprovado) : {tp}")
    report_lines.append(f"Falsos Negativos   (Sinal Certo, Reprovado): {fn_count}")
    report_lines.append(f"Verdadeiros Negat. (Sinal Errado, Bloqueado) : {tn}")
    report_lines.append(f"Falsos Positivos   (Sinal Errado, Vazou)   : {fp}")
    report_lines.append("-" * 90)
    report_lines.append(f"Precisão (100% = Zera Falsos Positivos): {precision:.2f}%")
    report_lines.append(f"Recall   (100% = Aceita toda variação) : {recall:.2f}%")
    report_lines.append(f"Especificidade                           : {specificity:.2f}%")
    report_lines.append("="*90)
    
    report_path = os.path.join("results", "benchmark_vkv_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nBenchmark finalizado. Relatório salvo em: {report_path}")
    print(f"Precisão: {precision:.2f}% | Recall: {recall:.2f}% | Especificidade: {specificity:.2f}%")

if __name__ == "__main__":
    main()
