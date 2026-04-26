# 📊 Relatório de Auditoria do Dataset (Libras AI)

**Total de Amostras**: 9361
**Data da Auditoria**: 15/04/2026

## 📈 Distribuição de Classes

| Classe | Total | Cache (Imagens) | Custom ( Trainer) | % do Peso | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | 1381 | 1381 | 0 | 14.8% | ✅ OK |
| **B** | 1317 | 1317 | 0 | 14.1% | ✅ OK |
| **X** | 1265 | 1265 | 0 | 13.5% | ✅ OK |
| **R** | 1051 | 1051 | 0 | 11.2% | ✅ OK |
| **N** | 783 | 783 | 0 | 8.4% | ✅ OK |
| **C** | 633 | 633 | 0 | 6.8% | ✅ OK |
| **D** | 625 | 625 | 0 | 6.7% | ✅ OK |
| **CONCHA** | 480 | 0 | 480 | 5.1% | 🛑 CRÍTICO |
| **U** | 417 | 417 | 0 | 4.5% | 🛑 CRÍTICO |
| **W** | 389 | 389 | 0 | 4.2% | 🛑 CRÍTICO |
| **V** | 298 | 298 | 0 | 3.2% | 🛑 CRÍTICO |
| **L** | 206 | 206 | 0 | 2.2% | 🛑 CRÍTICO |
| **A** | 202 | 202 | 0 | 2.2% | 🛑 CRÍTICO |
| **Y** | 112 | 112 | 0 | 1.2% | 🛑 CRÍTICO |
| **S** | 66 | 66 | 0 | 0.7% | 🛑 CRÍTICO |
| **E** | 63 | 63 | 0 | 0.7% | 🛑 CRÍTICO |
| **O** | 50 | 50 | 0 | 0.5% | 🛑 CRÍTICO |
| **M** | 23 | 23 | 0 | 0.2% | 🛑 CRÍTICO |

## 🔍 Diagnóstico de IA

### Recomendações:
1. **Meta de Estabilidade**: Para cada classe '⚠️ BAIXO', grave pelo menos mais 12 sessões de 60 frames.
2. **Diversidade**: Grave a 'CONCHA' em diferentes ângulos e distâncias para evitar confusão com o sinal 'V'.
3. **Oversampling**: Aplicaremos pesos de classe no próximo treinamento para compensar o desequilíbrio.
