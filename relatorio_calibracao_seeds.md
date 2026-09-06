# Relatório Técnico de Calibração de Seeds e Matriz de Confusão LIBRAS

**Data de Execução:** 2026-09-02 20:17:09  
**Módulo:** Pipeline Multiagente Autônomo de Visão Computacional e Biomecânica

---

## 1. Agente 1: Sanitização de Dados e Prevenção de Erros

- **Total de frames brutos ingeridos:** 1080
- **Frames descartados por Oclusão (Visibility < 0.7):** 54
- **Frames descartados por Outliers (Z-score > 3.0 ou Anomalia):** 66
- **Frames válidos retidos para calibração:** 960 (88.9% aproveitamento)

### 1.1 Amostra de Frames Descartados por Baixa Confiança (Oclusão)

| Classe | Arquivo | Frame ID | Motivo do Descarte |
|---|---|---|---|
| `classe_A` | `sessao_frontal.json` | `5` | Oclusão no landmark crítico 0 (visibility=0.582 < 0.7) |
| `classe_A` | `sessao_frontal.json` | `23` | Oclusão no landmark crítico 20 (visibility=0.600 < 0.7) |
| `classe_A` | `sessao_perfil_lateral.json` | `5` | Oclusão no landmark crítico 16 (visibility=0.265 < 0.7) |
| `classe_A` | `sessao_perfil_lateral.json` | `23` | Oclusão no landmark crítico 4 (visibility=0.608 < 0.7) |
| `classe_A` | `sessao_inclinada.csv` | `5` | Oclusão no landmark crítico 12 (visibility=0.631 < 0.7) |
| `classe_A` | `sessao_inclinada.csv` | `23` | Oclusão no landmark crítico 12 (visibility=0.334 < 0.7) |
| `classe_B` | `sessao_frontal.json` | `5` | Oclusão no landmark crítico 8 (visibility=0.323 < 0.7) |
| `classe_B` | `sessao_frontal.json` | `23` | Oclusão no landmark crítico 20 (visibility=0.542 < 0.7) |
| `classe_B` | `sessao_perfil_lateral.json` | `5` | Oclusão no landmark crítico 4 (visibility=0.447 < 0.7) |
| `classe_B` | `sessao_perfil_lateral.json` | `23` | Oclusão no landmark crítico 17 (visibility=0.406 < 0.7) |
| `classe_B` | `sessao_inclinada.csv` | `5` | Oclusão no landmark crítico 8 (visibility=0.263 < 0.7) |
| `classe_B` | `sessao_inclinada.csv` | `23` | Oclusão no landmark crítico 5 (visibility=0.556 < 0.7) |
| `classe_C` | `sessao_frontal.json` | `5` | Oclusão no landmark crítico 13 (visibility=0.213 < 0.7) |
| `classe_C` | `sessao_frontal.json` | `23` | Oclusão no landmark crítico 9 (visibility=0.408 < 0.7) |
| `classe_C` | `sessao_perfil_lateral.json` | `5` | Oclusão no landmark crítico 17 (visibility=0.316 < 0.7) |
| ... | ... | ... | *mais 39 frames ocluídos descartados* |

### 1.2 Amostra de Frames Descartados por Outlier / Anomalia Biomecânica

| Classe | Arquivo | Frame ID | Anomalia Identificada |
|---|---|---|---|
| `classe_A` | `sessao_frontal.json` | `13` | Anomalia biomecânica: Segmento (19->20) com comprimento impossível (4.43x da palma) |
| `classe_A` | `sessao_frontal.json` | `37` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.32x da palma) |
| `classe_A` | `sessao_frontal.json` | `38` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.28x da palma) |
| `classe_A` | `sessao_perfil_lateral.json` | `13` | Anomalia biomecânica: Segmento (19->20) com comprimento impossível (4.00x da palma) |
| `classe_A` | `sessao_perfil_lateral.json` | `25` | Anomalia biomecânica: Segmento (0->9) com comprimento impossível (1.29x da palma) |
| `classe_A` | `sessao_inclinada.csv` | `13` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.34x da palma) |
| `classe_A` | `sessao_inclinada.csv` | `19` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.35x da palma) |
| `classe_A` | `sessao_inclinada.csv` | `27` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.30x da palma) |
| `classe_A` | `sessao_inclinada.csv` | `31` | Anomalia biomecânica: Segmento (0->9) com comprimento impossível (1.27x da palma) |
| `classe_B` | `sessao_frontal.json` | `4` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.31x da palma) |
| `classe_B` | `sessao_frontal.json` | `9` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.26x da palma) |
| `classe_B` | `sessao_frontal.json` | `13` | Anomalia biomecânica: Segmento (7->8) com comprimento impossível (3.38x da palma) |
| `classe_B` | `sessao_frontal.json` | `33` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.27x da palma) |
| `classe_B` | `sessao_frontal.json` | `37` | Anomalia biomecânica: Segmento (0->5) com comprimento impossível (1.26x da palma) |
| `classe_B` | `sessao_perfil_lateral.json` | `13` | Anomalia biomecânica: Segmento (7->8) com comprimento impossível (3.24x da palma) |
| ... | ... | ... | *mais 51 anomalias descartadas* |

---

## 2. Agente 2 e Agente 3: Sementes Calibradas e Sub-Sementes Multiangulares

Resumo das sementes calculadas após normalização espacial invariante a rotações e escala:

| Classe | Variação Angular | Sub-Sementes Geradas | Amostras | Threshold Médio por Junta |
|---|---|---|---|---|
| `classe_A` | Sim (K-Means k=2) | `SEED_CLASSE_A_FRONTAL`, `SEED_CLASSE_A_PERFIL` | 105 | ±0.080 |
| `classe_B` | Sim (K-Means k=2) | `SEED_CLASSE_B_FRONTAL`, `SEED_CLASSE_B_PERFIL` | 95 | ±0.081 |
| `classe_C` | Sim (K-Means k=2) | `SEED_CLASSE_C_FRONTAL`, `SEED_CLASSE_C_PERFIL` | 111 | ±0.080 |
| `classe_CONCHA` | Sim (K-Means k=2) | `SEED_CLASSE_CONCHA_FRONTAL`, `SEED_CLASSE_CONCHA_PERFIL` | 106 | ±0.080 |
| `classe_I` | Sim (K-Means k=2) | `SEED_CLASSE_I_FRONTAL`, `SEED_CLASSE_I_PERFIL` | 109 | ±0.080 |
| `classe_L` | Sim (K-Means k=2) | `SEED_CLASSE_L_FRONTAL`, `SEED_CLASSE_L_PERFIL` | 101 | ±0.080 |
| `classe_PALMA_ABERTA` | Sim (K-Means k=2) | `SEED_CLASSE_PALMA_ABERTA_FRONTAL`, `SEED_CLASSE_PALMA_ABERTA_PERFIL` | 111 | ±0.082 |
| `classe_V` | Sim (K-Means k=2) | `SEED_CLASSE_V_FRONTAL`, `SEED_CLASSE_V_PERFIL` | 111 | ±0.081 |
| `classe_W` | Sim (K-Means k=2) | `SEED_CLASSE_W_FRONTAL`, `SEED_CLASSE_W_PERFIL` | 111 | ±0.081 |

---

## 3. Agente 4: Matriz de Confusão e Otimização Punitiva

- **Acurácia Inicial (Pesos Iguais 1.0):** **100.00%**
- **Acurácia Pós-Ponderação Punitiva:** **100.00%**
- **Falsos-Positivos Corrigidos:** **0**

### 3.1 Resolução de Pares de Falsos-Positivos

Nenhum falso positivo detectado no dataset de teste.

### 3.2 Matriz de Confusão Final (Otimizada)

| Real \ Previsto | `classe_A` | `classe_B` | `classe_C` | `classe_CONCHA` | `classe_I` | `classe_L` | `classe_PALMA_ABERTA` | `classe_V` | `classe_W` |
|---|---|---|---|---|---|---|---|---|---|
| `classe_A` | 105 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classe_B` | 0 | 95 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classe_C` | 0 | 0 | 111 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classe_CONCHA` | 0 | 0 | 0 | 106 | 0 | 0 | 0 | 0 | 0 |
| `classe_I` | 0 | 0 | 0 | 0 | 109 | 0 | 0 | 0 | 0 |
| `classe_L` | 0 | 0 | 0 | 0 | 0 | 101 | 0 | 0 | 0 |
| `classe_PALMA_ABERTA` | 0 | 0 | 0 | 0 | 0 | 0 | 111 | 0 | 0 |
| `classe_V` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 111 | 0 |
| `classe_W` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 111 |

---
Arquivo gerado automaticamente pelo Ecossistema Multiagente de Calibração LIBRAS.
