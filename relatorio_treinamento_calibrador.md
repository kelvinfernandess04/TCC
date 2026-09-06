# Relatório Técnico: Calibração Iterativa em Épocas (Conceito Treino de IA)

**Data de Execução:** 2026-09-02 22:24:14  
**Duração do Treinamento:** 14.47 segundos  
**Método:** Otimização Numérica Biomecânica (Fase 1) + Metric Learning Supervisionado (Fase 2)

---

## 1. Resumo Executivo das Métricas

| Métrica de Desempenho | Antes da Calibração | Pós-Calibração Iterativa | Melhoria |
|---|---|---|---|
| **Erro Biomecânico Médio (RMSE 3D)** | `0.2925` | **`0.2295`** | **+21.55%** |
| **Acurácia de Classificação LIBRAS** | `100.00%` | **`100.00%`** | **+0.00%** |
| **Épocas de Convergência** | - | **5 épocas** | Estabilizado |

---

## 2. Fase 1: Otimização Cinemática Biomecânica (Mão Física)

A Fase 1 ajustou iterativamente os ângulos dos 4 estágios de flexão e as falanges da mão 3D para convergir exatamente na anatomia gravada do usuário.

### 2.1 Erro Articular Médio (RMSE 3D) por Classe

| Classe / Sinal | RMSE Inicial | RMSE Final (Calibrado) | Redução do Erro |
|---|---|---|---|
| `classe_A` | `0.3518` | **`0.2571`** | -0.0947 (26.9%) |
| `classe_B` | `0.2318` | **`0.1788`** | -0.0530 (22.9%) |
| `classe_C` | `0.2390` | **`0.1773`** | -0.0617 (25.8%) |
| `classe_CONCHA` | `0.2912` | **`0.2233`** | -0.0679 (23.3%) |
| `classe_I` | `0.2431` | **`0.1813`** | -0.0618 (25.4%) |
| `classe_L` | `0.2995` | **`0.2415`** | -0.0580 (19.4%) |
| `classe_PALMA_ABERTA` | `0.3009` | **`0.2558`** | -0.0451 (15.0%) |
| `classe_V` | `0.3279` | **`0.2618`** | -0.0661 (20.2%) |
| `classe_W` | `0.3223` | **`0.2637`** | -0.0587 (18.2%) |

### 2.2 Ângulos Biomecânicos de Flexão Otimizados (Estágios)

| Estágio Anatômico | Junta MCP (J2_Pitch) | Junta PIP (J3_Pitch) | Junta DIP (J4_Pitch) |
|---|---|---|---|
| **Estágio 0 (Estendido)** | `0.0°` | `0.0°` | `0.0°` |
| **Estágio 1 (Curvado)** | `5.0°` | `39.2°` | `34.6°` |
| **Estágio 2 (Gancho/Plataforma)** | `33.0°` | `101.4°` | `74.0°` |
| **Estágio 3 (Punho Fechado)** | `70.0°` | `115.0°` | `86.4°` |

---

## 3. Fase 2: Histórico de Treinamento do Classificador (Épocas)

Evolução da função de perda (*Margin Loss*) e taxa de acurácia ao longo das épocas de ajuste dos pesos e sementes:

| Época | Loss de Margem | Acurácia (%) | Conflitos Residuais | Barra de Acurácia |
|---|---|---|---|---|
| Época 01 | `0.0140` | **`100.00%`** | `0` | `████████████████████` |
| Época 02 | `0.0131` | **`100.00%`** | `0` | `████████████████████` |
| Época 03 | `0.0125` | **`100.00%`** | `0` | `████████████████████` |
| Época 04 | `0.0119` | **`100.00%`** | `0` | `████████████████████` |
| Época 05 | `0.0115` | **`100.00%`** | `0` | `████████████████████` |

---

## 4. Matriz de Confusão Final (100% Calibrada)

| Real \ Previsto | `classe_A` | `classe_B` | `classe_C` | `classe_CONCHA` | `classe_I` | `classe_L` | `classe_PALMA_ABERTA` | `classe_V` | `classe_W` |
|---|---|---|---|---|---|---|---|---|---|
| `classe_A` | 94 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classe_B` | 0 | 111 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classe_C` | 0 | 0 | 111 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classe_CONCHA` | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 |
| `classe_I` | 0 | 0 | 0 | 0 | 103 | 0 | 0 | 0 | 0 |
| `classe_L` | 0 | 0 | 0 | 0 | 0 | 111 | 0 | 0 | 0 |
| `classe_PALMA_ABERTA` | 0 | 0 | 0 | 0 | 0 | 0 | 102 | 0 | 0 |
| `classe_V` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 107 | 0 |
| `classe_W` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 109 |

---

## 5. Pesos Punitivos Discriminativos por Junta

Pesos calibrados ($W_j$) para cada junta de 0 a 20 (onde valores $> 1.0$ penalizam desvios em juntas críticas):

| Classe | Peso Médio | Juntas Mais Discriminativas (Maior Peso) |
|---|---|---|
| `classe_A` | `1.52` | Junta 4 (W=2.88), Junta 3 (W=1.99), Junta 11 (W=1.98) |
| `classe_B` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_C` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_CONCHA` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_I` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_L` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_PALMA_ABERTA` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_V` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |
| `classe_W` | `1.00` | Junta 20 (W=1.00), Junta 9 (W=1.00), Junta 1 (W=1.00) |

---

## 6. Sincronização com o Ecossistema do Projeto

- `seeds_calibradas.json`: Atualizado na raiz e em `Treinamento IA/data/seeds/`.
- `calibration_settings.json`: Atualizado com os parâmetros biomecânicos em `Treinamento IA/data/`.
- `POC/seedsCalibradas.js`: Atualizado com exportação ES6 para o aplicativo React Native.
- `calibrated_classifier.py`: Consome nativamente o novo arquivo sem necessidade de adaptação.
