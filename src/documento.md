# PARTE A: ARQUITETURA LEGADA (v1.0 - v9.6)
> *Foco: Alinhamento Geométrico 3D e Minimização de Erro Euclidiano*

## 1. Visão Geral (v9.6)
A versão v9.6 representou o limite do motor puramente geométrico, utilizando a **Trindade 3D** (Shape, Position, Topology) e log-DTW.

A versão v9.6 solidifica a robustez do sistema através de refinamentos sequenciais em memória, ritmo e estabilidade espacial. O Threshold de Aprovação permanece em **`61.0`**, com **Especificidade de 100%**.

### 2.1 Pipeline de Processamento e Normalização 3D (v9.5)
1.  **Normalização Espacial 3D:** Utiliza a distância inter-ombros no espaço 3D para escala, garantindo resiliência a vídeos em close-up.
2.  **Âncoras Faciais Dinâmicas:** Centralização baseada no nariz e olhos com cálculo de profundidade (Eixo Z).
3.  **Filtragem de Repouso SMA:** Janela de 5 frames para imunidade a jittering.

### 2.2 Pilares de Scoring (Scoring Trinity v9.6)
*   **Forma (40%):** 20 ângulos articulares e **Matriz Topológica (20%)** para diferenciação semântica fina.
*   **Orientação (20%):** Cossenos da Palma e Pointer em 3D.
*   **Posição Estabilizada (15%):** Decomposição XY vs Z (Peso 1.0 vs 0.2). O eixo Z é atenuado para ignorar o jittering de profundidade do MediaPipe.
*   **Memória Temporal (Veto/Penalty):** `OCCLUSION_WINDOW` de 15 frames. Tolera desaparecimento súbito das mãos se o sinal de base esperar presença (Recupera sinais com oclusão).

### 2.3 Resiliência Rítmica e Temporal
*   **Log-DTW Gap Penalty:** Substituição da penalidade em degrau por uma curva logarítmica suave (`5.0 * log1p(gap)`). Permite variações naturais de velocidade do usuário sem punições drásticas.
*   **SMA On-the-Fly:** Redução de ruído integrada.

---

# PARTE B: ARQUITETURA SEMÂNTICA (v10.0+)
> *Foco: Categorização Fisiológica, Soft-Match e Probabilidades*

## 2. Nova Arquitetura: Biblioteca de Configurações de Mão (v10.1)

A v10.1 introduz o conceito de **Categorias de Mão**. Em vez de comparar ângulos brutos, o sistema identifica se a mão pertence a uma classe (ex: `CUP`) e aplica penalidades graduais por variações de execução.

### 2.1 Pilares do Motor v10.1
*   **Biblioteca de Templates:** 19 ângulos 3D (15 articulares + 4 abduções).
*   **Categorização por Classes:**
    *   `OPEN`: `OPEN_FLAT`, `OPEN_SPREAD`.
    *   `CUP`: `CUP_OPEN`, `CUP_DEEP`.
    *   `FIST`: `CLOSED_FIST`.
*   **Soft-Category Scoring:** Se o alvo for da mesma categoria da base (mesmo que sub-classe diferente), o score é mantido alto (ex: 90%).

### 2.2 Diferenciação Semântica e Refinamentos (v10.13)
O espaçamento entre dedos (Abdução) agora separa confiavelmente `Abacate` (dedos juntos) de `Abacaxi` (dedos espalhados), resolvendo o desafio da Margem Crítica.

### 2.3 Refinamentos Estruturais (v10.14 - v10.17)
1.  **Classe de Rejeição (Threshold Absoluto):** O sistema agora descarta classificações com Erro Quadrático Médio (MSE) acima de **250**, evitando a "Ilusão do Softmax" em mãos com geometrias inválidas.
2.  **Endurecimento Espacial:** Rebalanceamento do peso da distância entre as mãos (r_e = 0.5) e redução do divisor para **1.2**, tornando o motor extremamente sensível a erros de posicionamento.
3.  **Orientação Não-Linear:** Aplicação de potência (Power-4) ao produto escalar para penalizar desalinhamentos medianos de forma agressiva.
4.  **Threshold Final:** Calibrado em **64.0** para permitir recall de execuções orgânicas enquanto mantém alta especificidade.