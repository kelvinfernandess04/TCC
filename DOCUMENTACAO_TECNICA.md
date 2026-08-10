# Documentação Técnica: Sistema de Reconhecimento de Libras (TCC)

Esta documentação descreve detalhadamente a arquitetura, lógica e funcionalidade de todos os scripts presentes no repositório. O objetivo é permitir que qualquer desenvolvedor compreenda o funcionamento interno do sistema e consiga reproduzir o ambiente de treinamento e execução.

---

## 1. Visão Geral do Projeto

O projeto é dividido em dois grandes pilares:

1. **Pipeline de Treinamento (`Treinamento IA/`)**: Responsável por converter imagens brutas e bases de dados em coordenadas matemáticas (landmarks) e treinar um modelo de Rede Neural Profunda (Deep Learning).
2. **Scripts de Execução e Teste (`scripts/`)**: Ferramentas para capturar novos dados em tempo real e testar a assertividade do modelo em um ambiente de "Sandbox".

---

## 2. Estrutura de Diretórios Principal

- `Treinamento IA/scripts/`: Scripts do núcleo de Inteligência Artificial.
- `scripts/`: Utilitários de interface com o usuário e testes.
- `Treinamento IA/data/`: Bases de dados (datasets), cache de processamento e arquivos unificados.
- `Treinamento IA/models/`: Onde o modelo treinado (`.h5` e `.tflite`) e as labels são salvos.
- `Treinamento IA/reports/`: Relatórios automáticos gerados após cada fase de extração ou treino.

---

## 3. Pipeline de Treinamento (`Treinamento IA/scripts/`)

O pipeline é orquestrado de forma sequencial para garantir a integridade dos dados.

## 3. Pipeline de Treinamento (`Treinamento IA/scripts/`)

O pipeline de Treinamento agora é centralizado e puramente sintético, eliminando a dependência de datasets empíricos ruidosos capturados por câmeras.

### 3.1 `synthetic_generator.py` (Motor Biomecânico)

**Função**: Construir um dataset matemático perfeito de LIBRAS a partir da cinemática de esqueleto 3D (Forward Kinematics).

- **Arquétipos Posturais**: Substitui os limites contínuos por 4 posturas definitivas por dedo (Estendido, Garra, Soco, Plataforma).
- **Avanço Cinesiológico**: Implementa a Lei de Landsmeer (dobra passiva da ponta do dedo), restrições dos *Connexus Intertendinei* (impede flexão do anelar isolado) e independência matricial do Polegar (CMC).
- **Sim-to-Real Câmera**: Utiliza uma onda triangular linear (`bounce_wave`) para rotacionar a mão virtual continuamente de -85º a +85º em Pitch e Yaw, ensinando a IA a ler a mão por silhuetas de perfis colapsados, perfeitamente como o MediaPipe enxerga em casos de oclusão.
- **Saída**: Gera um JSON com centenas de milhares de amostras perfeitas, prontas para o motor neural.

### 3.2 `neural_engine.py` (Motor Neural)

**Função**: Definir a arquitetura da rede neural e realizar o treinamento usando TensorFlow/Keras.

- **Algoritmo de Treinamento**: Rede Neural Profunda (DNN) do tipo Multilayer Perceptron (MLP).
- **Estrutura de Dados (Input)**: Recebe um vetor de 42 números (21 landmarks x 2 coordenadas X/Y).
- **Data Augmentation Simplificado**:
  - Como a base sintética já cobre toda a esfera espacial (rotações de 85º) com ruído embutido, não há necessidade de augumentation rotacional dinâmico.
  - **Ambidestria (Mirroring)**: Aplica o Flip X (inversão horizontal) de todas as amostras sintéticas perfeitas da mão direita, clonando-as para a mão esquerda, resultando em milhões de amostras de treinamento simultâneo.
- **Arquitetura do Modelo**:
    1. **Camada de Entrada**: 42 neurônios.
    2. **Camada Oculta 1 (128 neurônios)**: Ativação ReLU + BatchNormalization + Dropout (0.2).
    3. **Camada Oculta 2 (64 neurônios)**: Ativação ReLU + BatchNormalization + Dropout (0.2).
    4. **Camada Oculta 3 (32 neurônios)**: Ativação ReLU + BatchNormalization.
    5. **Camada de Saída**: Neurônios correspondentes ao número de classes (Softmax).
- **Otimizador**: Adam (Taxa de aprendizado de 0.001).
- **Função de Perda**: Sparse Categorical Crossentropy.

#### Resumo do Modelo de IA

O modelo desenvolvido **não é um fine-tuning** de modelos de imagem pré-existentes (como ResNet ou MobileNet). Em vez disso, utilizamos o **MediaPipe Holistic** como um extrator de características fixo e robusto. Nossa rede neural é treinada do zero para ser um **Classificador Geométrico Coordenativo**.

- **Vantagens**: Inferência extremamente rápida (sub-milissegundos), arquivo TFLite levíssimo (~200KB) e total imunidade a variações de iluminação, cor de pele ou fundo, pois foca exclusivamente na geometria do esqueleto.

**Saídas**:

- `modelo_gestos.h5`: Modelo para uso em scripts Python desktop.
- `modelo_gestos.tflite`: Versão otimizada para Web/Mobile.
- `labels.txt`: Lista ordenada de sinais que o modelo aprendeu.

---

## 4. Ferramentas de Teste e Captura (`scripts/`)

### 4.1 `realtime_trainer.py` (Captura Customizada)

**Função**: Permitir que o usuário grave novos sinais que não existem em datasets públicos para alimentar a IA.

- **Captura em Lote**: Grava sequências de frames (ex: 60 frames) e salva em arquivos JSON organizados por pasta de classe.
- **HUD Visual**: Exibe na tela o que a IA está prevendo no momento e uma barra de progresso da gravação.
- **Normalização em Tempo Real**: Aplica a mesma lógica de Bounding Box do treinamento para garantir que os dados capturados sejam idênticos ao que a IA espera.

### 4.2 `dynamic_sandbox.py` (Ambiente de Teste e Validação Dinâmica)

**Função**: Validar sinais do alfabeto (estáticos) e sinais dinâmicos (com movimento).

- **Validação Dinâmica (DTW)**: Implementa o algoritmo **Dynamic Time Warping** para comparar trajetórias.
  - **Centro da Palma**: Calcula o centro geométrico da mão em vez de usar apenas o pulso.
  - **Vetor Normal**: Identifica a orientação da palma (para onde a mão aponta).
  - **Referencial de Pose**: Mede a posição da mão relativa ao centro dos ombros do usuário.
- **Modo de Debug (Importação)**: Permite pressionar `[I]` para importar vídeos MP4 ou fotos para testar a IA sem necessidade de câmera ao vivo.
- **Gravação de Templates**: Permite criar assinaturas dinâmicas em JSON que servem de base para os exercícios.
- **Lógica de Pontuação Unificada**:
    1. **Forma Estática (30%)**: Fidelidade ao formato da mão.
    2. **Trajetória (50%)**: Precisão do movimento via DTW.
    3. **Orientação (20%)**: Correção da direção da palma.

---

## 5. Prova de Conceito (POC Mobile)

A POC é uma aplicação mobile desenvolvida para demonstrar a portabilidade do modelo treinado para dispositivos de uso cotidiano.

### 5.1 Arquitetura da POC

- **Framework**: React Native com Expo.
- **Runtime de IA**: Utiliza uma `WebView` para rodar o motor de inferência em JavaScript de alta performance.
- **Bibliotecas**:
  - `@tensorflow/tfjs-tflite`: Permite rodar o modelo `.tflite` diretamente no navegador/WebView.
  - `@mediapipe/holistic`: Implementação completa do MediaPipe para ambiente web.

### 5.2 Lógica de Funcionamento (`VisionProcessor.js`)

1. **Captura**: A WebView acessa a câmera do celular via `navigator.mediaDevices.getUserMedia`.
2. **Processamento (Holistic)**: O esqueleto da mão e do corpo é extraído pelo MediaPipe Holistic.
3. **Inferência**: Os pontos são normalizados (usando a mesma lógica de Bounding Box do Python) e enviados para o modelo TFLite injetado via Base64.
4. **Correção de Exibição (Cover Fix)**: Implementa uma função de mapeamento de coordenadas (`getScaledCoords`) que ajusta o desenho do esqueleto ao CSS `object-fit: cover` do dispositivo, garantindo que os pontos fiquem perfeitamente alinhados com a mão física na tela, independente da proporção do celular.
5. **Ponte de Comunicação (Bridge)**: O resultado da predição é enviado para o código nativo através de `window.ReactNativeWebView.postMessage`.

---

## 6. Unificação e Estética do Sistema

O projeto foi totalmente unificado sob o modelo **MediaPipe Holistic** para garantir que o treinamento seja 100% compatível com a execução em tempo real.

### 6.1 Correção de Espelhamento

Nas ferramentas de interface (`realtime_trainer.py` e `dynamic_sandbox.py`), o frame da câmera é invertido **antes** do processamento da IA. Isso garante que:

- Os pontos desenhados fiquem perfeitamente "colados" na mão do usuário.
- O dado processado pela IA seja idêntico ao que o usuário vê na tela (Mirror Mode).

### 6.2 Visual "Premium" (Esqueleto)

Seguindo o padrão do Sandbox, o Trainer e a POC Mobile agora exibem o esqueleto completo:

- **Pontos Brancos**: Representam as juntas dos dedos.
- **Linhas Verdes**: Representam as conexões (falanges) da mão.
- Esse visual facilita a calibração do gesto pelo usuário antes de realizar um teste ou gravação.

---

## 7. Como Reproduzir os Scripts

### Requisitos

- Python 3.10 ou 3.11.
- Bibliotecas: `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`.

### Passo a Passo

1. **Preparação Sintética**: Rode `python Treinamento IA/scripts/synthetic_generator.py` para compilar todo o JSON biomecânico base.
2. **Treinamento e Automação POC**: Execute `python Treinamento IA/scripts/neural_engine.py`.
    - O motor vai ler a base sintética, aplicar Ambidestria (Mirroring X), treinar o modelo, exportar o `.tflite` e atualizar a sua POC Javascript automaticamente!
3. **Teste Final**: Rode `python scripts/dynamic_sandbox.py` (Modo Visual) ou teste direto rodando a POC com Expo localmente no seu smartphone.

---

## 8. Abordagem Sintética (Forward Kinematics)

Neste projeto, houve um pivô estratégico visando maior escalabilidade e robustez. Em vez de depender apenas de dados capturados empiricamente por fotos, o modelo baseia-se em um conjunto de dados 100% sintético gerado matematicamente.

### 8.1 Lógica do Motor Sintético (`synthetic_generator.py`)
Localizado na pasta `Treinamento IA/scripts/`, este script gera configurações precisas de mãos:
- **Cinemática Direta (Matrizes de Rotação Euler)**: O modelo constrói o esqueleto da mão (21 landmarks) sem recorrer a ferramentas trigonométricas imprecisas, utilizando multiplicação contínua de matrizes Rx, Ry, Rz.
- **Arquétipos Posturais e Landsmeer**: Em vez de variar linearmente juntas, usamos 4 arquétipos fisiológicos definidos clinicamente. A junta DIP da ponta do dedo flexiona automaticamente em coordenação passiva com a junta PIP.
- **Matrizes Isoladas do Polegar**: As 4 fases de rotação da junta CMC (Aberto, Aduto Transversal, Oposição Plena e Gatilho) operam perfeitamente sem cruzamento ou anomalia "zig-zag" dos ossos.
- **Oclusão Biológica (*Connexus Intertendinei*)**: Previne algoritmicamente a geração de permutações humanamente impossíveis (ex: Anelar esticado enquanto os adjacentes estão fechados), podando o excesso de lixo gerado antes mesmo do treinamento da rede.
- **Mapeamento Cinesiológico**: A geração dinâmica resulta em mais de 450 "Classes Base", as quais representam fisicamente todo e qualquer sinal estático possível com uma mão humana, sem depender de classes alfabéticas restritas.

### 8.2 Automação da Prova de Conceito (Sim-to-Real Pipeline)
O fluxo encerra-se com a integração fluida no React Native:
- O script de treinamento importa o módulo de `update_poc.py`.
- O modelo `.tflite` recém-cozinhado com os dados sintéticos é codificado em Base64 e imediatamente embutido na aplicação da POC.
- Isso estabelece um clico automatizado: Qualquer melhoria geométrica implementada no motor sintético reflete imediatamente no celular do usuário logo após o término do pipeline, blindando o projeto contra bases empíricas falhas.

---

## 9. Lógica de Dados (Landmarks)

O sistema não olha para a "imagem" (pixels), mas para o esqueleto da mão.

- Cada mão tem **21 pontos (Landmarks)**.
- Cada ponto tem coordenadas **X e Y**.
- O vetor de entrada da IA é uma lista de **42 números** (21 pontos * 2 coordenadas).
- Toda a inteligência do sistema baseia-se na relação espacial entre esses 42 números, independente de cor de pele, fundo ou iluminação.

---

## 10. Calibrador Anatômico LIBRAS 3D (`Treinamento IA/scripts/hand_calibrator.py`)

O **Calibrador Anatômico LIBRAS 3D** é uma interface gráfica avançada desenvolvida em Tkinter, OpenCV e MediaPipe para inspecionar, calibrar e salvar as amplitudes articulares e poses biomecânicas da mão humana para a base sintética de gestos.

### 10.1 Cinemática Direta Totalmente Relativa

O calibrador implementa uma cadeia cinemática tridimensional pura baseada em matrizes de rotação Euler, onde:
1. **Pulso como único ponto fixo**: O ponto `p0 = [0, 0, 0]` é a única coordenada estática do modelo no espaço tridimensional.
2. **MCP (Junta 1) Móveis**: Diferente de modelos estáticos anteriores, a junta MCP (`p1`) de cada dedo se move de forma totalmente dinâmica e relativa. O ponto `p1` é obtido rotacionando a reta da base do metacarpo `palm_bases[finger]` a partir do pulso por meio da rotação tridimensional de `J1_Yaw` e `J1_Pitch`.
3. **Propagação Relativa em Cadeia**:
   - Cada junta subsequente (PIP/J2 e DIP/J3) é calculada rotacionando o segmento local do osso correspondente relativamente ao seu predecessor (osso anterior na cadeia).
   - Quando Yaw e Pitch são configurados como `0.0`, os ossos subsequentes ficam perfeitamente colineares e retilíneos, eliminando distorções como o desalinhamento de yaw no ponto 18.
4. **Controle Total e Desbloqueado**: Todos os eixos e juntas (MCP, PIP, DIP) de todos os dedos estão 100% livres para ajustes de Yaw e Pitch, tanto porSliders, caixas de entrada de texto quanto por arrasto direto tridimensional com o mouse sobre os pontos do canvas.

### 10.2 Calibração Real-Time com Congelamento (Webcam)

Para criar calibrações personalizadas rápidas, o usuário pode acionar a calibração com câmera:
- **Medição Automática**: O MediaPipe Holistic mede em tempo real os ângulos das juntas MCP e PIP/DIP enquanto o usuário mexe a mão.
- **Congelamento Seguro (Espaço)**: Ao clicar em **Espaço**, a tela de salvamento é aberta e a captura de dados de flexão (`live_ranges`) é **congelada**. A câmera continua renderizando a imagem em tempo real com a legenda `"PONTOS CONGELADOS"` em vermelho. O usuário pode retirar a mão de frente da câmera com a garantia de que as métricas acumuladas não sofrerão nenhuma corrupção.
- **Cancelamento e Unfreeze**: Caso feche a janela de salvamento ou clique em "CANCELAR", os pontos medidos voltam a acumular normalmente.
- **Foco Pós-Salvamento (Autofocus)**: Ao confirmar a calibração, a câmera é fechada e a interface principal foca automaticamente no primeiro dedo e estágio que foram salvos, redesenhando e ajustando a tela principal para visualização imediata da calibração adotada.

### 10.3 Ingestão Inteligente de JSON (Side-by-Side)

A interface de Ingestão de JSON foi projetada para ser amigável e instrutiva, dividida em duas colunas:
- **Painel Esquerdo (Documentação)**: Um guia dinâmico e ricamente formatado que detalha como o parser inteligente traduz chaves em português, remove acentos, normaliza termos de dedos e juntas, e aplica o modelo LERP.
- **Painel Direito (Editor)**: Área de colagem do código JSON equipada com botões rápidos de limpeza e um preenchimento automático de exemplo com placeholders instrutivos.

#### Modelo de Ingestão com Placeholders (Esquema JSON)

Abaixo está o modelo completo aceito pelo parser inteligente do calibrador:

```json
{
    "stages": {
        "_comment_1": "FORMATO POR INTERVALO (AUTOMÁTICO LERP PARA ESTÁGIOS 0-3)",
        "indicador": {
            "MCP": [5.0, 85.0],
            "PIP": [5.0, 110.0]
        },
        "medio": {
            "MCP": [5.0, 90.0],
            "PIP": [5.0, 115.0]
        },
        "anelar": {
            "MCP": [5.0, 80.0],
            "PIP": [5.0, 105.0]
        },
        "mindinho": {
            "MCP": [5.0, 85.0],
            "PIP": [5.0, 100.0]
        },
        "_comment_2": "FORMATO EXPLÍCITO (ESTÁGIOS ESPECÍFICOS)",
        "polegar": {
            "estagio_0": {
                "CMC_Yaw": -25.0,
                "CMC_Pitch": 5.4,
                "MCP_Pitch": 10.0,
                "IP_Pitch": 5.0
            },
            "estagio_3": {
                "CMC_Yaw": -21.2,
                "CMC_Pitch": 37.3,
                "MCP_Pitch": 50.0,
                "IP_Pitch": 60.0
            }
        }
    }
}
```

```

### 10.4 Visualizador de Pose (Testador de DADADADAFP)

O Gerador Sintético cria milhões de classes identificadas por um código sequencial de 10 dígitos na notação estrutural **DADADADAFP**. O Calibrador possui um "Visualizador de Código de Pose" na interface para visualizar exatamente qual conformação 3D o gerador associou a esse código numérico.

#### Estrutura do Padrão DADADADAFP

Cada dígito do código numérico de 10 posições descreve um atributo físico da mão da extremidade externa (Mindinho) para a interna (Polegar):

- `[0] D`: **Mindinho (Pinky State)** - Estágios de 0 a 3 (0=Aberto, 1=Garra, 2=Plataforma, 3=Fechado)
- `[1] A`: **Spread Mindinho-Anelar** - 0=Aberto/Afastado, 1=Fechado/Junto
- `[2] D`: **Anelar (Ring State)** - Estágios de 0 a 3
- `[3] A`: **Spread Anelar-Médio** - 0=Aberto/Afastado, 1=Fechado/Junto
- `[4] D`: **Médio (Middle State)** - Estágios de 0 a 3
- `[5] A`: **Spread Médio-Indicador** - 0=Aberto/Afastado, 1=Fechado/Junto
- `[6] D`: **Indicador (Index State)** - Estágios de 0 a 3
- `[7] A`: **Spread Indicador-Polegar** - 0=Aberto/Afastado, 1=Fechado/Junto
- `[8] F`: **Polegar - Oposição/Fold** - 0=Polegar lateral, 1=Polegar em oposição contra a palma (como no número 4 da LIBRAS)
- `[9] P`: **Polegar - Estado Principal** - Estados [0, 2, 3] ou simplificados (0=Aberto, 1=Fechado)

O Visualizador no Calibrador interpreta esses dígitos imediatamente e reflete as aberturas (yaw constraints), inclinações e estados no esqueleto 3D.

---
*Documentação gerada para o projeto TCC - Sistema Libras Engine.*


## Taxonomia DADADADAFP

A taxonomia DADADADAFP é mapeada da seguinte forma (índice 0 a 9, lendo do Mindinho para o Polegar):

1. **[D] Mindinho**: Flexão (Estágios 0 a 3)
2. **[A] Abertura Mindinho-Anelar**: Spread lateral (0 = Aberto, 1 = Fechado)
3. **[D] Anelar**: Flexão (Estágios 0 a 3)
4. **[A] Abertura Anelar-Médio**: Spread lateral (0 = Aberto, 1 = Fechado)
5. **[D] Médio**: Flexão (Estágios 0 a 3)
6. **[A] Abertura Médio-Indicador**: Spread lateral (0 = Aberto, 1 = Fechado)
7. **[D] Indicador**: Flexão (Estágios 0 a 3)
8. **[A] Abertura Indicador-Polegar**: Spread lateral (0 = Aberto, 1 = Fechado)
9. **[F] Movimento Transversal (Polegar)**: (0 = Mesmo plano, 1 = Na frente da palma)
10. **[P] Ponta do Polegar (IP)**: Flexão específica (0 = Aberta, 1 = Fechada)
