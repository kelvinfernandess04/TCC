# Documentação Técnica: Sistema de Reconhecimento de Libras (TCC)

Esta documentação descreve detalhadamente a arquitetura, lógica e funcionalidade de todos os scripts presentes no repositório. O objetivo é permitir que qualquer desenvolvedor compreenda o funcionamento interno do sistema e consiga reproduzir o ambiente de treinamento e execução.

---

## 1. Visão Geral do Projeto

O projeto é dividido em dois grandes pilares:
1.  **Pipeline de Treinamento (`Treinamento IA/`)**: Responsável por converter imagens brutas e bases de dados em coordenadas matemáticas (landmarks) e treinar um modelo de Rede Neural Profunda (Deep Learning).
2.  **Scripts de Execução e Teste (`scripts/`)**: Ferramentas para capturar novos dados em tempo real e testar a assertividade do modelo em um ambiente de "Sandbox".

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

### 3.1 `treinamento.py` (O Orquestrador)
**Função**: Centralizar a execução de todas as fases do projeto.
-   **Fase 1**: Chama o `dataset_extractor.py` para processar as imagens.
-   **Fase 2-5**: Chama o `neural_engine.py` para treinar a rede neural.
-   **Fase 6**: Chama o `update_poc.py` (opcional) para atualizar o front-end com os novos modelos.
-   **Lógica**: Utiliza `subprocess.call` para garantir que uma fase só comece se a anterior terminar com sucesso (código de retorno 0).

### 3.2 `dataset_extractor.py` (Extração de Landmarks)
**Função**: Converter imagens/vídeos em pontos (landmarks) da mão usando MediaPipe, com alta performance via Multiprocessing.

-   **Multiprocessing (`ProcessPoolExecutor`)**: Utiliza `N-1` núcleos do processador para processar milhares de imagens em paralelo.
-   **Sistema de Cache (`extraction_cache.json`)**: Salva o resultado de cada imagem processada. Se o script for rodado novamente, ele só processa arquivos novos ou alterados (baseado em Hash/Mtime), economizando horas de processamento.
-   **Whitelist**: Apenas as classes definidas em `ALLOWED_LABELS` são processadas.
-   **Suporte NPY**: Consegue ler datasets virtuais gigantes (como o de 27 classes) sem carregar tudo na RAM, usando `mmap_mode`.

**Principais Funções**:
-   `process_chunk()`: O "trabalhador" que roda em cada núcleo. Inicializa o MediaPipe e processa um lote de imagens.
-   `run_extraction()`: Varre os diretórios, identifica o que precisa ser processado e distribui o trabalho entre os núcleos.

### 3.3 `neural_engine.py` (Motor Neural)
**Função**: Definir a arquitetura da rede neural e realizar o treinamento usando TensorFlow/Keras.

-   **Arquitetura do Modelo**: Uma rede `Sequential` com camadas `Dense` (Totalmente Conectadas), `BatchNormalization` (para estabilizar o treino) e `Dropout` (para evitar Overfitting).
-   **Data Augmentation**: Como as mãos podem estar rotacionadas ou em tamanhos diferentes, o script gera variações artificiais (Rotação, Ruído, Escala e Espelhamento/Ambidestria) para tornar o modelo mais robusto.
-   **Normalização Bounding Box**: Crucial para o sistema funcionar independente da distância da câmera. Ele centraliza a mão em um quadrado de 0 a 1.
-   **Early Stopping**: Interrompe o treino automaticamente se o modelo parar de evoluir, restaurando os melhores pesos.

**Saídas**:
-   `modelo_gestos.h5`: Modelo para uso em scripts Python desktop.
-   `modelo_gestos.tflite`: Versão otimizada para Web/Mobile.
-   `labels.txt`: Lista ordenada de sinais que o modelo aprendeu.

---

## 4. Ferramentas de Teste e Captura (`scripts/`)

### 4.1 `realtime_trainer.py` (Captura Customizada)
**Função**: Permitir que o usuário grave novos sinais que não existem em datasets públicos para alimentar a IA.

-   **Captura em Lote**: Grava sequências de frames (ex: 60 frames) e salva em arquivos JSON organizados por pasta de classe.
-   **HUD Visual**: Exibe na tela o que a IA está prevendo no momento e uma barra de progresso da gravação.
-   **Normalização em Tempo Real**: Aplica a mesma lógica de Bounding Box do treinamento para garantir que os dados capturados sejam idênticos ao que a IA espera.

### 4.2 `dynamic_sandbox.py` (Ambiente de Teste Definitivo)
**Função**: Validar sinais do alfabeto (estáticos) e sinais dinâmicos (com movimento).

-   **Modo IA Direta**: Avalia continuamente o que a câmera vê e mostra a confiança da predição.
-   **Interface de Digitação**: Usa OpenCV para permitir que o usuário digite o sinal que deseja testar sem sair da janela de vídeo.
-   **Lógica de Pontuação**:
    1.  **Precisão Temporal**: Quantos frames dos 2 segundos gravados bateram com a letra alvo.
    2.  **Confiança Média**: Qual foi o "grau de certeza" médio da IA durante o teste.
-   **Ambidestria**: Processa tanto mão esquerda quanto direita simultaneamente.

---

## 5. Prova de Conceito (POC Mobile)

A POC é uma aplicação mobile desenvolvida para demonstrar a portabilidade do modelo treinado para dispositivos de uso cotidiano.

### 5.1 Arquitetura da POC
- **Framework**: React Native com Expo.
- **Runtime de IA**: Utiliza uma `WebView` para rodar o motor de inferência em JavaScript de alta performance.
- **Bibliotecas**:
    - `@tensorflow/tfjs-tflite`: Permite rodar o modelo `.tflite` diretamente no navegador/WebView.
    - `@tensorflow-models/hand-pose-detection`: Implementação do MediaPipe para ambiente web.

### 5.2 Lógica de Funcionamento (`VisionProcessor.js`)
1.  **Captura**: A WebView acessa a câmera do celular via `navigator.mediaDevices.getUserMedia`.
2.  **Processamento**: O esqueleto da mão é extraído pelo MediaPipe Hands.
3.  **Inferência**: Os pontos são normalizados (usando a mesma lógica de Bounding Box do Python) e enviados para o modelo TFLite injetado via Base64.
4.  **Ponte de Comunicação (Bridge)**: O resultado da predição (Letra e Confiança) é enviado da WebView para o código nativo (React Native) através de `window.ReactNativeWebView.postMessage`.
5.  **Interface**: O React Native recebe esses dados e renderiza os feedbacks visuais de forma fluida.

---

## 6. Como Reproduzir os Scripts

### Requisitos
- Python 3.10 ou 3.11.
- Bibliotecas: `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`.

### Passo a Passo
1.  **Preparação**: Coloque suas pastas de imagens em `Treinamento IA/data/datasets/`. Cada pasta deve ter o nome da letra (ex: `A/`, `B/`).
2.  **Extração e Treino**: Execute `python Treinamento IA/scripts/treinamento.py`. 
    - Acompanhe o log. O cache será gerado primeiro, seguido pelo treino da rede neural.
3.  **Captura de Novos Dados**: Se quiser adicionar um sinal próprio, rode `python scripts/realtime_trainer.py`, pressione `[R]` para gravar e digite o nome do sinal no terminal quando solicitado.
4.  **Teste Final**: Rode `python scripts/dynamic_sandbox.py`. Use a tecla `[T]` para testar sua performance contra o modelo treinado.

---

---
## 7. Lógica de Dados (Landmarks)

O sistema não olha para a "imagem" (pixels), mas para o esqueleto da mão.
-   Cada mão tem **21 pontos (Landmarks)**.
-   Cada ponto tem coordenadas **X e Y**.
-   O vetor de entrada da IA é uma lista de **42 números** (21 pontos * 2 coordenadas).
-   Toda a inteligência do sistema baseia-se na relação espacial entre esses 42 números, independente de cor de pele, fundo ou iluminação.

---
*Documentação gerada para o projeto TCC - Sistema Libras Engine.*
