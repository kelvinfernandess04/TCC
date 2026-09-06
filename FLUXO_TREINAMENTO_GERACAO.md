# Fluxo Técnico Profundo: Geração de Base Sintética e Treinamento da Inteligência Artificial

Este documento descreve detalhadamente a arquitetura de ponta a ponta e o pipeline de dados responsáveis pela criação da base de dados sintética e pelo treinamento da Inteligência Artificial de reconhecimento de LIBRAS. O sistema é estruturado em três pilares principais, localizados em `Treinamento IA/scripts`:

1. **`seed_extractor.py`**: O componente cinemático que define, através de regras biomecânicas e dados de calibração, as posições-semente (poses chave puras 3D).
2. **`synthetic_generator.py`**: O gerador de aumentação de dados esféricos, responsável pela transformação espacial, projeções e perturbações que darão origem à massiva base de dados de treinamento.
3. **`neural_engine.py`**: O motor de Deep Learning, projetado nativamente para eficiência de memória e orquestração do treinamento da Topologia MLP.

Este manual decompõe não apenas as funções de alto nível, mas também as nuances matemáticas, transformações de matriz e fluxos lógicos presentes na arquitetura do TCC.

---

## Pilar 1: O Extrator de Sementes Cinemáticas (`seed_extractor.py`)

O objetivo primário do motor de sementes é resolver a ambiguidade de poses 2D através da fusão de dados em múltiplos planos e, em seguida, construir modelos completos de mãos articuladas por meio de cinemática direta, calculando ponto a ponto (21 Landmarks).

### 1.1 Fusão Múltipla Planar (Dual Plane Fusion)
O motor inicial busca por um arquivo de calibração (`calibration_settings.json`). As calibrações geralmente capturam posições visuais da mão humana. Para lidar com distorções inerentes ao 2D, foi empregado um fusor planar:
- **`fuse_dual_plane_landmarks(pose_dict)`**: Se o sistema obteve tanto a posição frontal da mão (`front`) quanto lateral (`profile`), a função mescla-as para recriar profundidade. A malha frontal dita as coordenadas `X` e `Y`, enquanto o perfil é responsável por injetar a coordenada `Z` (profundidade). Isso extrai uma estrutura 3D de alta precisão que seria impossível com uma única câmera simples.

### 1.2 Cinemática Direta Híbrida (`generate_anatomical_hand_3d`)
Esta é a função central que gera posições articulares realistas iterativamente:
- **Fallback State**: Caso nenhuma calibração manual exista, o script injeta um modelo "Fallback" 3D padrão idealizado com as proporções da mão adulta estendida.
- **Estruturação por Vetores Base**: Utiliza a base metacarpal (`stage_0_closed` e `stage_0_spread`) como referência para identificar o comportamento da mão antes da movimentação dos dedos.
- **Cinemática dos 4 Dedos (Indicador a Mínimo)**: 
  - Percorre cada dedo através de seus *Landmarks* articulares iterando os vetores que compõem cada falange (falange proximal, média e distal).
  - Utiliza pesos decimais provindos do `finger_states` da label de origem para calcular matrizes de peso ($w$).
  - A lógica utiliza Interpolação Linear (*Lerp*) para dobrar o dedo em estágios matemáticos perfeitamente conectados, mapeando pontos 1 (Semi-Estendido), 2 (Articulação primária flexionada) e 3 (Punho serrado).
  - O abduzimento ou "Spread" (espalhamento dos dedos para os lados) aplica matrizes clássicas de Rotação Z (`rot_z`), por exemplo, girando o dedo mínimo lateralmente até 14º negativos.
- **Cinemática do Polegar (A Abordagem Isolada)**: 
  O polegar difere do restante devido a sua base carpometacárpica (CMC).
  - **Spread e Opposition**: O polegar desliza ao longo da frente da palma. Isso é orquestrado através da flexão do `thumb_opp` transversalmente na direção das juntas metacárpicas dos outros dedos, somado ao espalhamento entre o indicador e o polegar.
  - **Flexão IP**: Permite a quebra da falange distal individualmente, alterando de modo profundo apenas as coordenadas Z do topo do dedo.

### 1.3 Explosão Combinatória e Loop
A rotina primária de execução (`main`) realiza um ataque estruturado de matriz tridimensional (Nested Loops).
- Multiplica matematicamente os estados isolados: (4 estados para cada dedo) × (2 espalhamentos, dependendo da flexão) × (oposição polegar) × (flexão polegar).
- O loop varre toda essa permutação respeitando restrições, totalizando tipicamente **3.936 classes semânticas** cinemáticas viáveis.
- Elas são gravadas em formato estático puro num enorme arquivo de metadados (`seeds.json`).

---

## Pilar 2: A Geração de Base Sintética (`synthetic_generator.py`)

A rede neural não pode aprender robustez com apenas quase 4.000 poses isoladas; ela não seria resistente às rotações humanas naturais (movimentos de punho, torções no ar, posições de câmera oblíquas). O Gerador Sintético é concebido para aplicar translação massiva à base pura, chegando a milhões de frames aumentados.

### 2.1 Transformação Espacial por Rotação Euleriana
O coração desse simulador depende da aplicação sucessiva das funções de rotação que representam eixos puramente tridimensionais (Roll, Pitch, Yaw):
- **`rot_x`, `rot_y`, `rot_z`**: Construções matriziais de rotação padrão trigonométrica que geram translações angulares $3\times3$ com base nos graus (`radians`). Multiplicá-las contra os pontos 3D da mão gera translações complexas no espaço simulado.

### 2.2 Desconstrução e Ajustes Anatômicos Avançados
Dentro de `generate_hand_3d` do motor gerador (quando roda proceduralmente, de forma "fall back" a cinemática não ancorada nas sementes):
- Define matrizes espaciais para a localização nativa de cada dedo.
- Aplica um algoritmo rigoroso de restrição de espalhamento (*Rule Spread Constraint*) caso se deseje forçar que dedos interligados por tendão flexor afetem vizinhos. A translação base aplica até 50 graus no polegar em certas oposição anatômicas calculadas através de determinantes.

### 2.3 Varredura Contínua 3D (Spherical Bouncing)
Cada uma das 3.936 "classes-semente" passa por uma esteira de produção para gerar um conjunto massivo de sub-frames. O modelo padrão determina `SAMPLES_PER_STATE = 600`. Isso gera tipicamente mais de 2,3 milhões de instâncias exclusivas.
- Para obter 600 exemplos rotacionais puros, utiliza-se a técnica de oscilação contínua: **`bounce_wave(progress, cycles)`**.
- A onda matemática perfeitamente reflete as rotações.
  - O **Roll** faz 2 giros totais simulando rotação total da câmera.
  - O **Pitch** reflete até 65 graus positivos e negativos (inclinação).
  - O **Yaw** reflete até 65 graus sobre a base esquerda/direita.

### 2.4 Projeção na Câmera, Normalização e Adição de Ruído de Sensor
As imagens de uma câmera web são, essencialmente, planificadas (2D) pelo sensor. O gerador aplica:
- **Z-Factor Simulation (Perspective Projection)**: Transforma `(X, Y, Z)` para `(X', Y')`. Onde $Z\_factor = offset / (offset - Z)$. Esse truque simples de projeção cria dilatação nas articulações mais próximas à câmera.
- **`normalize_and_add_noise`**: Processo crítico para prevenir "overfitting":
  1. Cria uma "bounding-box" perfeita para alinhar X e Y.
  2. Escalona ambos para sempre ficarem contidos no intervalo relacional `[0.0, 1.0]`.
  3. Adiciona **Ruído Gaussiano** via `random.gauss(0, 0.005)`. Essa perturbação minúscula simula tremores de câmera, distorção de inferência do MediaPipe em tempo real, sujeira na lente do usuário, luz baixa e outros fenômenos de captura "suja".

Ao longo de minutos (com monitoramento de progresso por logging e ETA em tempo real), o motor deposita esse mar de dados como arquivos locais de classe espalhados em centenas de diretórios, cada um sendo a personificação de uma combinação de falanges.

---

## Pilar 3: O Treinamento Profundo e o Motor Neural (`neural_engine.py`)

Com os dados espalhados localmente (na casa dos milhões), a engenharia de como treinar os dados sem que a RAM de um computador (por mais robusta que seja) sofra "Out-of-Memory / Kernal Panic" foi um desafio arquitetural fundamental do `neural_engine.py`. O script trabalha por fases estritamente modulares.

### FASE 1: Construção de Cache Serializado Incremental (`convert_json_to_npz`)
A abertura e o parsing de milhões de entradas num dicionário JSON é astronomicamente custoso no Python, gastando quase 40 vezes o espaço da RAM do que bytes de binário.
- O sistema varre os arquivos JSON contendo a matriz 21x2 recém-criada, em lotes isolados de um arquivo por vez, garantindo uso quase nulo de memória da máquina.
- As coordenadas são convertidas num padrão *Flattened Input*: os 21 pontos originais sofrem ancoragem no eixo de pulso (`landmark[0]`). Todo ponto `i` da matriz subtrai as coordenadas do pulso, garantindo invariância de translação posicional global. As coordenadas espalhadas são então "amassadas" de um formato de lista em um formato linear (Array tamanho 42).
- Os blocos numéricos puros são salvos numa base cache compressa em formato Numpy Zip (`.npz`).
- **Natureza Incremental**: O sistema é programado para olhar a estampa temporal do SO. Ele converte e guarda no disco, só reprocessando se houverem mudanças no timestamp do original.

### FASE 2: Engine de Memory-Mappping e Data Augmentation Interna
Com dezenas de `.npz` disponíveis na cache unificada:
- O sistema aloca prematuramente **toda** a necessidade de memória bruta em um Array Vazio de formato C `np.empty`, calculado dinamicamente, evitando o brutal esforço computacional e de coletas do *Garbage Collector* de fazer `np.concatenate` em tempo de execução.
- **Espelhamento Lateral Algorítmico em RAM**: Aumentar ainda mais as mãos é necessário para dar aos canhotos e posições reversas o tratamento adequado pela IA.
  - Ao povoar o *Array* da RAM, ele deposita os valores puros na primeira metade.
  - Ele simultaneamente deposita-os na segunda metade do Array.
  - Em uma execução vetorizada ultrarrápida (uma operação CPU pura `X_data[mirror_offset:, 0::2] *= -1`), ele varre todas as coordenadas de eixo X invertendo o seu aspecto. Assim nascem dezenas de milhares de dados para as "Mãos esquerdas" de forma programática imediata, em frações de segundo.

O TensorFlow engole toda essa array massiva utilizando um construtor de *pipelines* `tf.data.Dataset`, aplicando *batch size* e utilizando da configuração `AUTOTUNE` para a entrega de bytes sob demanda entre CPU (pré-processamento) e GPU (alocação Tensors).

### FASE 3 e 4: A Topologia de Múltiplas Camadas (MLP) e o Treino
O coração neurológico utiliza Keras em cima do TensorFlow 2.x para instanciar a estrutura:
- **Rede Neural Sequencial Simples (FFNN)**: Uma Rede Neural Feedforward provou-se altamente responsiva para 42 instâncias espaciais achatadas e limpas, se saindo em benchmark inferencial muitas dezenas de vezes melhor e menos pesada que Redes Convolucionais.
- A entrada recebe as *features* no tamanho exato `(42,)`.
- Ela sofre desidratação (Downsampling/Dimensionality Reduction) progressiva passando por matrizes de Neurônios Densamente conectadas: **512 -> 256 -> 128 -> Softmax Classes**.
- Utiliza **`BatchNormalization`**: Reduz variações estatísticas (*covariate shift*) durante o treinamento, acelerando dramaticamente a acurácia no decorrer das épocas.
- Utiliza **`Dropout(0.2)`**: Uma taxa aleatória de neurônios que param de responder propositalmente (20%). Previne a rede de "memorizar as posições exatas". Se forçada a advinhar com pontos cegos aleatórios, a rede extrai os padrões generalizados ocultos do objeto (Mão).
- **EarlyStopping e Checkpoint**: A fase de compilação utiliza o Otimizador `Adam`. Ela tenta por até `150` épocas, mas, equipada do *Callback EarlyStopping* `(patience=15)`, se a IA parar de aprender, ela detém o ciclo dezenas de épocas mais cedo, regressando seu estado de volta para o último `.h5` vitorioso na vala de verificação (*restore best weights*).

### FASE 5 e FASE 6: Conversão Edge (TFLite) e POC
Quando o modelo termina com alta métrica (usualmente beirando altas casas decimais de acurácia):
- É salvo no pesado formato hierárquico `modelo_gestos.h5`.
- Em seguida, passa pelo filtro **`TFLiteConverter`** do TensorFlow Lite. Todas as conexões do tensor que antes suportavam dezenas de atributos paralelos no TensorFlow Desktop/Cloud, são simplificadas. Variáveis de precisão flutuante passam por otimização que geram um modelo leve (arquivos usualmente com menos de 3MB ou até kBs).
- Os Rótulos (*labels*) convertidos pelo sklearn `LabelEncoder` retroativamente viram literais textuais salvos no mapa `labels.txt`.
- O código de Prova de Conceito Front-End (POC) é atualizado, garantindo que o programa de tempo real, sem dependências colossais do tensor local, inicie a leitura da webcam local perfeitamente amparado pela rede gerada. 
