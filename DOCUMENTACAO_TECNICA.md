# Documentação Técnica: Sistema de Reconhecimento de Libras (TCC)

Esta documentação descreve detalhadamente a arquitetura, lógica e funcionalidade de todos os scripts presentes no repositório após a atualização para o novo pipeline de extração baseada em vídeo contínuo. O objetivo é permitir que qualquer desenvolvedor compreenda o funcionamento interno do sistema e consiga reproduzir o ambiente de calibração, treinamento e execução.

---

## 1. Visão Geral do Projeto

O projeto é dividido em dois grandes pilares:

1. **Pipeline de Calibração e Treinamento (`Treinamento IA/`)**: Responsável por extrair limites biomecânicos precisos do usuário a partir de um único vídeo de calibração, convertê-los em sementes anatômicas, expandi-los sinteticamente para um dataset massivo e treinar o modelo de Inteligência Artificial.
2. **Ambiente de Validação (`scripts/`)**: Ferramentas para testar a assertividade do modelo treinado.

---

## 2. Estrutura de Diretórios Principal

- `Treinamento IA/scripts/`: Scripts do núcleo de calibração, extração e Inteligência Artificial.
- `Treinamento IA/data/`: Configurações de calibração (`calibration_settings.json`), vídeos gravados, datasets sementes e sintéticos gerados.
- `Treinamento IA/models/`: Onde o modelo treinado (`.h5` e `.tflite`) e as labels são salvos.
- `Treinamento IA/reports/`: Relatórios automáticos e renderizações de verificação da anatomia da mão em cada fase.

---

## 3. Pipeline de Extração e Treinamento (`Treinamento IA/scripts/`)

O novo pipeline centraliza toda a extração de dados em uma única gravação de calibração biomecânica, eliminando a necessidade de digitar ângulos manualmente ou usar bases empíricas ruidosas.

### 3.1 Captura e Calibração
O usuário grava os movimentos limites de sua mão, e o sistema extrai automaticamente as poses extremas.
- **`video_recorder.py`**: Aplicação para gravar o vídeo biomecânico, salvando apenas a renderização dos landmarks anatômicos em tela preta (por questões de privacidade) no diretório de gravações.
- **`video_calibrator.py`**: Roda algoritmos de análise contínua no vídeo gravado para encontrar as extensões máximas e ângulos limites.
- **`video_inspector.py`**: Permite ao desenvolvedor inspecionar o vídeo com uma timeline interativa, verificar a telemetria em tempo real (ângulos exatos e aberturas) e confirmar ou reatribuir os *keyframes* fundamentais (Mão aberta, Garra, Gancho, Punho fechado, Oposição do Polegar, etc.).
- **`hand_calibrator.py`**: Painel de controle (Hub UI) que integra gravador, inspetor e renderiza as coordenadas finais extraídas em um simulador 3D interativo para verificação.

### 3.2 Geração de Base (Seeds e Dataset Sintético)
- **`seed_extractor.py`**: Lê o `calibration_settings.json` exportado pelo inspetor e realiza o *Morphing* linear entre os *keyframes* capturados (ex: morphing do Estágio 0 para o Estágio 3). Ele gera um banco de milhares de sementes perfeitamente limitadas pela mecânica da mão real do usuário, salvando-as em `seeds.json`.
- **`generate_seed_limit_visualizations.py`**: Processa o `seeds.json` e gera os 4 painéis de relatório gráfico na pasta `reports/` permitindo auditoria visual dos eixos de dobras e spreads anatômicos extraídos.
- **`synthetic_generator.py` (Motor Biomecânico)**: Pega as sementes e as submete a rotações e variações espaciais rigorosas simulando pontos de vista de câmeras 3D, produzindo o banco de dados sintético final massivo (`synthetic_dataset.json`).

### 3.3 Treinamento da Inteligência Artificial
- **`neural_engine.py` (Motor Neural)**: Roda a construção da Deep Neural Network (DNN) com as amostras sintéticas processadas e exporta o modelo TFLite. Otimizado para ignorar variações de iluminação por usar coordenadas 3D estritas.

---

## 4. Teste e Validação

- **`dynamic_sandbox.py`**: Ambiente interativo em tempo real via Webcam onde o usuário pode observar o MediaPipe rodando em seu corpo e a resposta imediata da rede neural (`modelo_gestos.h5`) classificando a taxonomia LIBRAS DADADADAFP desenvolvida.

---

## 5. Taxonomia DADADADAFP

O modelo baseia a classificação em uma string de 10 dígitos (ex: `0100000000`), sendo cada dígito lido do Mindinho em direção ao Polegar:

1. **[D] Mindinho**: Flexão (Estágios `0` a `3`).
2. **[A] Abertura Mindinho-Anelar**: Spread lateral (`0` = Aberto, `1` = Fechado).
3. **[D] Anelar**: Flexão (Estágios `0` a `3`).
4. **[A] Abertura Anelar-Médio**: Spread lateral (`0` = Aberto, `1` = Fechado).
5. **[D] Médio**: Flexão (Estágios `0` a `3`).
6. **[A] Abertura Médio-Indicador**: Spread lateral (`0` = Aberto, `1` = Fechado).
7. **[D] Indicador**: Flexão (Estágios `0` a `3`).
8. **[A] Abertura Indicador-Polegar**: Spread lateral (`0` = Aberto, `1` = Fechado).
9. **[F] Movimento Transversal (Polegar)**: Posição do polegar em relação à palma (`0` = Aberto/Plano da Mão, `1` = Oposição).
10. **[P] Ponta do Polegar (IP)**: Flexão específica da falange distal do polegar (`0` = Aberta, `1` = Dobrada).

---

## 6. Documentação Extensa dos Scripts

Esta seção documenta pasta a pasta e script a script, fornecendo o nível de detalhe necessário para a recriação do sistema a partir do zero ou manutenção profunda.

### 6.1 Pasta: `Treinamento IA/scripts/`

O coração da lógica do sistema, onde ocorre a captura, calibração biomecânica, geração de base de dados e treinamento da inteligência artificial.

#### `video_recorder.py`
- **Funcionalidade**: Aplicação gráfica (Tkinter) para gravar o usuário executando uma série de movimentos biomecânicos limites.
- **Detalhes Técnicos**: 
  - Inicializa o MediaPipe Hands e captura frames da webcam via `cv2.VideoCapture`. 
  - Não grava a imagem real em RGB do usuário (garantindo privacidade dos dados da pessoa surda/usuário), mas sim um fundo preto com os landmarks anatômicos desenhados (`black_frame`).
  - Salva o vídeo final em `data/recordings/` com codec `.mp4` (H264/mp4v). Paralelamente, armazena um arquivo `_landmarks.json` com as coordenadas brutas normalizadas e espaciais (X, Y, Z) extraídas em tempo real frame a frame, para agilizar a etapa do calibrador sem necessitar rodar o MediaPipe novamente.
  - Oferece um roteiro de movimentos (espalmar mão, fechar mão, garras, gancho, oposição de polegar e aberturas).

#### `video_calibrator.py`
- **Funcionalidade**: Analisa automaticamente o vídeo recém-gravado para extrair as poses extremas (*keyframes*) e limites anatômicos da mão do usuário calibrador.
- **Detalhes Técnicos**:
  - Calcula a flexão de cada dedo individualmente e a abertura (spread) utilizando trigonometria profunda (vetores 3D e ângulos entre falanges) via a função auxiliar `joint_flexion`.
  - Percorre todos os quadros e aplica uma heurística lógica para detectar precisamente:
    - *Stage 0 Spread*: Mão aberta em leque máximo (baixa flexão média dos dedos, altíssimo spread lateral medido em graus).
    - *Stage 0 Closed*: Mão estendida, dedos perfeitamente juntos (baixa flexão, baixo spread).
    - *Stage 1*: Garra (Flexão média ao redor de 120°).
    - *Stage 2*: Plataforma / Gancho (MCP reto, mas as juntas PIP/DIP hiper flexionadas).
    - *Stage 3*: Punho totalmente fechado.
    - *Thumb Opposition*: Frame com a menor distância Euclidiana entre a ponta do polegar e o metacarpo (MCP) do dedo médio.
    - *Thumb IP Flexed*: Frame de maior flexão da falange distal do polegar, mas com seu MCP reto.
  - Os frames chave isolados por esse motor são salvos permanentemente em `calibration_settings.json`, junto das proporções e comprimentos ósseos extraídos da mão do indivíduo.

#### `video_inspector.py`
- **Funcionalidade**: Ferramenta de auditoria e validação visual de vídeo. Permite ao desenvolvedor ou pesquisador avançar frame a frame e auditar o trabalho feito pelo analisador automático.
- **Detalhes Técnicos**:
  - Implementado em Tkinter com Canvas customizado de renderização 3D, permitindo rotacionar o esqueleto (`pitch` e `yaw`) com arrasto de mouse para inspeções cirúrgicas de dobras.
  - Possui painéis telemétricos imprimindo ao vivo a flexão somada (graus) de cada dedo e a distância transversal do polegar.
  - O usuário pode sobrescrever a decisão do robô de calibração clicando em "Atribuir Quadro Atual", forçando que a calibração de um estágio específico utilize o frame visualizado no player.
  - Ao concluir, o botão "Salvar Calibração Oficial" escreve o manifesto robusto contendo `captured_poses` para o motor gerador.

#### `hand_calibrator.py`
- **Funcionalidade**: Hub centralizador da interface de calibração iterativa.
- **Detalhes Técnicos**: Orquestra a interconexão entre as ferramentas visuais e os módulos de otimização, centralizando logs e saídas de dados do calibrador estático e dinâmico.

#### `iterative_calibrator.py`
- **Funcionalidade**: Refinamento e otimização não-linear da cinemática teórica da mão.
- **Detalhes Técnicos**:
  - Atua como uma ponte entre a matemática purista (`HandKinematicsDirect`) e a realidade empírica (`calibration_settings.json`).
  - Emprega o método de otimização limit-bound `L-BFGS-B` da biblioteca `scipy.optimize`. 
  - Minimização de Função de Perda (Loss): Mede a distância geométrica entre as posições assumidas pelas equações da FK (Forward Kinematics) e as posições reais medidas no vídeo de calibração, calibrando os fatores rotacionais dos ângulos estritos até chegarem na maior acurácia morfológica. 
  - Atualiza o output em arquivos persistentes de `seeds_calibradas.json` agregando os pesos de descida.

#### `seed_extractor.py`
- **Funcionalidade**: É o Extrator Anatômico Cinemático Híbrido, responsável por unir os *keyframes* capturados no mundo real à árvore de taxonomia LIBRAS DADADADAFP.
- **Detalhes Técnicos**:
  - A função `generate_anatomical_hand_3d` funde matrizes (`fuse_dual_plane_landmarks`).
  - Baseando-se nas posições primordiais do `calibration_settings.json` (p_0_spread, p_1, p_2, p_3), ela constrói um algoritmo de Morphing / Interpolação Linear. Exemplo: Para o estado D2.5, a função sabe exatamente como interpolar a geometria entre o frame do Stage 2 e o frame do Stage 3 gravados no vídeo.
  - Varre *todas as combinações válidas* da taxonomia de dedos, espalhamentos e polegares, gerando de forma automatizada e híbrida cerca de milhares de poses únicas. A base fundamental (seed) do projeto é condensada e registrada no arquivo `seeds/seeds.json`.

#### `generate_seed_limit_visualizations.py`
- **Funcionalidade**: Script de relatórios gráficos de validação ortopédica (Auditoria Visual).
- **Detalhes Técnicos**: 
  - Lê o volumoso `seeds.json` e busca as sementes exatas correspondentes aos extremos lógicos (Padrão 100% aberto, Padrão 100% fechado, Spreads em leque, Configurações de letras de LIBRAS A, V, W, I).
  - Desenha um canvas limpo projetando os ossos da mão de uma visão 3D para um array 2D e gera quatro figuras (`.png`) compostas, que são depositadas no repositório em `reports/seed_verification/`. Garante que os *morphings* matemáticos não geraram ossos distorcidos.

#### `synthetic_generator.py` (Motor Biomecânico)
- **Funcionalidade**: Fabrica o Dataset Massivo sintético (Data Augmentation Baseada em Mecânica, não em imagem). 
- **Detalhes Técnicos**:
  - Lê cada semente limpa registrada no passo anterior.
  - Emprega matrizes de rotação de Euler (`rot_x`, `rot_y`, `rot_z`) simulando órbitas de câmeras num domo acima da mão com o algoritmo `bounce_wave` para simular uma Varredura Contínua 3D perfeita (passando por inclinações angulares variadas de visões frontais a laterais e top-down).
  - Projeta os pontos 3D girados com correção de perspectiva Z para extrair as posições pseudo-2D.
  - Injeta *Ruído Gaussiano* de sensores em microescala para forçar a inteligência artificial a não decorar os pontos, aumentando dramaticamente a capacidade de generalização e resiliência a câmeras ruins no mundo real.
  - Gera pastas por classe dentro de `data/datasets/synthetic_dataset/`.

#### `neural_engine.py` (Motor Neural)
- **Funcionalidade**: Pipeline autônomo, robusto e multi-etapas de Treinamento e Deploy da Inteligência Artificial em Deep Learning.
- **Detalhes Técnicos**:
  - **Fase 1 (Conversão p/ NPZ)**: Para processar o dataset gigantesco sem MemoryLeak, ele converte em blocos (incrementalmente) os arquivos `.json` em matrizes `.npz` da biblioteca `numpy`, comprimidas para velocidade altíssima e uso mínimo de I/O.
  - **Fase 2 (Carregamento Array)**: Carrega do disco pre-alocando Arrays de espaço imutável, e duplica espelhando o eixo X para forçar *Data Augmentation* nativo suportando destros e canhotos sem distinção. Divide em Treino (85%) e Validação (15%) acoplado em `tf.data.Dataset` (Performance `AUTOTUNE`).
  - **Fase 3 (Construção Keras)**: Montagem de uma rede `Sequential` de múltiplas camadas densas (512 -> 256 -> 128 neurônios) utilizando ativações ReLU, `BatchNormalization` para aceleração do gradiente, e regularizadores `Dropout(0.2)` contra *overfitting*. A saída usa `Softmax` (Multiclasse categórica correspondente às dezenas/centenas de taxonomias LIBRAS).
  - **Fase 4 (Treinamento)**: Emprega o Otimizador `Adam(lr=0.001)` e uma rotina rigorosa de `EarlyStopping` (patience 15) que devolve a rede para o estado mais performático caso o Loss de validação pare de descer.
  - **Fase 5 (Build/Export)**: Salva a matriz de pesos treinada no arquivo hierárquico `modelo_gestos.h5`, extrai e sanitiza os rótulos preditivos codificados no `LabelEncoder` para `labels.txt`, e realiza compilação ultra compacta para TFLite (TensorFlow Lite), permitindo a portabilidade da IA para C++, Java e Mobile.

#### `calibrated_classifier.py`
- **Funcionalidade**: Agente classificador matemático que trabalha por similaridade e matriz de tolerância rigorosa, servindo como uma IA Explicável/Dura em oposição ao modelo Deep Learning tradicional. Utilizado ativamente no pipeline de calibração e em cenários onde "False-Positives" não são toleráveis.
- **Detalhes Técnicos**:
  - Normaliza qualquer Input de imagem da câmera (via função abstrata baseada nos mesmos cálculos do `Agent2_SpatialNormalizer`) convertendo de pixels pra relações geométricas.
  - O casamento é obtido pelo comparativo do frame do usuário com o arquivo `seeds_calibradas.json` em uma mecânica mista: Avalia a Distância Euclidiana Ponderada pelas juntas (valorizando falanges com pesos maiores, `punitive_weights`) unida a uma verificação de Distância e Similaridade Cossecante (Cosine Similarity) da direção vetorial da mão.
  - Entrega no final um dossiê `finger_errors` discriminando precisamente qual dedo reprovou a validação, se foi erro da tolerância standard (desvio padrão) de um cluster ou se reprovou no threshold bruto.

#### `kinematic_seed_generator.py`
- **Funcionalidade**: Especialista Matemático Cinemático do projeto (`HandKinematicsDirect`). Modela esqueletos perfeitamente a partir do plano matemático sem a contaminação de câmeras.
- **Detalhes Técnicos**:
  - Contém constantes anatômicas universais (Comprimentos metacarpais, Proporções base, Ângulos de espalhamento intrínsecos e estágios radiais em graus de dobra).
  - Função Validadora `is_valid_pose`: Age como os ligamentos colaterais anatômicos. Restringe matrizes proibidas onde espalhamento de dedo acontece com juntas dobradas (bloqueado fisicamente em humanos). Corta abduções hiperbólicas e reduz ruído antes mesmo das matrizes rotacionais serem chamadas.
  - Implementa Cinemática Direta com multiplicação sucessiva de rotações cartesianas localizadas no topo das pontas de cada falange, processando os 10 dígitos da taxonomia DADADADAFP.

#### `pipeline_calibracao_multiagente.py`
- **Funcionalidade**: Coração do sistema de classificação multiagente, roda inteligência heurística para otimizar sementes sem as imperfeições da propagação reversa de Redes Neurais.
- **Detalhes Técnicos**: Orquestra quatro agentes sequenciais:
  - **Agent 1 (Sanitizer)**: Remove "lixo biológico" do dataset ingerido, usando varredura estatística via Z-Scores e cortes por `visibility` de rastreadores da câmera, impedindo mãos quebradas de poluir a base. 
  - **Agent 2 (Spatial Normalizer)**: Centraliza, arrasta pulsos para a origem, constrói eixos x,y,z da base local e cria um vetor invariante gigantesco imune a rotação espacial.
  - **Agent 3 (Dynamic Seed & Tolerance Maker)**: Usa a matemática de aprendizado não-supervisionado (`K-Means k=2`) avaliando o centroide da classe e identificando "sub-classes" espaciais (Ex: Se o sinal B tiver muita variação de Perfil vs Frontal, ele quebra o sinal em dois clusters internos). Mapeia também a tolerância aceitável baseando-se no limite natural (desvio padrão) com as quais o usuário executa suas aberturas.
  - **Agent 4 (Confusion Optimizer)**: Faz `Cross-Validation` contra as próprias sementes. Onde ocorrer `falso positivo`, a heurística identifica a junta que diferencia aquele sinal conflituoso e adiciona "multiplicadores punitivos" apenas àquela articulação naquele grupo de classes. Ex: Se A e S se confundem, a junta do polegar passa a valer 3x o erro. Exporta no final o `seeds_calibradas.json` consolidado.

#### `pose_verifier_live.py`
- **Funcionalidade**: Central de Diagnóstico Médico e de Software do projeto, unindo a visão computacional (Webcam) à geometria pura teórica.
- **Detalhes Técnicos**:
  - Levanta o tracker MediaPipe em tempo real.
  - Renderiza um esqueleto simulado tridimensional na tela ao lado (ou sobreposto por transparência Alpha = 0.5) comparando o físico capturado do usuário contra o teórico da classe (Ex: o usuário digita "3131313111" e aparece como ele de fato deveria estar fazendo).
  - Emite telemetria em milissegundos calculando o `RMSE (Root Mean Square Error)` global da pose e colorindo, em HUD tático, cada junta em Verde (<10%), Amarelo e Vermelho (>25%) para indicar quais articulações do indivíduo estão falhando na conformidade. Permite a emissão do log de telemetria pressionando a tecla `[S]`.

#### `dynamic_sandbox.py`
- **Funcionalidade**: Ambiente interativo Sandbox completo. A praça de testes finais simulando a aplicação no mundo real com a câmera ligada.
- **Detalhes Técnicos**: 
  - Pode carregar os modelos neurais Keras (`modelo_gestos.h5`) ou as pontuações do `CalibratedLibrasClassifier`.
  - Suporta testes de **trajetória no tempo** (*DTW - Dynamic Time Warping*) capturando os 60 frames da interação e detectando similaridades geométricas dinâmicas comparando os vetores da palma da mão e a translação (x,y) pelo peito. 
  - Oferece recursos em hotkeys `[T]` para Teste, `[G]` para gravar sinal novo instantaneamente adicionando metadados na base e `[S]` para gravar em fluxo raw os landmarks para *seeds*. 
  - Computa e plota o boletim estatístico (Forma Base Estática + Similaridade de Trajetória + Orientação) julgando "Aprovado / Reprovado".

---

### 6.3 Pasta: `scripts/` (Ferramentas e Wrappers da Raiz)

Estes scripts funcionam como pontes e utilitários auxiliares focados em atalhos para os executáveis profundos de `Treinamento IA` ou testes rápidos de integridade (DevOps/APIs).

#### `realtime_trainer.py`
- **Funcionalidade**: Capturador rápido e iterativo sem a complexidade visual do *sandbox*. Ferramenta fundamental para coletar *Continuous Learning*.
- **Detalhes Técnicos**: Com a câmera aberta, o usuário aperta a hotkey "R" para iniciar um buffer (array na RAM) que absorve até 60 poses consecutivas (aprox. 2 a 3 segundos). Ao finalizar o lote, pelo console ele solicita a letra correta à qual o sinal pertence e escreve o JSON com essas informações diretamente na pasta customizada `dataset_custom`, finalizando o loop instantaneamente.

#### `testeVM.py`
- **Funcionalidade**: Utilitário de infraestrutura de rede para o ecossistema Back-end do TCC.
- **Detalhes Técnicos**: Utiliza a biblioteca HTTP `requests` enviando pacotes `GET` simples para as rotas base (`/health` e `/signatures/batch`) no endereço da Máquina Virtual da nuvem, verificando a conectividade do banco de dados remoto e possíveis timeouts de API.

#### `visualizador_calibracao.py`
- **Funcionalidade**: Script de ponte/Wrapper (atalho).
- **Detalhes Técnicos**: Modifica o PATH do sistema provisoriamente em tempo de execução via `sys.path.insert` injetando o contexto do `Treinamento IA/scripts` na raíz para ser capaz de evocar e importar o pacote `Calibration3DVisualizer` e lançar o estúdio de inspeção sem causar conflitos de diretório modular.

#### `pose_verifier_live.py` e `kinematic_seed_generator.py` (Versões Raiz)
- **Funcionalidade**: Scripts wrappers para lançar as ferramentas complexas localizadas dentro de `Treinamento IA/scripts/`.
- **Detalhes Técnicos**: Permitem que o desenvolvedor invoque diretamente os especialistas executando `python scripts/nome_do_arquivo.py` sem precisar entrar e manipular referências relativas complicadas ou criar PYTHONPATH fixo, padronizando a chamada através da função `main()`.
