# Documentação Técnica: Sistema de Reconhecimento de LIBRAS (TCC)

Esta documentação descreve detalhadamente a arquitetura, lógica biomecânica, formulação matemática e funcionalidade de todos os componentes e scripts presentes no repositório. O objetivo é fornecer uma referência técnica exaustiva para manutenção, auditoria ou reprodução completa do pipeline de calibração, geração de sementes, síntese e inteligência artificial.

---

## 1. Visão Geral do Projeto

O projeto baseia-se em um pipeline biomecânico determinístico projetado para superar a limitação de ruído em profundidade (eixo Z) de câmeras monoculares 2D convencionais via MediaPipe Hands:

1. **Calibração Guiada por Visão Dupla (`guided_hand_calibrator.py` & `guided_thumb_calibrator.py`)**: O usuário calibra diretamente pela webcam, recebendo instruções cirúrgicas na tela. Cada postura limite é gravada em dois ângulos consecutivos: **Frontal** (plano coronal XY) e **Lateral 90°** (plano sagital YZ), com detecção de estabilidade temporal (1.2s) e preservação estrita de comprimentos ósseos.
2. **Referencial Canônico da Palma e Motor Cinemático Direto (`kinematic_seed_generator.py`)**: Transforma os dados em coordenadas ortonormais da palma onde o vetor normal $\hat{Z}$ aponta perpendicularmente para a frente do observador.
3. **Poda Biomecânica Anatômica (*Juncturae Tendinum* Pruning - `prune_impossible_classes.py`)**: Elimina combinações articulares fisiologicamente inviáveis na mão humana (ex: dedo Anelar estendido em 0 associado a dedo Mínimo fechado plano contra a palma em 3 ou 4), podando 204 classes impossíveis e estabelecendo um espaço amostral de **2.364 classes anatômicas reais** e redefinindo sinais canônicos como 'W' (`1000000110`).
4. **Data Augmentation Biomecânico (`synthetic_generator.py`)**: Expande as sementes em centenas de milhares de amostras simulando pontos de vista 3D em domo esférico com ruído gaussiano controlado de sensores.
5. **Rede Neural Profunda (`neural_engine.py`)**: Constrói e treina uma Deep Neural Network (DNN) em Keras/TensorFlow e exporta o modelo final nos formatos `.h5` e `.tflite` para integração com o aplicativo mobile (`POC/`) e desktop. Acurácia de validação alcançada: **99.84%**.
6. **Validador Desktop em Tempo Real (`validate_signs_live.py`)**: Validador com suporte a webcam, seleção interativa de alvos, exibição de gabarito visual PiP e motor de feedback biomecânico diferencial dedo a dedo.
7. **Aplicativo Mobile POC (`POC/`)**: Desenvolvido em React Native e Expo SDK 57, permitindo prática guiada e modo sandbox em dispositivos móveis.

---

## 2. Estrutura de Diretórios Limpa e Organizada

```
TCC/
├── POC/                                      # Aplicativo Mobile React Native / Expo SDK 57 (TCC)
│   ├── App.js
│   ├── app.json                              # Manifesto e permissões Android/iOS
│   ├── modelBase64.js                        # Modelo TFLite compilado em Base64
│   ├── labels.js                             # Catálogo das 2.364 classes
│   ├── referenceSeeds.js                     # Projeções 2D dos 21 pontos de referência
│   ├── screens/                              # Telas: ExerciseScreen, SandboxScreen, TrailScreen
│   └── utils/                                # Motor biomecânico e dicionários
│
├── Treinamento IA/                           # Núcleo de Inteligência Artificial e Biomecânica
│   ├── data/
│   │   ├── calibration_captures/             # Snapshots PNG anotados das calibrações
│   │   ├── calibration_settings.json         # Manifesto mestre de calibração
│   │   ├── seeds/
│   │   │   └── seeds.json                    # Catálogo oficial das 2.364 sementes 3D podadas
│   │   └── unified_cache/                    # Cache compacto de 2.364 arquivos .npz
│   │
│   ├── models/                               # Modelos treinados e exportados
│   │   ├── modelo_gestos.h5                  # Modelo Keras hierárquico
│   │   ├── modelo_gestos.tflite              # Modelo compilado TensorFlow Lite (1.9 MB, ~0.6ms)
│   │   └── labels.txt                        # Rótulos das 2.364 classes LIBRAS
│   │
│   ├── reports/                              # Relatórios e documentações de auditoria
│   │   ├── overnight_training.log            # Log persistente de treino por época
│   │   └── training_report.json              # Métricas consolidadas do treinamento
│   │
│   ├── scripts/                              # Scripts executáveis do pipeline de IA
│   │   ├── prune_impossible_classes.py       # Poda cinesiológica de classes impossíveis
│   │   ├── validate_signs_live.py            # Validador desktop interativo via webcam
│   │   ├── kinematic_seed_generator.py       # Motor cinemático direto canônico
│   │   ├── neural_engine.py                  # Treinamento da DNN, validação e compilação TFLite
│   │   ├── synthetic_generator.py            # Motor sintético biomecânico
│   │   └── update_poc.py                     # Sincronizador de modelos com o app mobile
│   │
│   ├── validar_sinais.bat                    # Atalho Windows duplo-clique para validação
│   └── validar_sinais.ps1                    # Atalho PowerShell para validação
```
│   ├── guided_hand_calibrator.py             # Wrapper para o calibrador guiado da mão
│   ├── guided_thumb_calibrator.py            # Wrapper para o calibrador guiado do polegar
│   ├── inspect_seeds.py                      # Wrapper para o inspetor de sementes
│   └── kinematic_seed_generator.py           # Wrapper para o gerador de sementes
│
├── executar_calibrador.bat                   # Atalho Windows para iniciar o calibrador de mão
├── executar_calibrador_polegar.bat           # Atalho Windows para iniciar o calibrador do polegar
├── CALIBRADOR_DOCUMENTACAO.md                # Guia operacional do calibrador guiado
├── DOCUMENTACAO_TECNICA.md                   # Esta documentação arquitetural completa
├── FLUXO_TREINAMENTO_GERACAO.md              # Fluxo conceitual do treinamento sintético
├── relatorio_calibracao_seeds.md             # Relatório analítico da calibração
└── relatorio_treinamento_calibrador.md       # Métricas de treinamento e acurácia
```

---

## 3. Formulação Matemática: Referencial Canônico da Palma

### 3.1 Causa Raiz do Desvio Lateral Histórico
Em gravações monoculares, a mão frequentemente apresentava inclinação de rotação em *yaw* (~68° entre os metacarpos 5 e 17). Quando o produto vetorial tradicional $\vec{Z} = \vec{X} \times \vec{Y}$ era calculado em coordenadas da câmera, o vetor normal $\hat{Z}$ possuía uma componente indesejada massiva no eixo X ($Z_x \approx -0.932$). Consequentemente, ao flexionar os dedos para a frente, as falanges desviavam severamente para a esquerda da tela.

### 3.2 Solução: Sistema Ortonormal Canônico
O algoritmo implementado em `to_canonical_palm_frame(pts)` estabelece uma base ortonormal rigorosa com origem no pulso (Landmark 0):

$$\vec{P}_0 = \text{Landmark 0 (Pulso)}$$

1. **Eixo Longitudinal $\hat{e}_y$**: Aponta do pulso ao metacarpo médio (Landmark 9). Em coordenadas de tela, orienta-se para cima ($-Y$):
   $$\vec{v}_y = \frac{\vec{P}_9 - \vec{P}_0}{\|\vec{P}_9 - \vec{P}_0\|}, \quad \hat{e}_y = -\vec{v}_y$$

2. **Eixo Transversal $\hat{e}_x$**: Aponta do metacarpo indicador (5) ao mínimo (17), ortogonalizado em relação a $\vec{v}_y$ via processo de Gram-Schmidt:
   $$\vec{v}_x^{\text{raw}} = \vec{P}_{17} - \vec{P}_5$$
   $$\vec{v}_x = \vec{v}_x^{\text{raw}} - (\vec{v}_x^{\text{raw}} \cdot \vec{v}_y) \vec{v}_y, \quad \hat{e}_x = \frac{\vec{v}_x}{\|\vec{v}_x\|}$$

3. **Eixo Normal da Palma $\hat{e}_z$**: Perpendicular perfeito à palma da mão, apontando diretamente para o observador (+Z):
   $$\hat{e}_z = \frac{\hat{e}_x \times \hat{e}_y}{\|\hat{e}_x \times \hat{e}_y\|}$$

4. **Matriz de Alinhamento Canônico $R_{\text{canon}}$**:
   $$R_{\text{canon}} = \begin{bmatrix} \hat{e}_x^T \\ \hat{e}_y^T \\ \hat{e}_z^T \end{bmatrix}, \quad \vec{P}_{\text{canon}} = R_{\text{canon}} (\vec{P} - \vec{P}_0)$$

**Garantias Biomecânicas Desta Transformação:**
- O metacarpo médio (9) repousa rigorosamente sobre o eixo longitudinal ($X = 0, Z = 0$).
- Os metacarpos do indicador (5) e do mindinho (17) possuem **exatamente a mesma profundidade $Z$** ($\Delta Z = 0.000000$), eliminando qualquer rotação espúria em *yaw*.
- A flexão sagital dos 4 dedos longos opera com $\hat{Z}_{\text{canon}} = (0, 0, 1)$, mantendo $\Delta X = 0$ em todas as juntas. Os dedos curvam estritamente em frente aos seus respectivos nós metacarpais.

---

## 4. Taxonomia DADADADAFP Simplificada (2.364 Classes Anatômicas)

A classificação adota uma string padronizada de 10 dígitos, lida do Mindinho em direção ao Polegar:

$$\text{Código} = [D_4][A_3][D_3][A_2][D_2][A_1][D_1][A_0][F][P]$$

1. **$[D_4]$ Mindinho (Pinky)**: Flexão (Estágios `0` a `4`).
2. **$[A_3]$ Abertura Mindinho-Anelar**: Spread lateral (`0` = Aberto / Leque, `1` = Fechado / Paralelo).
3. **$[D_3]$ Anelar (Ring)**: Flexão (Estágios `0` a `4`).
4. **$[A_2]$ Abertura Anelar-Médio**: Spread lateral (`0` = Aberto, `1` = Fechado).
5. **$[D_2]$ Médio (Middle)**: Flexão (Estágios `0` a `4`).
6. **$[A_1]$ Abertura Médio-Indicador**: Spread lateral (`0` = Aberto, `1` = Fechado).
7. **$[D_1]$ Indicador (Index)**: Flexão (Estágios `0` a `4`).
8. **$[A_0]$ Abertura Indicador-Polegar**: Abdução radial do polegar (`0` = Aberto esticado, `1` = Junto aos dedos).
9. **$[F]$ Movimento Transversal (Polegar)**: Oposição transversal (`0` = No plano lateral, `1` = Cruzado na frente da palma).
10. **$[P]$ Ponta do Polegar (IP)**: Desconsiderado na simplificação canônica (fixado em `0`).

### 4.1 Estágios de Flexão [D] (Cinemática Sagital Pura)
- **`0` - Estendido**: Dedo 100% reto no plano da mão ($0^\circ, 0^\circ, 0^\circ$).
- **`1` - Curvado / Concha**: Curvatura suave e uniforme em formato de "C" ($25^\circ, 35^\circ, 25^\circ$).
- **`2` - Gancho / Hook**: Base (MCP) reta, falanges distais dobradas em garra ($0^\circ, 85^\circ, 80^\circ$).
- **`3` - Plataforma / Mesa (Tabletop)**: Base (MCP) flexionada a $90^\circ$ para a frente, falanges distais retas ($90^\circ, 0^\circ, 0^\circ$).
- **`4` - Fechado / Punho**: Dedo totalmente dobrado e colado contra a palma ($85^\circ, 95^\circ, 75^\circ$).

### 4.2 Os 3 Estados Fundamentais do Polegar
- **Estado 0: Aberto Esticado (`A0=0, F=0, P=0`)**: Abdução radial máxima no plano da mão aberta.
- **Estado 1: Junto aos Dedos (`A0=1, F=0, P=0`)**: Polegar aduzido encostado ao lado do dedo indicador / palma.
- **Estado 2: Na Transversal (`A0=1, F=1, P=0`)**: Polegar cruzando transversalmente a frente da palma em oposição.

### 4.3 Poda Biomecânica e Limitação Anatômica (*Juncturae Tendinum*)
1. **IP Desconsiderado**: $P = 0$ obrigatório.
2. **Coerência da Oposição**: $A_0 = 0$ e $F = 1$ é anatomicamente impossível (não se pode estar em abdução máxima e na transversal ao mesmo tempo).
3. **Indicador Fechado**: Quando $D_1 \ge 2$, o polegar não pode estar aberto esticado ($A_0 = 1$ forçado).
4. **Travamento de Abertura em Flexão**: Dedos dobrados ($D \ge 2$) não realizam abdução colateral ($A = 1$ forçado).
5. **Restrição Fisiológica do Anelar (*Juncturae Tendinum*)**:
   - Devido às bandas tendíneas interdigitais que interconectam os tendões extensores dos dedos, quando o dedo Anelar está estendido ($D_3 = 0$), o dedo Mínimo é tracionado e **não consegue fechar completamente plano contra a palma** ($D_4 = 3$ ou $D_4 = 4$).
   - A poda anatômica eliminou **204 classes inviáveis** ($D_3 = 0$ e $D_4 \ge 3$).
   - O sinal canônico da letra 'W' foi corrigido para `1000000110` (Mínimo em estágio 1/curvado natural), com tolerância biomecânica permitindo estágios 1 e 2 sem gerar penalidades.
   - Assimetria anatômica respeitada: quando o Mínimo está estendido ($D_4 = 0$), o Anelar **pode** ser fechado ($D_3 = 4$) com o auxílio do polegar travando as falanges, como nos sinais canônicos 'I' (`0141414110`) e 'Y' (`0041414100`).

Total de classes resultantes no dataset: **2.364 sementes anatômicas válidas**.

---

## 5. Documentação dos Scripts do Pipeline

### 5.1 `guided_hand_calibrator.py`
- **Função**: Calibrador guiado em tempo real da mão completa.
- **Estrutura**: Executa 11 passos instrucionais cobrindo baseline da palma, os 5 estágios dos 4 dedos longos em bloco, spreads e posições básicas de polegar.
- **Captura em Duplo Ângulo**: Cada passo exige estabilização de 1.2s no ângulo Frontal e em seguida no Perfil Lateral 90°.
- **Saída**: Grava snapshots anotados em `data/calibration_captures/` e salva o manifesto em `data/calibration_settings.json`.

### 5.2 `guided_thumb_calibrator.py`
- **Função**: Assistente interativo especializado exclusivamente na calibração cirúrgica do polegar.
- **Etapas**:
  1. `thumb_open`: Polegar aberto esticado no plano da mão aberta.
  2. `thumb_closed`: Polegar aduzido colado ao indicador com dedos fechados.
  3. `thumb_transversal`: Polegar em oposição transversal cruzando a palma.
- **Preservação Rígida de Comprimentos Ósseos**: Extrai os comprimentos $L_1, L_2, L_3$ da captura `baseline_open` e projeta os vetores de rotação reais mantendo rigorosamente constantes as distâncias inter-articulares.
- **Sincronização com o Pipeline**: Atualiza `calibration_settings.json` na chave `thumb_extracted` e dispara imediatamente a regeneração de `seeds.json`.

### 5.3 `prune_impossible_classes.py`
- **Função**: Filtro biomecânico estrito que aplica a poda anatômica baseada nas *Juncturae Tendinum*.
- **Ação**: Lê `seeds.json`, elimina as 204 classes onde $D_3 = 0$ e $D_4 \ge 3$, regrava o arquivo limpo com 2.364 classes e remove do diretório `unified_cache/` os arquivos `.npz` órfãos.

### 5.4 `kinematic_seed_generator.py`
- **Função**: Motor de Cinemática Direta Canônica (`HandKinematicsDirect`).
- **Estrutura**: Transforma as bases da palma para o referencial ortonormal canônico, monta os dedos longos com flexão sagital pura e os polegares calibrados, exportando as sementes oficiais para `data/seeds/seeds.json`.

### 5.5 `synthetic_generator.py`
- **Função**: Motor biomecânico gerador de dataset massivo sintético.
- **Operação**: Aplica rotações espaciais 3D em domo esférico sobre cada uma das 2.364 sementes, simulando múltiplos pontos de vista de câmeras reais e adicionando ruído gaussiano calibrado em cada articulação para robustez.

### 5.6 `neural_engine.py`
- **Função**: Pipeline de Deep Learning para treinamento e deploy.
- **Arquitetura**: DNN Sequencial com camadas densas (512 -> 256 -> 128 neurônios), ativações ReLU, normalização em lote (`BatchNormalization`), `Dropout(0.2)` e classificação final via `Softmax` sobre as 2.364 classes.
- **Acurácia e Performance**: Acurácia de validação atingida de **99.84%** (val_loss: 0.0050, checkpoint restaurado da época 64).
- **Deploy**: Exporta `modelo_gestos.h5` e realiza a compilação otimizada para `modelo_gestos.tflite` (1.9 MB, ~0.6ms de latência de inferência).

### 5.7 `update_poc.py`
- **Função**: Sincronizador automatizado entre o ambiente de IA e o aplicativo mobile (`POC/`).
- **Ação**: Converte `modelo_gestos.tflite` para string Base64 em `POC/modelBase64.js`, extrai as 2.364 classes para `POC/labels.js` e gera as coordenadas 2D normalizadas em `POC/referenceSeeds.js` para renderização visual do esqueleto-guia na tela.

### 5.8 `validate_signs_live.py` (Novo Validador Desktop)
- **Função**: Ferramenta de validação interativa em tempo real com webcam, permitindo testar diretamente a eficácia do treinamento e o feedback biomecânico dedo a dedo.
- **Recursos**:
  - Exibição de gabarito visual (PiP - Picture in Picture) com o esqueleto canônico da classe esperada.
  - Reconhecimento com inferência direta via TensorFlow Lite.
  - Motor de feedback biomecânico diferencial: aponta com exatidão se o dedo está muito dobrado, muito esticado ou na postura ideal.
  - Seleção interativa por teclado (`[N]` próxima letra, `[P]` letra anterior, `[L]` alternar letra/código, `[C]` digitar código livre).

---

## 6. Como Operar o Sistema

### 1. Validação Rápida de Sinais LIBRAS (Recomendado)
Para testar os sinais com a webcam e receber feedback biomecânico em tempo real:
```powershell
cd "Treinamento IA"
.\validar_sinais.ps1 --letra W
# ou
python scripts/validate_signs_live.py --letra W
# ou dê duplo clique em: Treinamento IA/validar_sinais.bat
```

### 2. Calibração da Mão Completa
```powershell
python scripts/guided_hand_calibrator.py
# ou duplo clique em: executar_calibrador.bat
```

### 3. Calibração do Polegar
```powershell
python scripts/guided_thumb_calibrator.py
# ou duplo clique em: executar_calibrador_polegar.bat
```

### 4. Poda Anatômica e Sincronização
```powershell
python "Treinamento IA/scripts/prune_impossible_classes.py"
```

### 5. Treinamento da Rede Neural e Atualização da POC Mobile
```powershell
python "Treinamento IA/scripts/neural_engine.py"
python "Treinamento IA/scripts/update_poc.py"
```

### 6. Execução do App Mobile POC (React Native)
```powershell
cd POC
npx expo start --clear
```
