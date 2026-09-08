# Guia de Execução do Pipeline de Treinamento em Nova Máquina

Este documento orienta como configurar o ambiente, clonar o projeto e executar o pipeline automatizado de geração de dados sintéticos e treinamento da rede neural (Deep Learning) em outra máquina.

---

## 1. Contexto e Status Atual do Desenvolvimento

### 1.1. Reformulação Biomecânica das Sementes (`seeds.json`)
* As 2.568 sementes cinemáticas canônicas foram atualizadas em [seeds.json](file:///c:/DevTools/Repositories/Faculdade/TCC/Treinamento%20IA/data/seeds/seeds.json) com amplitudes fisiológicas reais de **abdução e adução** (com base na literatura anatômica de Kapandji e Tubiana).
* Agora há separação nítida no espaço 3D entre posturas com dedos fechados e abertos:
  * Indicador (D1): até $-26.0^\circ$ de abertura radial.
  * Médio (D2): até $+10.0^\circ$ de abertura ulnar em sinais como "V".
  * Anelar (D3): até $+20.0^\circ$ de abertura ulnar.
  * Mínimo (D4): até $+40.0^\circ$ de abertura ulnar.
  * Polegar: 4 posições canônicas purificadas com preservação estrita de comprimentos ósseos.

### 1.2. Pipeline Automatizado de 2 Etapas
O script orquestrador [Treinamento IA/scripts/run_overnight_pipeline.py](file:///c:/DevTools/Repositories/Faculdade/TCC/Treinamento%20IA/scripts/run_overnight_pipeline.py) executa em sequência:
1. **Etapa 1 (Geração Massiva)**: O gerador [synthetic_generator.py](file:///c:/DevTools/Repositories/Faculdade/TCC/Treinamento%20IA/scripts/synthetic_generator.py) lê diretamente as sementes e cria 500 amostras sintéticas com variações de rotação 3D e ruído de sensor por classe, salvando arquivos `.npz` de cache unificado em `Treinamento IA/data/unified_cache/` (1.284.000 amostras base em ~60 segundos).
2. **Etapa 2 (Treinamento Neural Engine)**: O motor [neural_engine.py](file:///c:/DevTools/Repositories/Faculdade/TCC/Treinamento%20IA/scripts/neural_engine.py) carrega o cache, aplica espelhamento horizontal (2.568.000 amostras no total), treina a DNN Keras por até 150 épocas (com EarlyStopping e redução de Learning Rate), compila o modelo em `.h5` e `.tflite`, e aciona [update_poc.py](file:///c:/DevTools/Repositories/Faculdade/TCC/Treinamento%20IA/scripts/update_poc.py) para atualizar a POC mobile com o novo modelo em Base64.

---

## 2. Configuração do Ambiente na Nova Máquina

### 2.1. Requisitos de Software
* **Sistema Operacional**: Windows 10/11 ou Linux (Ubuntu 20.04+).
* **Python**: Versão 3.10 ou 3.11 recomendada (64-bit).
* **Git**: Para clonar e sincronizar as alterações.

### 2.2. Clonar o Repositório
No terminal da nova máquina:
```bash
git clone <URL_DO_REPOSITORIO>
cd TCC
git checkout main
git pull origin main
```

### 2.3. Instalação das Dependências Python
Instale os pacotes necessários:
```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib mediapipe
```

> [!TIP]
> **Aceleração por GPU (Opcional, mas Altamente Recomendada):**
> Se a nova máquina possuir placa de vídeo NVIDIA com drivers atualizados e suporte CUDA, o TensorFlow utilizará os núcleos de GPU automaticamente. Isso reduz o tempo de treinamento de horas (em CPU) para poucos minutos.

---

## 3. Cuidados Críticos Antes de Iniciar (Plano de Energia)

> [!WARNING]
> **Desative a Suspensão Automática do Computador!**
> Se o Windows entrar em modo de suspensão / hibernação enquanto o treinamento estiver rodando, a execução do Python será pausada pela CPU.
> 
> **Como ajustar no Windows:**
> 1. Pressione `Win + I` para abrir as **Configurações**.
> 2. Acesse **Sistema** > **Energia e Bateria** (ou *Opções de Energia* no Painel de Controle).
> 3. Na seção **Tela e Suspensão**, defina:
>    * *"Quando conectado, colocar o dispositivo para suspender após"*: **Nunca**.

---

## 4. Como Executar o Pipeline na Nova Máquina

Você pode iniciar o pipeline completo com apenas um comando.

### Opção A: Via Arquivo Batch (Windows - 1 Clique)
Na raiz do repositório `TCC/`, execute com duplo clique ou pelo terminal:
```cmd
executar_pipeline_noturno.bat
```

### Opção B: Via Terminal Python (Windows ou Linux)
Na raiz do repositório `TCC/`:
```bash
python "Treinamento IA/scripts/run_overnight_pipeline.py"
```

---

## 5. Como Acompanhar a Execução

* **No Terminal**: O Keras exibirá a barra de progresso de cada época, informando:
  * `loss` e `accuracy` (dados de treino).
  * `val_loss` e `val_accuracy` (dados de validação com dados inéditos).
  * `ETA` (tempo estimado para cada época).
* **No Arquivo de Log Persistente**:
  Todas as ações com timestamps detalhados são gravadas em:
  `Treinamento IA/reports/overnight_training.log`

---

## 6. Arquivos Finais e Próximos Passos (Etapa 3)

Ao término automático da Etapa 2, os seguintes arquivos estarão atualizados:
* `Treinamento IA/models/modelo_gestos.h5` (Pesos Keras de alta precisão).
* `Treinamento IA/models/modelo_gestos.tflite` (Modelo compilado e otimizado).
* `POC/modelBase64.js` (Modelo codificado para consumo direto no app mobile).
* `POC/labels.js` (Mapeamento oficial de rótulos).

### Etapa 3: Testando os Sinais na Prática
Para testar o reconhecimento ao vivo com sua webcam:
```bash
python "Treinamento IA/scripts/dynamic_sandbox.py"
```
Ou inicie a POC mobile para validar o aplicativo:
```bash
cd POC
npm start
```
