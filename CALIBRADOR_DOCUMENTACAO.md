# Documentação do Hand Calibrator

O **Hand Calibrator** é uma ferramenta de simulação 3D e extração de dados reais via câmera projetada para calibrar limites biomecânicos e capturar poses (Estágios) de mãos humanas para treinamento de Redes Neurais (como arquiteturas DADADADAFP).

A aplicação conta com um painel interativo 3D (mouse drag, sliders) e uma interface de Computer Vision via MediaPipe que espelha os movimentos em tempo real para os metadados da aplicação.

## Funcionalidades Principais

1. **Visualizador 3D Interativo:** Renderização cinemática direta de uma mão. Cada junta pode ser arrastada livremente com o mouse ou controlada precisamente via sliders.
2. **Câmera ao Vivo (MediaPipe):** Extrai a malha de juntas do usuário em tempo real, transcrevendo a pose física diretamente para os valores de Pitch (Flexão) e Yaw (Abertura/Spread) da anatomia do modelo.
3. **Estágios de Fechamento (0 a 3):** Permite calibrar poses chaves progressivas. O "Estágio 0" é a mão aberta e "Estágio 3" é a mão completamente fechada.
4. **Cinemática Diferenciada (Thumb vs Demais Dedos):** O sistema respeita restrições biomecânicas. Dedos D (Indicador, Médio, Anelar e Mínimo) possuem flexão acoplada de tendões (DIP dobra junto com PIP). O Polegar possui 4 eixos de juntas 100% independentes.
5. **Salvamento Parcial Direcionado:** Exportação cirúrgica. É possível sobreescrever no arquivo apenas 1 dedo de 1 estágio, preservando todo o restante já calibrado.

---

## Estrutura de Ingestão do JSON (`calibration_settings.json`)

O arquivo gerado e consumido pela aplicação (localizado em `data/calibration_settings.json`) é o coração dos limites anatômicos. Se você deseja injetar dados biomecânicos externamente, basta formatar um JSON respeitando a hierarquia abaixo. 

Ao iniciar, o `hand_calibrator.py` consome esse arquivo e o aplica instantaneamente ao esqueleto 3D.

### Anatomia das Chaves do JSON

```json
{
    "stages": {
        "Thumb": {
            "0": {
                "J1_Yaw": 15.0,
                "J1_Pitch": 20.0,
                "J2_Yaw": -5.0,
                "J2_Pitch": 10.0,
                "J3_Yaw": 0.0,
                "J3_Pitch": 5.0,
                "J4_Yaw": 0.0,    // Sempre 0 (Restrição da ponta)
                "J4_Pitch": 0.0
            },
            "1": { ... },
            "2": { ... },
            "3": { ... }
        },
        "Index": {
            "0": {
                "J1_Yaw": 5.0,     // Abertura lateral do dedo (Spread)
                "J1_Pitch": 0.0,   // Sempre 0 para Dedos D
                "J2_Yaw": 0.0,     // Sempre 0
                "J2_Pitch": 5.0,   // Flexão da base (MCP)
                "J3_Yaw": 0.0,     // Sempre 0
                "J3_Pitch": 2.0,   // Flexão do meio (PIP)
                "J4_Yaw": 0.0,     // Sempre 0
                "J4_Pitch": 2.0    // Acoplado fisicamente ao J3
            }
        },
        "Middle": { ... },
        "Ring": { ... },
        "Pinky": { ... }
    },
    "avg_lengths": { ... },
    "avg_palm": { ... }
}
```

### Regras de Ouro para Alimentação Externa (Limitações Biomecânicas)

Se você usar uma LLM para popular este JSON, forneça as seguintes regras de restrição da engine física da nossa aplicação:

#### 1. Dedos Padrão (Index, Middle, Ring, Pinky)
- **`J1` (Base/CMC-MCP na palma)**: É o responsável exclusivo pela abertura lateral (Spread) do dedo. Deve possuir valor apenas em `J1_Yaw`. O `J1_Pitch` é ignorado ou forçado a zero, pois a flexão verdadeira começa no J2.
- **`J2` (MCP-PIP)**: É a junta principal de dobrar o dedo na raiz. Deve possuir valor apenas em `J2_Pitch`. O `J2_Yaw` é zero (não torcemos o dedo no meio).
- **`J3` e `J4` (PIP-DIP e DIP-TIP)**: Seguem a mesma regra do J2 (apenas Pitch). Devido ao acoplamento de tendões da mão humana, na maioria das vezes, `J4_Pitch` deve ser igual ou muito próximo de `J3_Pitch`.
- **Acoplamento Anelar-Mínimo**: O dedo Mínimo e Anelar compartilham raízes. Valores de `J2_Pitch` (flexão da base) de um costumam impactar a flexão do outro na vida real.

#### 2. O Polegar (Thumb) - O Dedo Diferente
A base cinemática do Polegar é 100% livre devido à sua articulação selar.
- **Independência Total**: O polegar tem movimento lateral em quase todos os ossos. Portanto, as chaves `J1_Yaw`, `J1_Pitch`, `J2_Yaw`, `J2_Pitch`, `J3_Yaw` e `J3_Pitch` podem (e devem) possuir valores complexos para cruzar a palma da mão.
- **A Ponta (J4)**: A única restrição é o `J4_Yaw` que deve ser zero. A ponta final do polegar apenas flexiona (`J4_Pitch`), ela não dobra para os lados.

#### 3. Estágios (0 a 3)
- **"0"**: Mão esticada com força. (Valores de Pitch negativos ou próximos a zero).
- **"1"**: Mão relaxada/curvada levemente.
- **"2"**: Dedos fechados quase tocando a palma.
- **"3"**: Punho cerrado com tensão. O Polegar neste estágio costuma possuir `J1_Yaw` e `J1_Pitch` elevados para sobrepor/cruzar os dedos Index e Middle.

---

## Formato Dinâmico de Ingestão via Botão (LLM Friendly)

A interface do aplicativo possui um botão **"Ingerir JSON"** que contém um parser super inteligente desenvolvido para facilitar a vida de Modelos de Linguagem (LLMs). 

Ele aceita:
1. **Nomes em Português ou Inglês:** `"indicador"`, `"index"`, `"polegar"`, `"thumb"`, `"mindinho"`, etc.
2. **Nomes Descritivos de Juntas:** `"MCP"`, `"PIP"`, `"DIP"`, `"CMC"`, `"IP"`, `"flexao"`, `"abertura lateral"`, etc.
3. **Formato Direto por Intervalos (Auto-Interpolado):**
   Se você quiser que o sistema calcule os estágios intermediários automaticamente (LERP) do Aberto (Estágio 0) até Fechado (Estágio 3), você pode passar um array com dois valores `[mínimo, máximo]`:

```json
{
    "stages": {
        "indicador": {
            "MCP": [ -5.0, 90.0 ],
            "PIP": [ 0.0, 100.0 ]
        },
        "polegar": {
            "estagio_0": {
                "CMC_Yaw": 50.0,
                "CMC_Pitch": 0.0,
                "MCP_Pitch": 0.0,
                "IP_Pitch": -10.0
            },
            "estagio_3": {
                "CMC_Yaw": -20.0,
                "CMC_Pitch": 53.0,
                "MCP_Pitch": 53.0,
                "IP_Pitch": 53.0
            }
        }
    }
}
```

O próprio aplicativo, ao receber esse JSON, irá quebrar a matriz em juntas exatas (`J1`, `J2`, etc), aplicar os tendões acoplados por baixo dos panos, salvar nativamente no `calibration_settings.json` local e espelhar as mudanças instantaneamente na tela 3D.

---

---

## Classe e Taxonomia do Modelo (DADADADAFP)

O modelo final de classificação utilizará a taxonomia **DADADADAFP**, composta por uma string de 10 dígitos (ex: `0100000000`), sendo cada dígito lido do Mindinho em direção ao Polegar:

1. **[D] Mindinho**: Flexão (Estágios `0` a `3`).
2. **[A] Abertura Mindinho-Anelar**: Spread lateral (`0` = Aberto, `1` = Fechado).
3. **[D] Anelar**: Flexão (Estágios `0` a `3`).
4. **[A] Abertura Anelar-Médio**: Spread lateral (`0` = Aberto, `1` = Fechado).
5. **[D] Médio**: Flexão (Estágios `0` a `3`).
6. **[A] Abertura Médio-Indicador**: Spread lateral (`0` = Aberto, `1` = Fechado).
7. **[D] Indicador**: Flexão (Estágios `0` a `3`).
8. **[A] Abertura Indicador-Polegar**: Spread lateral (`0` = Aberto, `1` = Fechado).
9. **[F] Movimento Transversal (Polegar)**: Posição do polegar em relação à palma (`0` = Aberto/No mesmo plano dos dedos, `1` = Fechado/Dobrado na frente da palma).
10. **[P] Ponta do Polegar (IP)**: Flexão específica da falange distal do polegar (`0` = Aberta, `1` = Fechada).

### Tabela de Valores das Variáveis

| Variável | Significado | Valores Possíveis | Descrição Física |
|----------|-------------|-------------------|------------------|
| **D** | Dedo (Flexão) | `0` | Totalmente Aberto (esticado) |
| | | `1` | Parcialmente Curvado (J3/PIP e J4/DIP flexionados) |
| | | `2` | Totalmente Fechado (Garrado antes de encostar na palma) |
| | | `3` | Punho Fechado (Dedo colado na palma) |
| **A** | Abertura (Spread) | `0` | Dedos Aberto / Separados |
| | | `1` | Dedos Fechados / Unidos |
| **F** | Transversal | `0` | Polegar no mesmo plano da mão (Mão espalmada) |
| | | `1` | Polegar cruzando na frente da palma |
| **P** | Ponta Polegar | `0` | Falange distal Aberta (esticada) |
| | | `1` | Falange distal Fechada (dobrada) |
