/**
 * Biomechanical Guide & Feedback Engine (TCC - LIBRAS)
 * ====================================================
 * Analisa e compara a postura da mão lida pela IA com a postura esperada,
 * gerando correções biomecânicas em tempo real (dedo a dedo e abertura).
 */

// Códigos canônicos das letras de LIBRAS (DADADADAFP)
// D4=Mindinho, A3=Spread(Min-Ane), D3=Anelar, A2=Spread(Ane-Med),
// D2=Médio, A1=Spread(Med-Ind), D1=Indicador, A0=Spread(Ind-Pol), F=Oposição Pol, P=Ponta Pol
export const LETTER_KINEMATICS = {
  'A': { code: '4141414110', name: "Sinal 'A'", description: "Punho fechado com polegar apoiado na lateral" },
  'B': { code: '0101010110', name: "Sinal 'B'", description: "4 dedos erguidos juntos, polegar dobrado na frente da palma" },
  'C': { code: '1010101000', name: "Sinal 'C'", description: "Todos os dedos curvados em formato de arco/concha" },
  'D': { code: '4141410110', name: "Sinal 'D'", description: "Indicador erguido, outros 3 dedos fechados tocando o polegar" },
  'E': { code: '2121212110', name: "Sinal 'E'", description: "Dedos em garra recolhida com pontas sobre o polegar" },
  'I': { code: '0141414110', name: "Sinal 'I'", description: "Apenas o mindinho erguido, outros dedos fechados" },
  'L': { code: '4141410000', name: "Sinal 'L'", description: "Indicador reto e polegar aberto em 90°" },
  'M': { code: '3131314110', name: "Sinal 'M'", description: "Indicador, médio e anelar dobrados para baixo sobre o polegar" },
  'N': { code: '4131314110', name: "Sinal 'N'", description: "Indicador e médio dobrados para baixo sobre o polegar" },
  'O': { code: '1010101010', name: "Sinal 'O'", description: "Dedos curvados formando um círculo com o polegar" },
  'R': { code: '4141010110', name: "Sinal 'R'", description: "Indicador e médio cruzados, outros dedos fechados" },
  'S': { code: '4141414110', name: "Sinal 'S'", description: "Punho cerrado com polegar cruzando a frente dos dedos" },
  'U': { code: '4141010110', name: "Sinal 'U'", description: "Indicador e médio estendidos retos e bem juntos" },
  'V': { code: '4141000110', name: "Sinal 'V'", description: "Indicador e médio estendidos e abertos em 'V'" },
  'W': { code: '1000000110', name: "Sinal 'W'", description: "Indicador, médio e anelar estendidos em 'W', mindinho apoiado" },
  'X': { code: '4141412110', name: "Sinal 'X'", description: "Indicador em gancho (dobrado na ponta), outros fechados" },
  'Y': { code: '0041414100', name: "Sinal 'Y'", description: "Polegar e mindinho estendidos abertos, dedos do meio fechados" }
};

const STAGE_NAMES = {
  0: 'Estendido (Reto)',
  1: 'Curvado (Concha)',
  2: 'Gancho (Hook)',
  3: 'Plataforma (Tabletop)',
  4: 'Fechado (Palma)'
};

/**
 * Normaliza qualquer formato de entrada (Letra, 10 dígitos ou formato legado S_XXXXX_Y)
 * para um código cinemático padronizado de 10 caracteres numéricos.
 */
export function resolveKinematicCode(input) {
  if (!input) return '0000000000';
  const clean = String(input).trim().toUpperCase();

  // 1. Se for uma letra direta (ex: 'A', 'V')
  if (LETTER_KINEMATICS[clean]) {
    return LETTER_KINEMATICS[clean].code;
  }

  // 2. Se for formato classe_X (ex: 'classe_A')
  const letterMatch = clean.match(/^CLASSE_([A-Z])$/);
  if (letterMatch && LETTER_KINEMATICS[letterMatch[1]]) {
    return LETTER_KINEMATICS[letterMatch[1]].code;
  }

  // 3. Se for código de 10 dígitos puro (ex: '4141000110')
  if (/^\d{10}$/.test(clean)) {
    return clean;
  }

  // 4. Se for formato legado S_XXXXX_Y (ex: 'S_00120_1')
  if (clean.startsWith('S_')) {
    const parts = clean.split('_');
    if (parts.length >= 2 && parts[1].length === 5) {
      const p = parts[1];
      // mapeamento anatômico de 5 dígitos para 10 dígitos:
      // p[0]=Thumb, p[1]=Index, p[2]=Middle, p[3]=Ring, p[4]=Pinky
      const d4 = p[4] || '0';
      const d3 = p[3] || '0';
      const d2 = p[2] || '0';
      const d1 = p[1] || '0';
      const thumbFlex = p[0] || '0';
      const f = thumbFlex >= '2' ? '1' : '0';
      const spreadY = parts[2] || '1';
      const a1 = spreadY === '0' ? '0' : '1';
      return `${d4}1${d3}1${d2}${a1}${d1}1${f}0`;
    }
  }

  return '0000000000';
}

/**
 * Encontra a letra do alfabeto de LIBRAS canônica mais próxima de um código cinemático.
 */
export function getClosestLetter(input) {
  const code = resolveKinematicCode(input);
  let bestLetter = null;
  let bestDist = Infinity;

  for (const [letter, data] of Object.entries(LETTER_KINEMATICS)) {
    let dist = 0;
    for (let i = 0; i < 10; i++) {
      dist += Math.abs(parseInt(code[i] || '0', 10) - parseInt(data.code[i] || '0', 10));
    }
    if (dist < bestDist) {
      bestDist = dist;
      bestLetter = letter;
    }
  }

  return {
    letter: bestLetter,
    distance: bestDist,
    isExact: bestDist === 0,
    info: LETTER_KINEMATICS[bestLetter]
  };
}

/**
 * Decompõe um código cinemático em um objeto anatômico legível.
 */
export function parseHandPose(input) {
  const code = resolveKinematicCode(input);
  const d4 = parseInt(code[0], 10) || 0; // Mindinho
  const a3 = parseInt(code[1], 10) || 0; // Spread Min-Ane
  const d3 = parseInt(code[2], 10) || 0; // Anelar
  const a2 = parseInt(code[3], 10) || 0; // Spread Ane-Med
  const d2 = parseInt(code[4], 10) || 0; // Médio
  const a1 = parseInt(code[5], 10) || 0; // Spread Med-Ind
  const d1 = parseInt(code[6], 10) || 0; // Indicador
  const a0 = parseInt(code[7], 10) || 0; // Spread Ind-Pol
  const f  = parseInt(code[8], 10) || 0; // Polegar Oposição
  const p  = parseInt(code[9], 10) || 0; // Polegar Ponta

  return {
    rawCode: code,
    pinky:  { stage: d4, name: STAGE_NAMES[d4] || 'Desconhecido', isExtended: d4 === 0, isClosed: d4 >= 3 },
    ring:   { stage: d3, name: STAGE_NAMES[d3] || 'Desconhecido', isExtended: d3 === 0, isClosed: d3 >= 3 },
    middle: { stage: d2, name: STAGE_NAMES[d2] || 'Desconhecido', isExtended: d2 === 0, isClosed: d2 >= 3 },
    index:  { stage: d1, name: STAGE_NAMES[d1] || 'Desconhecido', isExtended: d1 === 0, isClosed: d1 >= 3 },
    thumb: {
      isOpposed: f === 1,
      isSpreadOpen: a0 === 0,
      isTipFolded: p === 1,
      description: (a0 === 0 && f === 0) ? 'Aberto Lateralmente (90°)' : (f === 1 ? 'Cruzando a Palma' : 'Apoiado nos Dedos')
    },
    spreads: {
      pinkyRing: a3 === 0 ? 'Aberto' : 'Junto',
      ringMiddle: a2 === 0 ? 'Aberto' : 'Junto',
      middleIndex: a1 === 0 ? 'Aberto (V)' : 'Junto (U)'
    }
  };
}

/**
 * Compara a pose lida pela IA com a pose esperada (alvo do exercício),
 * identificando as divergências dedo a dedo e fornecendo instruções claras.
 */
export function getBiomechanicalGuidance(detectedInput, expectedInput) {
  if (!detectedInput || !expectedInput) {
    return {
      match: false,
      accuracyScore: 0,
      mainAdvice: 'Posicione sua mão em frente à câmera.',
      hints: ['Aguardando leitura estável da mão...'],
      fingerStatus: {}
    };
  }

  const detected = parseHandPose(detectedInput);
  const expected = parseHandPose(expectedInput);

  const hints = [];
  const fingerStatus = {
    index: 'OK',
    middle: 'OK',
    ring: 'OK',
    pinky: 'OK',
    thumb: 'OK',
    spread: 'OK'
  };

  let totalDiffs = 0;

  // 1. ANÁLISE DO DEDO INDICADOR
  if (expected.index.stage !== detected.index.stage) {
    totalDiffs++;
    if (expected.index.isExtended && !detected.index.isExtended) {
      hints.push('☝️ Estique o dedo Indicador totalmente para cima.');
      fingerStatus.index = 'ERR_NEED_EXTEND';
    } else if (expected.index.isClosed && !detected.index.isClosed) {
      hints.push('👇 Dobre o dedo Indicador para a palma.');
      fingerStatus.index = 'ERR_NEED_FOLD';
    } else if (expected.index.stage === 1) {
      hints.push('🌙 Curve o Indicador suavemente em formato de arco.');
      fingerStatus.index = 'ERR_ADJUST';
    } else if (expected.index.stage === 2) {
      hints.push('🪝 Dobre o Indicador em formato de gancho/anzol.');
      fingerStatus.index = 'ERR_ADJUST';
    }
  }

  // 2. ANÁLISE DO DEDO MÉDIO
  if (expected.middle.stage !== detected.middle.stage) {
    totalDiffs++;
    if (expected.middle.isExtended && !detected.middle.isExtended) {
      hints.push('🖕 Estique o dedo Médio para cima.');
      fingerStatus.middle = 'ERR_NEED_EXTEND';
    } else if (expected.middle.isClosed && !detected.middle.isClosed) {
      hints.push('👇 Dobre o dedo Médio para a palma.');
      fingerStatus.middle = 'ERR_NEED_FOLD';
    } else if (expected.middle.stage === 1) {
      hints.push('🌙 Curve o dedo Médio em arco.');
      fingerStatus.middle = 'ERR_ADJUST';
    }
  }

  // 3. ANÁLISE DO DEDO ANELAR
  if (expected.ring.stage !== detected.ring.stage) {
    totalDiffs++;
    if (expected.ring.isExtended && !detected.ring.isExtended) {
      hints.push('☝️ Estique o dedo Anelar.');
      fingerStatus.ring = 'ERR_NEED_EXTEND';
    } else if (expected.ring.isClosed && !detected.ring.isClosed) {
      hints.push('👇 Dobre o dedo Anelar para baixo.');
      fingerStatus.ring = 'ERR_NEED_FOLD';
    }
  }

  // 4. ANÁLISE DO DEDO MÍNIMO (MINDINHO)
  // Tolerância anatômica: quando Anelar, Médio e Indicador estão estendidos (como no sinal W),
  // o mindinho não fecha a 100% por tração tendínea (juncturae tendinum). Estágios 1 e 2 são válidos.
  const isWPosture = expected.ring.isExtended && expected.middle.isExtended && expected.index.isExtended;
  if (isWPosture && (detected.pinky.stage === 1 || detected.pinky.stage === 2) && (expected.pinky.stage === 1 || expected.pinky.stage === 2)) {
    // Tolerância anatômica confirmada
  } else if (expected.pinky.stage !== detected.pinky.stage) {
    totalDiffs++;
    if (expected.pinky.isExtended && !detected.pinky.isExtended) {
      hints.push('🤙 Estique o dedo Mindinho para cima.');
      fingerStatus.pinky = 'ERR_NEED_EXTEND';
    } else if (expected.pinky.isClosed && !detected.pinky.isClosed) {
      hints.push('👇 Dobre o dedo Mindinho para a palma.');
      fingerStatus.pinky = 'ERR_NEED_FOLD';
    }
  }

  // 5. ANÁLISE DO POLEGAR
  const expectedThumbSpread = expected.thumb.isSpreadOpen;
  const detectedThumbSpread = detected.thumb.isSpreadOpen;
  const expectedThumbOpp = expected.thumb.isOpposed;
  const detectedThumbOpp = detected.thumb.isOpposed;

  if (expectedThumbSpread !== detectedThumbSpread || expectedThumbOpp !== detectedThumbOpp) {
    totalDiffs++;
    if (expectedThumbSpread && !detectedThumbSpread) {
      hints.push('👈 Abra o Polegar esticado para o lado (formando 90°).');
      fingerStatus.thumb = 'ERR_NEED_OPEN';
    } else if (!expectedThumbSpread && detectedThumbSpread) {
      hints.push('👉 Aproxime o Polegar dos outros dedos.');
      fingerStatus.thumb = 'ERR_NEED_CLOSE';
    } else if (expectedThumbOpp && !detectedThumbOpp) {
      hints.push('✊ Dobre o Polegar cruzando a frente dos dedos.');
      fingerStatus.thumb = 'ERR_NEED_CROSS';
    }
  }

  // 6. ANÁLISE DO ESPALHAMENTO ENTRE INDICADOR E MÉDIO (V vs U)
  const expMedIndSpread = expected.spreads.middleIndex;
  const detMedIndSpread = detected.spreads.middleIndex;
  if (expected.index.isExtended && expected.middle.isExtended) {
    if (expMedIndSpread.startsWith('Aberto') && detMedIndSpread.startsWith('Junto')) {
      hints.push('✌️ Separe/abra o Indicador e o Médio (em formato de V).');
      fingerStatus.spread = 'ERR_NEED_OPEN';
      totalDiffs++;
    } else if (expMedIndSpread.startsWith('Junto') && detMedIndSpread.startsWith('Aberto')) {
      hints.push('🤞 Junte o Indicador e o Médio bem colados.');
      fingerStatus.spread = 'ERR_NEED_CLOSE';
      totalDiffs++;
    }
  }

  const isMatch = totalDiffs === 0;
  const accuracyScore = Math.max(0, 100 - (totalDiffs * 18));

  let mainAdvice = 'Perfeito! Mantenha essa posição.';
  if (!isMatch) {
    mainAdvice = hints[0] || 'Ajuste os dedos para o sinal correto.';
  }

  return {
    match: isMatch,
    accuracyScore,
    totalDiffs,
    mainAdvice,
    hints: hints.length > 0 ? hints : ['Sinal alinhado perfeitamente!'],
    detectedPose: detected,
    expectedPose: expected,
    fingerStatus
  };
}

export const POPULAR_CLASSES = [
  { code: '4141000110', letter: 'V', name: '4141000110 (Letra V)', desc: 'Indicador e Médio abertos em V' },
  { code: '4141010110', letter: 'U', name: '4141010110 (Letra U/R)', desc: 'Indicador e Médio estendidos juntos' },
  { code: '0101010110', letter: 'B', name: '0101010110 (Letra B)', desc: '4 dedos erguidos juntos' },
  { code: '4141414110', letter: 'A', name: '4141414110 (Letra A/S)', desc: 'Punho fechado com polegar' },
  { code: '4141410000', letter: 'L', name: '4141410000 (Letra L)', desc: 'Indicador e polegar abertos em 90°' },
  { code: '4141410110', letter: 'D', name: '4141410110 (Letra D)', desc: 'Indicador erguido, outros fechados' },
  { code: '0141414110', letter: 'I', name: '0141414110 (Letra I)', desc: 'Mindinho erguido, outros fechados' },
  { code: '1010101000', letter: 'C', name: '1010101000 (Letra C)', desc: 'Dedos em concha / arco C' },
  { code: '2121212110', letter: 'E', name: '2121212110 (Letra E)', desc: 'Dedos em garra recolhida' },
  { code: '3131314110', letter: 'M', name: '3131314110 (Letra M)', desc: '3 dedos dobrados sobre polegar' },
  { code: '4131314110', letter: 'N', name: '4131314110 (Letra N)', desc: '2 dedos dobrados sobre polegar' },
  { code: '1010101010', letter: 'O', name: '1010101010 (Letra O)', desc: 'Dedos em círculo com polegar' },
  { code: '1000000110', letter: 'W', name: '1000000110 (Letra W)', desc: '3 dedos estendidos em W, mindinho apoiado' },
  { code: '4141412110', letter: 'X', name: '4141412110 (Letra X)', desc: 'Indicador em gancho' },
  { code: '0041414100', letter: 'Y', name: '0041414100 (Letra Y)', desc: 'Polegar e mindinho abertos' },
  { code: '0000000000', letter: null, name: '0000000000 (Mão Aberta)', desc: 'Todos os dedos abertos em leque' }
];
