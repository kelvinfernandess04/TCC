import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, SafeAreaView, Dimensions } from 'react-native';
import VisionProcessor from '../VisionProcessor';
import { SignImages } from '../utils/dictionary';
import { getBiomechanicalGuidance, getClosestLetter } from '../utils/biomechanicalGuide';

export default function ExerciseScreen({ route, navigation }) {
  const { lesson } = route.params; 
  // lesson.content = ['A', 'B', 'C']
  
  const [currentLetterIndex, setCurrentLetterIndex] = useState(0);
  const currentLetter = lesson.content[currentLetterIndex];

  const [gameState, setGameState] = useState('IDLE'); // IDLE | COUNTDOWN | RECORDING | RESULT
  const [countdown, setCountdown] = useState(3);
  const [facingMode, setFacingMode] = useState('environment'); 
  
  const [predictions, setPredictions] = useState([]);
  const [resultTitle, setResultTitle] = useState('');
  const [resultMessage, setResultMessage] = useState('');
  const [liveGuidance, setLiveGuidance] = useState(null);
  const [resultGuidance, setResultGuidance] = useState(null);

  const predictionsRef = useRef([]);

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'environment' ? 'user' : 'environment');
  };

  const handleStart = () => {
    console.log("[Exercise] Starting Game... Countdown initiated.");
    setGameState('COUNTDOWN');
    setCountdown(3);
    predictionsRef.current = [];
    setPredictions([]);
    setLiveGuidance(null);
    setResultGuidance(null);
  };

  // Logica de Contagem
  useEffect(() => {
    let timer;
    if (gameState === 'COUNTDOWN') {
      if (countdown > 0) {
        timer = setTimeout(() => setCountdown(c => c - 1), 1000);
      } else {
        setGameState('RECORDING');
      }
    }
    return () => clearTimeout(timer);
  }, [gameState, countdown]);

  // Logica de Gravação (5 segundos)
  useEffect(() => {
    let timer;
    if (gameState === 'RECORDING') {
      timer = setTimeout(() => {
        finalizeValidation();
      }, 5000);
    }
    return () => clearTimeout(timer);
  }, [gameState]);

  const handleMessage = (data) => {
    if (gameState === 'RECORDING' && data.type === 'prediction' && data.confidence > 0) {
      predictionsRef.current.push(data);
      if (data.label) {
        const guidance = getBiomechanicalGuidance(data.label, currentLetter);
        setLiveGuidance(guidance);
      }
    }
  };

  const finalizeValidation = () => {
    const records = predictionsRef.current;
    if (records.length === 0) {
      setResultTitle('Nenhuma mão!');
      setResultMessage('O sistema não encontrou nenhuma mão na tela.');
      setResultGuidance(null);
      setGameState('RESULT');
      return;
    }

    // Achar estatística modal (Qual letra apareceu mais vezes durante os 5 segs)
    let counts = {};
    let confidences = {};
    
    records.forEach(r => {
      counts[r.label] = (counts[r.label] || 0) + 1;
      confidences[r.label] = (confidences[r.label] || 0) + r.confidence;
    });

    const dominantLabel = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    const avgConfidence = confidences[dominantLabel] / counts[dominantLabel];
    const finalGuidance = getBiomechanicalGuidance(dominantLabel, currentLetter);
    setResultGuidance(finalGuidance);

    const closest = getClosestLetter(dominantLabel);
    const detectedName = closest.letter ? `Sinal '${closest.letter}'` : dominantLabel;

    if (finalGuidance.match || dominantLabel.toUpperCase() === currentLetter.toUpperCase()) {
       setResultTitle('Incrível!! 🎉');
       setResultMessage(`Você executou o sinal com ${(avgConfidence*100).toFixed(1)}% de precisão e excelente postura!`);
    } else {
       setResultTitle('Ops, Tente Novamente! ❌');
       setResultMessage(`Detectamos maior semelhança com ${detectedName} do que com a letra '${currentLetter}'.`);
    }

    setGameState('RESULT');
  };

  const handleNext = () => {
    if (resultTitle.includes('Incrível')) {
       // Avança pra proxima
       if (currentLetterIndex + 1 < lesson.content.length) {
          setCurrentLetterIndex(c => c + 1);
          setGameState('IDLE');
       } else {
          // Fim da Fase
          navigation.goBack();
       }
    } else {
       // Errou, volta pro tentar de novo
       setGameState('IDLE');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      
      {/* HEADER ESCURO */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Text style={styles.backTxt}>Desistir</Text>
        </TouchableOpacity>
        <Text style={styles.progressTxt}>Exercício {currentLetterIndex + 1} de {lesson.content.length}</Text>
      </View>

      {/* CORE AREA */}
      <View style={styles.coreArea}>
        
        {/* TELA DE IDLE (Mostrar o alvo e deixar dar start) */}
        {gameState === 'IDLE' && (
          <View style={styles.idlePanel}>
             <Text style={styles.instructionText}>Faça o sinal para a letra:</Text>
             <Text style={styles.bigLetter}>{currentLetter}</Text>
             <Image source={SignImages[currentLetter]} style={styles.referenceImage} />
             
             <View style={styles.camToggleRow}>
                <Text style={styles.camLabel}>Usando: {facingMode === 'environment' ? 'Traseira' : 'Frontal'}</Text>
                <TouchableOpacity style={styles.toggleBtn} onPress={toggleCamera}>
                   <Text style={styles.toggleBtnTxt}>Inverter Câmera</Text>
                </TouchableOpacity>
             </View>

             <TouchableOpacity style={styles.playButton} onPress={handleStart}>
                <Text style={styles.playButtonText}>COMEÇAR</Text>
             </TouchableOpacity>
          </View>
        )}

        {/* CÂMERA E FEEDBACKS */}
        {(gameState === 'COUNTDOWN' || gameState === 'RECORDING') && (
          <View style={styles.cameraBox}>
             <VisionProcessor 
                  key={facingMode} 
                  facingMode={facingMode} 
                  onHandsDetected={handleMessage} 
             />
             
             {/* Overlay Countdown */}
             {gameState === 'COUNTDOWN' && (
                <View style={styles.overlayCenter}>
                   <Text style={styles.countdownTitle}>Prepare-se!</Text>
                   <Text style={styles.countdownNumber}>{countdown}</Text>
                </View>
             )}
             
             {/* Overlay Gravação com Feedback Biomecânico */}
             {gameState === 'RECORDING' && (
                <View style={styles.overlayTop}>
                   <View style={[
                      styles.recordingBadge,
                      liveGuidance?.match ? styles.badgeSuccess : (liveGuidance ? styles.badgeFeedback : null)
                   ]}>
                      <View style={[styles.redDot, liveGuidance?.match && styles.greenDot]} />
                      <Text style={styles.recordingText}>
                        {liveGuidance ? liveGuidance.mainAdvice : 'Lendo Gesto...'}
                      </Text>
                   </View>
                   {liveGuidance && !liveGuidance.match && liveGuidance.hints.length > 1 && (
                      <View style={styles.secondaryAdviceCard}>
                         <Text style={styles.secondaryAdviceTxt}>{liveGuidance.hints[1]}</Text>
                      </View>
                   )}
                </View>
             )}
          </View>
        )}

         {/* RESULTADO (POPUP POR CIMA) */}
         {gameState === 'RESULT' && (
          <View style={styles.idlePanel}>
             <Text style={styles.instructionText}>{resultTitle}</Text>
             <Text style={styles.resultDescText}>{resultMessage}</Text>
             
             {/* Guia Didático de Correção Anatômica */}
             {resultGuidance && !resultGuidance.match && (
               <View style={styles.guidanceBox}>
                 <Text style={styles.guidanceTitle}>💡 Dicas de postura para '{currentLetter}':</Text>
                 {resultGuidance.hints.map((hint, idx) => (
                   <View key={idx} style={styles.hintItem}>
                     <Text style={styles.hintText}>{hint}</Text>
                   </View>
                 ))}
               </View>
             )}
             
             <TouchableOpacity style={styles.playButton} onPress={handleNext}>
                <Text style={styles.playButtonText}>{resultTitle.includes('Incrível') ? 'CONTINUAR' : 'TENTAR DE NOVO'}</Text>
             </TouchableOpacity>
          </View>
        )}

      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#131F24' },
  header: { flexDirection: 'row', alignItems: 'center', padding: 20, paddingTop: 40, borderBottomWidth: 1, borderColor: '#23343A' },
  backBtn: { marginRight: 20 },
  backTxt: { color: '#FF4B4B', fontWeight: 'bold', fontSize: 16 },
  progressTxt: { color: '#FFF', fontSize: 16, fontWeight: 'bold' },
  coreArea: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  
  idlePanel: { width: '100%', backgroundColor: '#1A2A30', borderRadius: 20, padding: 30, alignItems: 'center', borderWidth: 2, borderColor: '#23343A' },
  instructionText: { color: '#A0B1B6', fontSize: 20, fontWeight: 'bold', textAlign: 'center', marginBottom: 10 },
  resultDescText: { color: '#FFF', fontSize: 18, textAlign: 'center', marginBottom: 30, marginTop: 10 },
  bigLetter: { color: '#1CB0F6', fontSize: 72, fontWeight: '900', marginBottom: 20 },
  referenceImage: { width: 120, height: 120, borderRadius: 15, marginBottom: 30 },
  
  camToggleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', width: '100%', marginBottom: 30, backgroundColor: '#0F171A', padding: 15, borderRadius: 12 },
  camLabel: { color: '#FFF', fontSize: 14, fontWeight: 'bold' },
  toggleBtn: { backgroundColor: '#23343A', padding: 8, borderRadius: 8 },
  toggleBtnTxt: { color: '#FFF', fontSize: 12, fontWeight: 'bold' },
  
  playButton: { backgroundColor: '#58CC02', paddingVertical: 18, width: '100%', borderRadius: 16, alignItems: 'center', borderBottomWidth: 5, borderColor: '#58A700' },
  playButtonText: { color: '#FFF', fontSize: 20, fontWeight: '900', letterSpacing: 1 },

  cameraBox: { width: Dimensions.get('window').width * 0.9, height: Dimensions.get('window').height * 0.6, borderRadius: 30, overflow: 'hidden', borderWidth: 4, borderColor: '#1CB0F6' },
  overlayCenter: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center' },
  countdownTitle: { color: '#FFF', fontSize: 30, fontWeight: 'bold', marginBottom: 10 },
  countdownNumber: { color: '#1CB0F6', fontSize: 100, fontWeight: '900' },
  
  overlayTop: { position: 'absolute', top: 20, left: 10, right: 10, alignItems: 'center' },
  recordingBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.85)', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 20, borderWidth: 1, borderColor: '#33464F', maxWidth: '95%' },
  badgeSuccess: { borderColor: '#58CC02', backgroundColor: 'rgba(20, 60, 20, 0.9)' },
  badgeFeedback: { borderColor: '#1CB0F6', backgroundColor: 'rgba(15, 30, 45, 0.9)' },
  redDot: { width: 12, height: 12, borderRadius: 6, backgroundColor: '#FF4B4B', marginRight: 10 },
  greenDot: { backgroundColor: '#58CC02' },
  recordingText: { color: '#FFF', fontSize: 15, fontWeight: 'bold', textAlign: 'center' },
  secondaryAdviceCard: { marginTop: 8, backgroundColor: 'rgba(0,0,0,0.75)', paddingHorizontal: 14, paddingVertical: 6, borderRadius: 12, borderWidth: 1, borderColor: '#FF9600' },
  secondaryAdviceTxt: { color: '#FFD900', fontSize: 13, fontWeight: '600' },

  guidanceBox: { width: '100%', backgroundColor: '#0F171A', borderRadius: 14, padding: 16, marginBottom: 20, borderWidth: 1, borderColor: '#23343A' },
  guidanceTitle: { color: '#1CB0F6', fontSize: 16, fontWeight: 'bold', marginBottom: 10 },
  hintItem: { paddingVertical: 6, borderBottomWidth: 1, borderColor: '#1A2A30' },
  hintText: { color: '#E5E5E5', fontSize: 14, fontWeight: '500', lineHeight: 20 }
});
