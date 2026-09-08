import React, { useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, TouchableOpacity, StatusBar, ScrollView } from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import VisionProcessor from '../VisionProcessor';
import { 
  getBiomechanicalGuidance, 
  parseHandPose, 
  getClosestLetter, 
  LETTER_KINEMATICS 
} from '../utils/biomechanicalGuide';

const TARGET_LETTERS = Object.keys(LETTER_KINEMATICS);

export default function SandboxScreen({ navigation }) {
  const [visionData, setVisionData] = useState({ 
    type: 'status', message: 'Inicializando Módulo...', label: '', confidence: 0
  });
  
  const [facingMode, setFacingMode] = useState('environment');
  const [isRunning, setIsRunning] = useState(false); // Toggle Start/Stop
  const [selectedTarget, setSelectedTarget] = useState(null); // Alvo opcional para comparar
  const [permission, requestPermission] = useCameraPermissions();

  const handleMessage = (data) => {
    if(!isRunning) return; 
    setVisionData(data);
  };

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'environment' ? 'user' : 'environment');
  };

  const toggleRunning = () => {
    setIsRunning(!isRunning);
    if(isRunning) {
        setVisionData({ type: 'status', message: 'Câmera Pausada.', label: '', confidence: 0 });
    }
  };

  // Computa orientação biomecânica se houver alvo selecionado e predição válida
  const guidance = (visionData.label && selectedTarget)
    ? getBiomechanicalGuidance(visionData.label, selectedTarget)
    : null;

  const parsedPose = visionData.label ? parseHandPose(visionData.label) : null;
  const closestLetter = visionData.label ? getClosestLetter(visionData.label) : null;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {(!permission || !permission.granted) ? (
        <View style={styles.permissionContainer}>
            <Text style={styles.permissionText}>O aplicativo precisa acessar a câmera.</Text>
            <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
                <Text style={styles.permissionButtonText}>Permitir Câmera</Text>
            </TouchableOpacity>
        </View>
      ) : (
        <>
          {/* SÓ RENDERIZA A CAMERA SE ESTIVER RODANDO */}
          <View style={styles.visionContainer}>
              {isRunning && (
                <VisionProcessor 
                    key={facingMode} 
                    facingMode={facingMode} 
                    onHandsDetected={handleMessage} 
                />
              )}
          </View>
    
          <View style={styles.uiOverlay} pointerEvents="box-none">
            
            {/* CABEÇALHO */}
            <View style={styles.header}>
              <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
                 <Text style={styles.backTxt}>Sair</Text>
              </TouchableOpacity>
              <Text style={styles.title}>Modo Livre & Diagnóstico</Text>
              <TouchableOpacity style={styles.switchButton} onPress={toggleCamera}>
                 <Text style={styles.switchText}>Inverter</Text>
              </TouchableOpacity>
            </View>

            {/* SELETOR DE ALVO BIOMECÂNICO PARA TESTE */}
            <View style={styles.targetSelectorWrapper}>
              <Text style={styles.selectorLabel}>Alvo para Testar Instrução:</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.targetScroll}>
                <TouchableOpacity 
                  style={[styles.targetPill, selectedTarget === null && styles.targetPillActive]}
                  onPress={() => setSelectedTarget(null)}
                >
                  <Text style={[styles.targetPillTxt, selectedTarget === null && styles.targetPillTxtActive]}>Nenhum</Text>
                </TouchableOpacity>
                {TARGET_LETTERS.map(letter => (
                  <TouchableOpacity 
                    key={letter}
                    style={[styles.targetPill, selectedTarget === letter && styles.targetPillActive]}
                    onPress={() => setSelectedTarget(selectedTarget === letter ? null : letter)}
                  >
                    <Text style={[styles.targetPillTxt, selectedTarget === letter && styles.targetPillTxtActive]}>{letter}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>

            {/* CONTROLES E RESULTADOS */}
            <View style={styles.footer}>
                
                {/* Botão Start/Stop */}
                <TouchableOpacity style={isRunning ? styles.stopBtn : styles.startBtn} onPress={toggleRunning}>
                    <Text style={styles.btnText}>{isRunning ? 'PAUSAR VALIDADOR' : 'INICIAR VALIDADOR'}</Text>
                </TouchableOpacity>

                {visionData.type === 'error' && (
                    <View style={styles.alertBox}>
                        <Text style={styles.alertText}>Erro: {visionData.message}</Text>
                    </View>
                )}
                
                {visionData.type === 'status' && (
                    <View style={styles.glassBox}>
                        <Text style={styles.statusText}>{visionData.message}</Text>
                    </View>
                )}

                {visionData.type === 'prediction' && (
                    <View style={[styles.glassBox, guidance?.match && styles.glassBoxSuccess]}>
                        
                        {/* BANNER DE ORIENTAÇÃO BIOMECÂNICA AO VIVO (SE HOUVER ALVO) */}
                        {guidance && (
                          <View style={[styles.guidanceBanner, guidance.match ? styles.guidanceBannerSuccess : styles.guidanceBannerNotice]}>
                            <Text style={styles.guidanceAdviceTxt}>{guidance.mainAdvice}</Text>
                            <Text style={styles.guidanceScoreTxt}>Correspondência: {guidance.accuracyScore}%</Text>
                          </View>
                        )}

                        <Text style={styles.labelTitle}>
                          {selectedTarget ? `LIDO VS ALVO (${selectedTarget})` : 'SINAL DETECTADO'}
                        </Text>

                        {/* NOME AMIGÁVEL DA LETRA CORRESPONDENTE OU CÓDIGO */}
                        <Text style={styles.labelText}>
                          {closestLetter?.letter ? `Sinal '${closestLetter.letter}'` : visionData.label.toUpperCase()}
                        </Text>
                        
                        {visionData.confidence > 0 && (
                            <View style={styles.confidenceBarContainer}>
                                <View style={[styles.confidenceBar, { width: `${(visionData.confidence * 100).toFixed(0)}%` }]} />
                                <Text style={styles.confidenceText}>{(visionData.confidence * 100).toFixed(1)}%</Text>
                            </View>
                        )}

                        {/* DIAGNÓSTICO ANATÔMICO DEDO A DEDO */}
                        {parsedPose && (
                          <View style={styles.anatomicalGrid}>
                            <View style={styles.chipRow}>
                              <View style={styles.chip}><Text style={styles.chipTxt}>Mindinho: {parsedPose.pinky.isExtended ? '☝️ Reto' : (parsedPose.pinky.isClosed ? '✊ Dobrado' : '🌙 Curvo')}</Text></View>
                              <View style={styles.chip}><Text style={styles.chipTxt}>Anelar: {parsedPose.ring.isExtended ? '☝️ Reto' : (parsedPose.ring.isClosed ? '✊ Dobrado' : '🌙 Curvo')}</Text></View>
                            </View>
                            <View style={styles.chipRow}>
                              <View style={styles.chip}><Text style={styles.chipTxt}>Médio: {parsedPose.middle.isExtended ? '☝️ Reto' : (parsedPose.middle.isClosed ? '✊ Dobrado' : '🌙 Curvo')}</Text></View>
                              <View style={styles.chip}><Text style={styles.chipTxt}>Indicador: {parsedPose.index.isExtended ? '☝️ Reto' : (parsedPose.index.isClosed ? '✊ Dobrado' : '🌙 Curvo')}</Text></View>
                            </View>
                            <View style={styles.chipRow}>
                              <View style={styles.chip}><Text style={styles.chipTxt}>V/U: {parsedPose.spreads.middleIndex}</Text></View>
                              <View style={styles.chip}><Text style={styles.chipTxt}>Polegar: {parsedPose.thumb.isOpposed ? '✊ Cruzando' : '👈 Aberto'}</Text></View>
                            </View>
                          </View>
                        )}
                    </View>
                )}
            </View>

          </View>
        </>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F171A' },
  permissionContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 30 },
  permissionText: { color: '#FFF', fontSize: 18, textAlign: 'center', marginBottom: 20 },
  permissionButton: { backgroundColor: '#58CC02', padding: 15, borderRadius: 10 },
  permissionButtonText: { color: '#000', fontWeight: 'bold', fontSize: 16 },
  visionContainer: { ...StyleSheet.absoluteFillObject },
  uiOverlay: { flex: 1, justifyContent: 'space-between', padding: 16, zIndex: 10 },
  
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 20, backgroundColor: 'rgba(15, 23, 26, 0.85)', padding: 14, borderRadius: 15, borderWidth: 1, borderColor: '#23343A' },
  backBtn: { backgroundColor: '#33464F', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8 },
  backTxt: { color: '#FFF', fontWeight: 'bold' },
  title: { color: '#FFF', fontSize: 16, fontWeight: '800' },
  switchButton: { backgroundColor: 'rgba(255, 255, 255, 0.2)', paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.3)' },
  switchText: { color: '#FFF', fontWeight: '600', fontSize: 12 },

  targetSelectorWrapper: { backgroundColor: 'rgba(15, 23, 26, 0.85)', borderRadius: 14, padding: 10, borderWidth: 1, borderColor: '#23343A' },
  selectorLabel: { color: '#A0B1B6', fontSize: 12, fontWeight: 'bold', marginBottom: 6 },
  targetScroll: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  targetPill: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12, backgroundColor: '#23343A', borderWidth: 1, borderColor: '#33464F' },
  targetPillActive: { backgroundColor: '#1CB0F6', borderColor: '#FFF' },
  targetPillTxt: { color: '#FFF', fontSize: 13, fontWeight: 'bold' },
  targetPillTxtActive: { color: '#000' },
  
  footer: { marginBottom: 20, alignItems: 'center' },
  
  startBtn: { backgroundColor: '#1CB0F6', paddingVertical: 14, paddingHorizontal: 36, borderRadius: 20, marginBottom: 12, borderBottomWidth: 4, borderColor: '#1899D6' },
  stopBtn: { backgroundColor: '#FF4B4B', paddingVertical: 14, paddingHorizontal: 36, borderRadius: 20, marginBottom: 12, borderBottomWidth: 4, borderColor: '#EA2B2B' },
  btnText: { color: '#FFF', fontWeight: 'bold', fontSize: 15 },

  glassBox: { width: '100%', backgroundColor: 'rgba(20, 20, 25, 0.9)', borderRadius: 18, padding: 16, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.15)', alignItems: 'center' },
  glassBoxSuccess: { borderColor: 'rgba(0, 255, 128, 0.6)', backgroundColor: 'rgba(10, 40, 20, 0.92)' },
  alertBox: { backgroundColor: 'rgba(255, 50, 50, 0.9)', padding: 16, borderRadius: 15 },
  alertText: { color: 'white', fontWeight: 'bold' },
  statusText: { color: '#AAA', fontSize: 15, fontWeight: '500' },
  labelTitle: { color: '#00FF80', fontSize: 11, fontWeight: '700', letterSpacing: 2, marginBottom: 4 },
  labelText: { color: '#FFF', fontSize: 26, fontWeight: '900', textAlign: 'center', marginBottom: 8 },
  confidenceBarContainer: { width: '100%', height: 6, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden', position: 'relative', marginTop: 4, marginBottom: 10 },
  confidenceBar: { height: '100%', backgroundColor: '#00FF80', borderRadius: 4 },
  confidenceText: { position: 'absolute', right: 0, top: -18, color: '#00FF80', fontSize: 11, fontWeight: 'bold' },

  guidanceBanner: { width: '100%', padding: 10, borderRadius: 10, marginBottom: 10, borderWidth: 1 },
  guidanceBannerSuccess: { backgroundColor: 'rgba(0, 100, 40, 0.6)', borderColor: '#00FF80' },
  guidanceBannerNotice: { backgroundColor: 'rgba(255, 150, 0, 0.25)', borderColor: '#FF9600' },
  guidanceAdviceTxt: { color: '#FFF', fontSize: 14, fontWeight: 'bold', textAlign: 'center' },
  guidanceScoreTxt: { color: '#DDD', fontSize: 11, textAlign: 'center', marginTop: 2 },

  anatomicalGrid: { width: '100%', marginTop: 8, borderTopWidth: 1, borderColor: '#33464F', paddingTop: 8 },
  chipRow: { flexDirection: 'row', justifyContent: 'space-between', gap: 6, marginBottom: 4 },
  chip: { flex: 1, backgroundColor: '#0F171A', paddingVertical: 4, paddingHorizontal: 8, borderRadius: 8, borderWidth: 1, borderColor: '#23343A', alignItems: 'center' },
  chipTxt: { color: '#A0B1B6', fontSize: 11, fontWeight: '600' }
});
