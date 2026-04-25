import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, SafeAreaView, TouchableOpacity, StatusBar } from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import VisionProcessor from '../VisionProcessor';

export default function SandboxScreen({ navigation }) {
  const [visionData, setVisionData] = useState({ 
    type: 'status', message: 'Inicializando Módulo...', label: '', confidence: 0
  });
  
  const [facingMode, setFacingMode] = useState('environment');
  const [isRunning, setIsRunning] = useState(false); // Toggle Start/Stop
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
              <Text style={styles.title}>Modo Livre</Text>
              <TouchableOpacity style={styles.switchButton} onPress={toggleCamera}>
                 <Text style={styles.switchText}>Inverter</Text>
              </TouchableOpacity>
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
                    <View style={[styles.glassBox, visionData.confidence > 0.8 && styles.glassBoxSuccess]}>
                        <Text style={styles.labelTitle}>SINAL DETECTADO</Text>
                        <Text style={styles.labelText}>{visionData.label.toUpperCase()}</Text>
                        
                        {visionData.confidence > 0 && (
                            <View style={styles.confidenceBarContainer}>
                                <View style={[styles.confidenceBar, { width: `${(visionData.confidence * 100).toFixed(0)}%` }]} />
                                <Text style={styles.confidenceText}>{(visionData.confidence * 100).toFixed(1)}%</Text>
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
  uiOverlay: { flex: 1, justifyContent: 'space-between', padding: 20, zIndex: 10 },
  
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 20, backgroundColor: 'rgba(15, 23, 26, 0.7)', padding: 15, borderRadius: 15 },
  backBtn: { backgroundColor: '#33464F', padding: 8, borderRadius: 8 },
  backTxt: { color: '#FFF', fontWeight: 'bold' },
  title: { color: '#FFF', fontSize: 18, fontWeight: '800' },
  switchButton: { backgroundColor: 'rgba(255, 255, 255, 0.2)', paddingHorizontal: 15, paddingVertical: 8, borderRadius: 20, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.3)' },
  switchText: { color: '#FFF', fontWeight: '600', fontSize: 12 },
  
  footer: { marginBottom: 30, alignItems: 'center' },
  
  startBtn: { backgroundColor: '#1CB0F6', paddingVertical: 15, paddingHorizontal: 40, borderRadius: 20, marginBottom: 20, borderBottomWidth: 4, borderColor: '#1899D6' },
  stopBtn: { backgroundColor: '#FF4B4B', paddingVertical: 15, paddingHorizontal: 40, borderRadius: 20, marginBottom: 20, borderBottomWidth: 4, borderColor: '#EA2B2B' },
  btnText: { color: '#FFF', fontWeight: 'bold', fontSize: 16 },

  glassBox: { width: '100%', backgroundColor: 'rgba(20, 20, 25, 0.8)', borderRadius: 20, padding: 24, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.15)', alignItems: 'center' },
  glassBoxSuccess: { borderColor: 'rgba(0, 255, 128, 0.5)', backgroundColor: 'rgba(10, 40, 20, 0.8)' },
  alertBox: { backgroundColor: 'rgba(255, 50, 50, 0.9)', padding: 20, borderRadius: 15 },
  alertText: { color: 'white', fontWeight: 'bold' },
  statusText: { color: '#AAA', fontSize: 16, fontWeight: '500' },
  labelTitle: { color: '#00FF80', fontSize: 12, fontWeight: '700', letterSpacing: 2, marginBottom: 8 },
  labelText: { color: '#FFF', fontSize: 36, fontWeight: '900', textAlign: 'center', marginBottom: 15 },
  confidenceBarContainer: { width: '100%', height: 8, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden', position: 'relative', marginTop: 10 },
  confidenceBar: { height: '100%', backgroundColor: '#00FF80', borderRadius: 4 },
  confidenceText: { position: 'absolute', right: 0, top: -20, color: '#00FF80', fontSize: 12, fontWeight: 'bold' }
});
