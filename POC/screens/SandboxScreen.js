import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  SafeAreaView, 
  TouchableOpacity, 
  StatusBar, 
  ScrollView, 
  Image, 
  Modal, 
  TextInput 
} from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import VisionProcessor from '../VisionProcessor';
import { referenceSeeds } from '../referenceSeeds';
import { SignImages } from '../utils/dictionary';
import { 
  getBiomechanicalGuidance, 
  parseHandPose, 
  getClosestLetter, 
  POPULAR_CLASSES 
} from '../utils/biomechanicalGuide';
import { requestAppCameraPermission } from '../utils/cameraPermission';

export default function SandboxScreen({ navigation }) {
  const [visionData, setVisionData] = useState({ 
    type: 'status', message: 'Iniciando câmera...', label: '', confidence: 0
  });
  
  const [facingMode, setFacingMode] = useState('environment');
  const [isRunning, setIsRunning] = useState(true); // Câmera inicia ativa automaticamente
  const [selectedTargetClass, setSelectedTargetClass] = useState('4141000110'); // Default: Letra V
  const [modalVisible, setModalVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [permission, requestPermission] = useCameraPermissions();

  // Solicita permissões nativas de câmera automaticamente ao abrir a tela
  React.useEffect(() => {
    (async () => {
      if (!permission || !permission.granted) {
        await requestAppCameraPermission();
      }
    })();
  }, []);

  const handleRequestPermission = async () => {
    await requestAppCameraPermission();
    await requestPermission();
  };

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

  // Resolução da classe alvo esperada
  const targetParsed = selectedTargetClass ? parseHandPose(selectedTargetClass) : null;
  const targetClosest = selectedTargetClass ? getClosestLetter(selectedTargetClass) : null;
  const targetImage = (targetClosest?.letter && SignImages[targetClosest.letter]) ? SignImages[targetClosest.letter] : null;
  const targetPoints = selectedTargetClass ? referenceSeeds[selectedTargetClass] : null;

  // Comparação biomecânica entre classe lida e esperada
  const guidance = (visionData.label && selectedTargetClass && visionData.confidence > 0)
    ? getBiomechanicalGuidance(visionData.label, selectedTargetClass)
    : null;

  const detectedParsed = visionData.label ? parseHandPose(visionData.label) : null;
  const detectedClosest = visionData.label ? getClosestLetter(visionData.label) : null;

  // Filtragem de classes no modal
  const filteredPopular = POPULAR_CLASSES.filter(c => 
    c.code.includes(searchQuery) || 
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.desc.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {(!permission || !permission.granted) ? (
        <View style={styles.permissionContainer}>
            <Text style={styles.permissionTitle}>📷 Acesso à Câmera Necessário</Text>
            <Text style={styles.permissionText}>O aplicativo precisa acessar a câmera do seu celular para reconhecer seus sinais em LIBRAS em tempo real.</Text>
            <TouchableOpacity style={styles.permissionButton} onPress={handleRequestPermission}>
                <Text style={styles.permissionButtonText}>Permitir Acesso à Câmera</Text>
            </TouchableOpacity>
        </View>
      ) : (
        <>
          {/* CÂMERA COM VISUALIZAÇÃO E ESQUELETO DIRETO NO CANVAS */}
          <View style={styles.visionContainer}>
              {isRunning && (
                <VisionProcessor 
                    key={facingMode} 
                    facingMode={facingMode} 
                    targetPoints={targetPoints}
                    targetLabel={selectedTargetClass ? (targetClosest?.letter ? `Sinal ${targetClosest.letter}` : selectedTargetClass) : ''}
                    onHandsDetected={handleMessage} 
                />
              )}
          </View>
    
          {/* OVERLAY DE INTERFACE DIDÁTICA */}
          <View style={styles.uiOverlay} pointerEvents="box-none">
            
            {/* CABEÇALHO */}
            <View style={styles.header}>
              <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
                 <Text style={styles.backTxt}>Voltar</Text>
              </TouchableOpacity>
              <Text style={styles.title}>Modo Livre & Diagnóstico</Text>
              <TouchableOpacity style={styles.switchButton} onPress={toggleCamera}>
                 <Text style={styles.switchText}>{facingMode === 'environment' ? 'Frontal' : 'Traseira'}</Text>
              </TouchableOpacity>
            </View>

            {/* SELETOR RÁPIDO DE CLASSE ALVO */}
            <View style={styles.targetBar}>
              <View style={styles.targetBarHeader}>
                <Text style={styles.targetBarTitle}>🎯 CLASSE ALVO ESPERADA:</Text>
                <TouchableOpacity style={styles.changeClassBtn} onPress={() => setModalVisible(true)}>
                  <Text style={styles.changeClassBtnTxt}>Trocar Classe ⚙️</Text>
                </TouchableOpacity>
              </View>

              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.classChipsScroll}>
                <TouchableOpacity 
                  style={[styles.classChip, !selectedTargetClass && styles.classChipActive]}
                  onPress={() => setSelectedTargetClass(null)}
                >
                  <Text style={[styles.classChipTxt, !selectedTargetClass && styles.classChipTxtActive]}>Nenhum (Livre)</Text>
                </TouchableOpacity>

                {POPULAR_CLASSES.map(item => (
                  <TouchableOpacity 
                    key={item.code}
                    style={[styles.classChip, selectedTargetClass === item.code && styles.classChipActive]}
                    onPress={() => setSelectedTargetClass(item.code)}
                  >
                    <Text style={[styles.classChipTxt, selectedTargetClass === item.code && styles.classChipTxtActive]}>
                      {item.letter ? `${item.letter} (${item.code.slice(0, 4)}..)` : item.code}
                    </Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>

            {/* PAINEL DE CONTROLE E RESULTADOS */}
            <View style={styles.footer}>
                
                {/* STATUS OU ERRO */}
                {visionData.type === 'error' && (
                    <View style={styles.alertBox}>
                        <Text style={styles.alertText}>Erro: {visionData.message}</Text>
                    </View>
                )}
                {visionData.type === 'status' && (
                    <View style={styles.statusBox}>
                        <Text style={styles.statusText}>{visionData.message}</Text>
                    </View>
                )}

                {/* PAINEL COMBINADO: MODELO ALVO + CLASSE LIDA + INSTRUÇÃO DIDÁTICA */}
                <View style={styles.combinedCard}>
                  
                  {/* SEÇÃO DO MODELO ESPERADO (SE HOUVER ALVO) */}
                  {selectedTargetClass && targetParsed && (
                    <View style={styles.modelSection}>
                      <View style={styles.modelHeaderRow}>
                        <Text style={styles.sectionHeader}>MODELO DO SINAL ESPERADO</Text>
                        <Text style={styles.targetClassCodeBadge}>{selectedTargetClass}</Text>
                      </View>
                      
                      <View style={styles.modelContentRow}>
                        {targetImage && (
                          <Image source={targetImage} style={styles.targetSignThumbnail} />
                        )}
                        <View style={styles.modelSpecsCol}>
                          <Text style={styles.modelDescTxt}>
                            {targetClosest?.info?.description || 'Postura anatômica correspondente'}
                          </Text>
                          <Text style={styles.modelAnatomyTxt}>
                            Ind: {targetParsed.index.name} | Méd: {targetParsed.middle.name}
                          </Text>
                          <Text style={styles.modelAnatomyTxt}>
                            V/U: {targetParsed.spreads.middleIndex} | Pol: {targetParsed.thumb.description}
                          </Text>
                        </View>
                      </View>
                    </View>
                  )}

                  {/* INSTRUÇÃO BIOMECÂNICA AO VIVO */}
                  {guidance && (
                    <View style={[styles.guidanceBox, guidance.match ? styles.guidanceBoxSuccess : styles.guidanceBoxWarning]}>
                      <Text style={styles.guidanceMainTxt}>{guidance.mainAdvice}</Text>
                      <Text style={styles.guidanceSubTxt}>Acurácia postural: {guidance.accuracyScore}%</Text>
                    </View>
                  )}

                  {/* SEÇÃO DA CLASSE LIDA PELA IA */}
                  <View style={styles.detectedSection}>
                    <View style={styles.detectedHeaderRow}>
                      <Text style={styles.sectionHeader}>CLASSE LIDA PELA IA</Text>
                      {visionData.confidence > 0 && (
                        <Text style={styles.confidenceText}>{(visionData.confidence * 100).toFixed(1)}%</Text>
                      )}
                    </View>

                    <Text style={styles.detectedClassCode}>
                      {visionData.label ? visionData.label.toUpperCase() : 'Aguardando mão...'}
                    </Text>

                    {detectedClosest?.letter && visionData.label && (
                      <Text style={styles.equivalentLetterTxt}>
                        Equivalente ao Sinal: '{detectedClosest.letter}'
                      </Text>
                    )}

                    {/* STATUS ANATÔMICO EM TEMPO REAL DEDO A DEDO */}
                    {detectedParsed && (
                      <View style={styles.fingerStatusRow}>
                        <Text style={styles.fingerStatusPill}>
                          Ind: {detectedParsed.index.isExtended ? '☝️ Reto' : (detectedParsed.index.isClosed ? '✊ Dobrado' : '🌙 Curvo')}
                        </Text>
                        <Text style={styles.fingerStatusPill}>
                          Méd: {detectedParsed.middle.isExtended ? '☝️ Reto' : (detectedParsed.middle.isClosed ? '✊ Dobrado' : '🌙 Curvo')}
                        </Text>
                        <Text style={styles.fingerStatusPill}>
                          V/U: {detectedParsed.spreads.middleIndex.split(' ')[0]}
                        </Text>
                        <Text style={styles.fingerStatusPill}>
                          Pol: {detectedParsed.thumb.isOpposed ? '✊ Cruzando' : '👈 Aberto'}
                        </Text>
                      </View>
                    )}
                  </View>

                </View>

                {/* BOTÃO START / PAUSE */}
                <TouchableOpacity style={isRunning ? styles.stopBtn : styles.startBtn} onPress={toggleRunning}>
                  <Text style={styles.btnText}>{isRunning ? 'PAUSAR CÂMERA' : 'RETOMAR CÂMERA'}</Text>
                </TouchableOpacity>

            </View>

          </View>

          {/* MODAL PARA SELEÇÃO COMPLETA DE CLASSES */}
          <Modal visible={modalVisible} animationType="slide" transparent>
            <View style={styles.modalBackdrop}>
              <View style={styles.modalCard}>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Selecione a Classe Alvo</Text>
                  <TouchableOpacity onPress={() => setModalVisible(false)} style={styles.modalCloseBtn}>
                    <Text style={styles.modalCloseTxt}>✕</Text>
                  </TouchableOpacity>
                </View>

                {/* CAMPO DE BUSCA OU DIGITAÇÃO DE CLASSE */}
                <TextInput
                  style={styles.searchInput}
                  placeholder="Digite o código (ex: 4141000110) ou letra..."
                  placeholderTextColor="#7A8B90"
                  value={searchQuery}
                  onChangeText={setSearchQuery}
                  autoCapitalize="characters"
                />

                {/* BOTÃO PARA USAR CLASSE CUSTOMIZADA DIGITADA */}
                {searchQuery.length === 10 && /^\d{10}$/.test(searchQuery) && (
                  <TouchableOpacity 
                    style={styles.customClassBtn}
                    onPress={() => {
                      setSelectedTargetClass(searchQuery);
                      setModalVisible(false);
                    }}
                  >
                    <Text style={styles.customClassBtnTxt}>Usar código customizado: {searchQuery}</Text>
                  </TouchableOpacity>
                )}

                <ScrollView style={styles.classesList}>
                  <Text style={styles.listSubtitle}>Classes Canônicas (LIBRAS):</Text>
                  {filteredPopular.map(item => (
                    <TouchableOpacity 
                      key={item.code}
                      style={[styles.classItem, selectedTargetClass === item.code && styles.classItemActive]}
                      onPress={() => {
                        setSelectedTargetClass(item.code);
                        setModalVisible(false);
                      }}
                    >
                      <View>
                        <Text style={styles.classItemTitle}>{item.name}</Text>
                        <Text style={styles.classItemDesc}>{item.desc}</Text>
                      </View>
                      <Text style={styles.classItemCode}>{item.code}</Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>
            </View>
          </Modal>

        </>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F171A' },
  permissionContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 30, backgroundColor: '#0F171A' },
  permissionTitle: { color: '#00E5FF', fontSize: 20, fontWeight: '900', marginBottom: 12, textAlign: 'center' },
  permissionText: { color: '#A0B1B6', fontSize: 15, textAlign: 'center', marginBottom: 24, lineHeight: 22 },
  permissionButton: { backgroundColor: '#00E5FF', paddingVertical: 14, paddingHorizontal: 28, borderRadius: 14, shadowColor: '#00E5FF', shadowOpacity: 0.4, shadowRadius: 10, elevation: 6 },
  permissionButtonText: { color: '#000', fontWeight: '900', fontSize: 16 },
  visionContainer: { ...StyleSheet.absoluteFillObject },
  uiOverlay: { flex: 1, justifyContent: 'space-between', padding: 12, zIndex: 10 },
  
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 15, backgroundColor: 'rgba(15, 23, 26, 0.85)', padding: 12, borderRadius: 14, borderWidth: 1, borderColor: '#23343A' },
  backBtn: { backgroundColor: '#33464F', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
  backTxt: { color: '#FFF', fontWeight: 'bold', fontSize: 13 },
  title: { color: '#FFF', fontSize: 15, fontWeight: '800' },
  switchButton: { backgroundColor: 'rgba(255, 255, 255, 0.2)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 14, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.3)' },
  switchText: { color: '#FFF', fontWeight: '600', fontSize: 12 },

  targetBar: { backgroundColor: 'rgba(15, 23, 26, 0.85)', borderRadius: 14, padding: 10, borderWidth: 1, borderColor: '#23343A', marginTop: 6 },
  targetBarHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  targetBarTitle: { color: '#00E5FF', fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  changeClassBtn: { backgroundColor: '#23343A', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  changeClassBtnTxt: { color: '#00E5FF', fontSize: 11, fontWeight: 'bold' },
  classChipsScroll: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  classChip: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 10, backgroundColor: '#1A262C', borderWidth: 1, borderColor: '#33464F' },
  classChipActive: { backgroundColor: '#00E5FF', borderColor: '#FFF' },
  classChipTxt: { color: '#FFF', fontSize: 12, fontWeight: 'bold' },
  classChipTxtActive: { color: '#000' },
  
  footer: { marginBottom: 15, alignItems: 'center' },
  
  combinedCard: { width: '100%', backgroundColor: 'rgba(15, 23, 26, 0.92)', borderRadius: 16, padding: 14, borderWidth: 1, borderColor: '#23343A', marginBottom: 10 },
  sectionHeader: { color: '#7A8B90', fontSize: 10, fontWeight: '900', letterSpacing: 1 },
  
  modelSection: { borderBottomWidth: 1, borderColor: '#23343A', paddingBottom: 10, marginBottom: 10 },
  modelHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  targetClassCodeBadge: { color: '#00E5FF', fontSize: 11, fontWeight: 'bold', backgroundColor: 'rgba(0, 229, 255, 0.15)', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  modelContentRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  targetSignThumbnail: { width: 55, height: 55, borderRadius: 8, borderWidth: 1, borderColor: '#00E5FF' },
  modelSpecsCol: { flex: 1 },
  modelDescTxt: { color: '#FFF', fontSize: 13, fontWeight: 'bold', marginBottom: 2 },
  modelAnatomyTxt: { color: '#A0B1B6', fontSize: 11, fontWeight: '500' },

  guidanceBox: { width: '100%', padding: 10, borderRadius: 10, marginBottom: 10, borderWidth: 1 },
  guidanceBoxSuccess: { backgroundColor: 'rgba(0, 255, 128, 0.15)', borderColor: '#00FF80' },
  guidanceBoxWarning: { backgroundColor: 'rgba(255, 150, 0, 0.2)', borderColor: '#FF9600' },
  guidanceMainTxt: { color: '#FFF', fontSize: 13, fontWeight: 'bold', textAlign: 'center' },
  guidanceSubTxt: { color: '#DDD', fontSize: 10, textAlign: 'center', marginTop: 2 },

  detectedSection: {},
  detectedHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  confidenceText: { color: '#00FF80', fontSize: 11, fontWeight: 'bold' },
  detectedClassCode: { color: '#FFF', fontSize: 24, fontWeight: '900', textAlign: 'center', marginVertical: 4, letterSpacing: 2 },
  equivalentLetterTxt: { color: '#00E5FF', fontSize: 12, fontWeight: 'bold', textAlign: 'center', marginBottom: 6 },
  fingerStatusRow: { flexDirection: 'row', justifyContent: 'space-between', gap: 4, marginTop: 4 },
  fingerStatusPill: { flex: 1, backgroundColor: '#091012', color: '#A0B1B6', fontSize: 10, fontWeight: '600', paddingVertical: 4, textAlign: 'center', borderRadius: 6, borderWidth: 1, borderColor: '#1F2E35' },

  startBtn: { backgroundColor: '#1CB0F6', paddingVertical: 12, paddingHorizontal: 30, borderRadius: 16, borderBottomWidth: 3, borderColor: '#1899D6' },
  stopBtn: { backgroundColor: '#FF4B4B', paddingVertical: 10, paddingHorizontal: 26, borderRadius: 14, borderBottomWidth: 3, borderColor: '#EA2B2B' },
  btnText: { color: '#FFF', fontWeight: 'bold', fontSize: 13 },

  alertBox: { backgroundColor: 'rgba(255, 50, 50, 0.9)', padding: 10, borderRadius: 10, marginBottom: 8, width: '100%' },
  alertText: { color: 'white', fontWeight: 'bold', textAlign: 'center', fontSize: 12 },
  statusBox: { backgroundColor: 'rgba(28, 176, 246, 0.2)', padding: 8, borderRadius: 10, marginBottom: 8, width: '100%', borderWidth: 1, borderColor: '#1CB0F6' },
  statusText: { color: '#FFF', fontSize: 12, fontWeight: '600', textAlign: 'center' },

  modalBackdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.8)', justifyContent: 'flex-end' },
  modalCard: { backgroundColor: '#131F24', borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: 20, maxHeight: '80%', borderWidth: 1, borderColor: '#23343A' },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  modalTitle: { color: '#FFF', fontSize: 18, fontWeight: 'bold' },
  modalCloseBtn: { padding: 5 },
  modalCloseTxt: { color: '#FFF', fontSize: 18, fontWeight: 'bold' },
  searchInput: { backgroundColor: '#0F171A', color: '#FFF', padding: 12, borderRadius: 12, borderWidth: 1, borderColor: '#23343A', marginBottom: 10, fontSize: 14 },
  customClassBtn: { backgroundColor: '#00E5FF', padding: 10, borderRadius: 10, marginBottom: 10, alignItems: 'center' },
  customClassBtnTxt: { color: '#000', fontWeight: 'bold', fontSize: 13 },
  listSubtitle: { color: '#7A8B90', fontSize: 12, fontWeight: 'bold', marginBottom: 8 },
  classesList: { maxHeight: 350 },
  classItem: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, borderBottomWidth: 1, borderColor: '#1F2E35' },
  classItemActive: { backgroundColor: 'rgba(0, 229, 255, 0.15)', borderRadius: 8, paddingHorizontal: 8 },
  classItemTitle: { color: '#FFF', fontSize: 14, fontWeight: 'bold' },
  classItemDesc: { color: '#7A8B90', fontSize: 11 },
  classItemCode: { color: '#00E5FF', fontSize: 12, fontWeight: 'bold' }
});
