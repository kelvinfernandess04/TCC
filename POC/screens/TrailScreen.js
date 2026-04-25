import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, Modal, StatusBar } from 'react-native';
import { labels } from '../labels';
import { SignImages } from '../utils/dictionary';

export default function TrailScreen({ navigation }) {
  const [selectedLesson, setSelectedLesson] = useState(null);

  // Dividir os Sinais de 3 em 3
  const trailLevels = useMemo(() => {
    const levels = [];
    for (let i = 0; i < labels.length; i += 3) {
      levels.push({
        id: i / 3 + 1,
        content: labels.slice(i, i + 3)
      });
    }
    return levels;
  }, []);

  const openLessonPopup = (level) => {
    setSelectedLesson(level);
  };

  const startExercise = () => {
    const targetLesson = selectedLesson;
    setSelectedLesson(null);
    navigation.navigate('Exercise', { lesson: targetLesson });
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Trilha de Libras</Text>
        <TouchableOpacity style={styles.sandboxButton} onPress={() => navigation.navigate('Sandbox')}>
          <Text style={styles.sandboxText}>Modo Livre 🚀</Text>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        {trailLevels.map((level, index) => {
          // Efeito ZigZag do Duolingo
          const isLeft = index % 2 === 0;
          return (
            <View key={level.id} style={[styles.nodeContainer, isLeft ? styles.nodeLeft : styles.nodeRight]}>
              <TouchableOpacity 
                style={[styles.nodeCircle, index === 0 ? styles.nodeActive : styles.nodeLocked]} 
                onPress={() => openLessonPopup(level)}
                activeOpacity={0.7}
              >
                <Text style={styles.nodeText}>{level.id}</Text>
              </TouchableOpacity>
            </View>
          );
        })}
      </ScrollView>

      {/* POPUP DE LIÇÃO */}
      <Modal visible={!!selectedLesson} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.popupCard}>
            <Text style={styles.popupTitle}>Lição {selectedLesson?.id}</Text>
            <Text style={styles.popupSubtitle}>Nesta etapa, exercitaremos:</Text>
            
            <View style={styles.lettersContainer}>
              {selectedLesson?.content.map(letter => (
                <View key={letter} style={styles.letterBox}>
                   <Image source={SignImages[letter]} style={styles.letterImage} />
                   <Text style={styles.letterBadge}>{letter}</Text>
                </View>
              ))}
            </View>
            
            <View style={styles.buttonsRow}>
              <TouchableOpacity style={styles.closeBtn} onPress={() => setSelectedLesson(null)}>
                <Text style={styles.closeBtnText}>Voltar</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.startBtn} onPress={startExercise}>
                <Text style={styles.startBtnText}>Bora Lá!</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#131F24' },
  header: {
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: '#0F171A',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderColor: '#23343A'
  },
  headerTitle: { color: '#FFF', fontSize: 24, fontWeight: '800' },
  sandboxButton: { backgroundColor: '#58CC02', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12 },
  sandboxText: { color: '#FFF', fontWeight: 'bold', fontSize: 13 },
  scrollContainer: { paddingVertical: 40, alignItems: 'center' },
  nodeContainer: { width: '100%', alignItems: 'center', marginBottom: 40 },
  nodeLeft: { paddingRight: 80 },
  nodeRight: { paddingLeft: 80 },
  nodeCircle: {
    width: 80, height: 80, borderRadius: 40,
    justifyContent: 'center', alignItems: 'center',
    borderBottomWidth: 6,
  },
  nodeActive: { backgroundColor: '#58CC02', borderColor: '#58A700' },
  nodeLocked: { backgroundColor: '#CE82FF', borderColor: '#A559D6' }, // Todos livres para POC, mas roxos
  nodeText: { color: '#FFF', fontSize: 32, fontWeight: '900' },
  
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', padding: 20 },
  popupCard: { backgroundColor: '#1A2A30', borderRadius: 24, padding: 25, borderWidth: 2, borderColor: '#23343A' },
  popupTitle: { color: '#FFF', fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginBottom: 5 },
  popupSubtitle: { color: '#A0B1B6', fontSize: 16, textAlign: 'center', marginBottom: 20 },
  lettersContainer: { flexDirection: 'row', justifyContent: 'center', flexWrap: 'wrap', gap: 15, marginBottom: 30 },
  letterBox: { alignItems: 'center', backgroundColor: '#0F171A', borderRadius: 16, padding: 10 },
  letterImage: { width: 60, height: 60, borderRadius: 10, marginBottom: 10, resizeMode: 'cover' },
  letterBadge: { color: '#fff', fontSize: 20, fontWeight: '900' },
  
  buttonsRow: { flexDirection: 'row', justifyContent: 'space-between', gap: 15 },
  closeBtn: { flex: 1, backgroundColor: '#23343A', paddingVertical: 15, borderRadius: 16, alignItems: 'center', borderBottomWidth: 4, borderColor: '#131F24' },
  closeBtnText: { color: '#FFF', fontSize: 16, fontWeight: 'bold' },
  startBtn: { flex: 2, backgroundColor: '#58CC02', paddingVertical: 15, borderRadius: 16, alignItems: 'center', borderBottomWidth: 4, borderColor: '#58A700' },
  startBtnText: { color: '#FFF', fontSize: 16, fontWeight: 'bold' }
});
