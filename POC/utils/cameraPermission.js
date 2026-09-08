import { Platform, PermissionsAndroid } from 'react-native';
import { Camera } from 'expo-camera';

export async function requestAppCameraPermission() {
  try {
    if (Platform.OS === 'android') {
      const androidGranted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.CAMERA,
        {
          title: 'Permissão de Câmera',
          message: 'O aplicativo precisa da câmera para capturar e traduzir seus sinais em LIBRAS.',
          buttonPositive: 'Permitir',
          buttonNegative: 'Cancelar',
        }
      );
      if (androidGranted !== PermissionsAndroid.RESULTS.GRANTED) {
        console.warn('[CameraPermission] Android permission not granted:', androidGranted);
      }
    }
    const expoStatus = await Camera.requestCameraPermissionsAsync();
    return expoStatus.status === 'granted';
  } catch (error) {
    console.error('[CameraPermission] Error requesting camera permission:', error);
    return false;
  }
}

export async function checkAppCameraPermission() {
  try {
    if (Platform.OS === 'android') {
      const hasAndroid = await PermissionsAndroid.check(PermissionsAndroid.PERMISSIONS.CAMERA);
      if (!hasAndroid) return false;
    }
    const expoStatus = await Camera.getCameraPermissionsAsync();
    return expoStatus.status === 'granted';
  } catch (error) {
    console.error('[CameraPermission] Error checking camera permission:', error);
    return false;
  }
}
