import React, { useRef, useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import { WebView } from 'react-native-webview';
import { modelBase64 } from './modelBase64';
import { labels } from './labels';

export default function VisionProcessor({ 
    facingMode = 'environment', 
    targetPoints = null, 
    targetLabel = '', 
    onHandsDetected 
}) {
    const webViewRef = useRef(null);

    // Sincroniza o esqueleto do modelo alvo com o WebView em tempo real
    useEffect(() => {
        if (webViewRef.current) {
            const payload = JSON.stringify({ points: targetPoints, label: targetLabel });
            webViewRef.current.injectJavaScript(`
                if (window.updateTarget) {
                    window.updateTarget(${payload});
                }
                true;
            `);
        }
    }, [targetPoints, targetLabel]);

    // Limpeza de recursos da câmera ao desmontar
    useEffect(() => {
        return () => {
            if (webViewRef.current) {
                webViewRef.current.injectJavaScript(`
                    if (window.stopCamera) {
                        window.stopCamera();
                    }
                    true;
                `);
            }
        };
    }, []);

    const isUserFacing = facingMode === 'user';

    const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
        <style>
            * { box-sizing: border-box; }
            body, html { 
                margin: 0; 
                padding: 0; 
                width: 100%; 
                height: 100%; 
                background: #000; 
                overflow: hidden; 
            }
            #video {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                z-index: 1;
                background: #000;
            }
            #video.mirrored {
                transform: scaleX(-1);
                -webkit-transform: scaleX(-1);
            }
            #canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 2;
                pointer-events: none;
                background: transparent;
            }
            #status-pill {
                position: absolute;
                top: 14px;
                left: 14px;
                z-index: 10;
                background: rgba(15, 23, 26, 0.88);
                border: 1px solid #23343A;
                color: #00E5FF;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.3px;
                padding: 5px 12px;
                border-radius: 20px;
                pointer-events: none;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            }
            #status-pill.error {
                border-color: #FF5252;
                color: #FF5252;
                background: rgba(40, 10, 10, 0.9);
            }
            #status-pill.success {
                border-color: #00FF80;
                color: #00FF80;
            }
        </style>
        <!-- MediaPipe Hands -->
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
        
        <!-- TensorFlow.js Core & TFLite -->
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js" crossorigin="anonymous"></script>
    </head>
    <body>
        <video id="video" autoplay playsinline webkit-playsinline muted class="${isUserFacing ? 'mirrored' : ''}"></video>
        <canvas id="canvas"></canvas>
        <div id="status-pill">Iniciando câmera...</div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const statusPill = document.getElementById('status-pill');
            
            let hands;
            let tfliteModel = null;
            const classLabels = ${JSON.stringify(labels)};
            let lastInferenceTime = Date.now();
            const isUserCamera = ${isUserFacing};

            window.currentTarget = { points: ${JSON.stringify(targetPoints)}, label: "${targetLabel}" };

            function setStatus(msg, type) {
                if (statusPill) {
                    statusPill.innerText = msg;
                    statusPill.className = type === 'error' ? 'error' : (type === 'success' ? 'success' : '');
                }
                if (window.ReactNativeWebView) {
                    window.ReactNativeWebView.postMessage(JSON.stringify({
                        type: type === 'error' ? 'error' : 'status',
                        message: msg
                    }));
                }
            }

            window.updateTarget = function(data) {
                window.currentTarget = data || { points: null, label: '' };
            };

            window.stopCamera = function() {
                if (video && video.srcObject) {
                    video.srcObject.getTracks().forEach(t => t.stop());
                    video.srcObject = null;
                }
            };
            window.addEventListener('beforeunload', window.stopCamera);

            const HAND_CONNECTIONS = [
                [0,1],[1,2],[2,3],[3,4],
                [0,5],[5,6],[6,7],[7,8],
                [5,9],[9,10],[10,11],[11,12],
                [9,13],[13,14],[14,15],[15,16],
                [13,17],[17,18],[18,19],[19,20],
                [0,17]
            ];

            // Mapeamento geométrico das coordenadas normalizadas do MediaPipe (0..1)
            // para as dimensões reais renderizadas pelo 'object-fit: cover' do vídeo
            function mapLandmark(p) {
                const screenW = canvas.width;
                const screenH = canvas.height;
                const videoW = video.videoWidth || 640;
                const videoH = video.videoHeight || 480;

                const screenRatio = screenW / screenH;
                const videoRatio = videoW / videoH;

                let renderW, renderH, offsetX, offsetY;
                if (screenRatio > videoRatio) {
                    renderW = screenW;
                    renderH = screenW / videoRatio;
                    offsetX = 0;
                    offsetY = (screenH - renderH) / 2;
                } else {
                    renderH = screenH;
                    renderW = screenH * videoRatio;
                    offsetX = (screenW - renderW) / 2;
                    offsetY = 0;
                }

                const rx = isUserCamera ? (1 - p.x) : p.x;
                return {
                    x: offsetX + rx * renderW,
                    y: offsetY + p.y * renderH
                };
            }

            // Renderiza o esqueleto real da mão diretamente sobre a imagem da câmera
            function drawRealHand(landmarks) {
                ctx.strokeStyle = '#00FF80';
                ctx.lineWidth = 4;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                for (const [start, end] of HAND_CONNECTIONS) {
                    const p1 = mapLandmark(landmarks[start]);
                    const p2 = mapLandmark(landmarks[end]);
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }

                for (let i = 0; i < landmarks.length; i++) {
                    const p = mapLandmark(landmarks[i]);
                    const isTip = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20);
                    ctx.fillStyle = isTip ? '#FF007F' : '#FFFFFF';
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, isTip ? 6 : 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }

            // Renderiza o "Modelo Alvo" em formato de esqueleto num card PiP no canto superior
            function drawTargetPip(targetPoints, label) {
                if (!targetPoints || targetPoints.length < 21) return;
                
                const boxW = 125;
                const boxH = 135;
                const pad = 14;
                const boxX = canvas.width - boxW - pad;
                const boxY = pad;

                // Fundo translúcido
                ctx.fillStyle = 'rgba(15, 23, 26, 0.90)';
                ctx.strokeStyle = '#00E5FF';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(boxX, boxY, boxW, boxH, 12);
                } else {
                    ctx.rect(boxX, boxY, boxW, boxH);
                }
                ctx.fill();
                ctx.stroke();

                // Cabeçalho
                ctx.fillStyle = '#00E5FF';
                ctx.font = 'bold 11px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('MODELO ALVO', boxX + boxW / 2, boxY + 18);
                if (label) {
                    ctx.fillStyle = '#FFFFFF';
                    ctx.font = 'bold 10px sans-serif';
                    const shortLbl = label.length > 15 ? label.slice(0, 15) + '..' : label;
                    ctx.fillText(shortLbl, boxX + boxW / 2, boxY + 30);
                }

                // Esqueleto modelo
                const skelX = boxX + 16;
                const skelY = boxY + 36;
                const skelW = boxW - 32;
                const skelH = boxH - 44;

                ctx.strokeStyle = '#00E5FF';
                ctx.lineWidth = 2.5;
                for (const [s, e] of HAND_CONNECTIONS) {
                    const p1 = targetPoints[s];
                    const p2 = targetPoints[e];
                    if (!p1 || !p2) continue;
                    ctx.beginPath();
                    ctx.moveTo(skelX + p1[0] * skelW, skelY + p1[1] * skelH);
                    ctx.lineTo(skelX + p2[0] * skelW, skelY + p2[1] * skelH);
                    ctx.stroke();
                }

                ctx.fillStyle = '#FFFFFF';
                for (let i = 0; i < targetPoints.length; i++) {
                    const p = targetPoints[i];
                    if (!p) continue;
                    ctx.beginPath();
                    ctx.arc(skelX + p[0] * skelW, skelY + p[1] * skelH, 2.5, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }

            // Normalização 42 features (mesma utilizada no treino da IA)
            function predictGesture(landmarks) {
                const now = Date.now();
                if (now - lastInferenceTime < 100 || !tfliteModel) return;
                lastInferenceTime = now;

                try {
                    let minX = Infinity, maxX = -Infinity;
                    let minY = Infinity, maxY = -Infinity;
                    for (let i = 0; i < 21; i++) {
                        minX = Math.min(minX, landmarks[i].x);
                        maxX = Math.max(maxX, landmarks[i].x);
                        minY = Math.min(minY, landmarks[i].y);
                        maxY = Math.max(maxY, landmarks[i].y);
                    }
                    const width = Math.max(maxX - minX, 1e-6);
                    const height = Math.max(maxY - minY, 1e-6);
                    const size = Math.max(width, height);

                    const norm = [];
                    for (let i = 0; i < 21; i++) {
                        norm.push((landmarks[i].x - minX) / size);
                        norm.push((landmarks[i].y - minY) / size);
                    }

                    const inputTensor = tf.tensor2d(norm, [1, 42], 'float32');
                    const output = tfliteModel.predict(inputTensor);
                    const outputTensor = output instanceof tf.Tensor ? output : output[0];
                    const outputData = outputTensor.dataSync();

                    let maxProb = 0, maxIndex = 0;
                    for (let i = 0; i < outputData.length; i++) {
                        if (outputData[i] > maxProb) {
                            maxProb = outputData[i];
                            maxIndex = i;
                        }
                    }

                    const predictedClass = classLabels[maxIndex] || '0000000000';
                    window.ReactNativeWebView.postMessage(JSON.stringify({
                        type: 'prediction',
                        label: predictedClass,
                        confidence: maxProb,
                        landmarks: landmarks.map(p => ({ x: p.x, y: p.y }))
                    }));

                    inputTensor.dispose();
                    if (output instanceof tf.Tensor) output.dispose();
                } catch(e) {
                    console.error("Inference error:", e);
                }
            }

            function onResults(results) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Desenha o esqueleto e processa predição se houver mão
                if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                    const handLandmarks = results.multiHandLandmarks[0];
                    drawRealHand(handLandmarks);
                    predictGesture(handLandmarks);
                } else {
                    const now = Date.now();
                    if (now - lastInferenceTime > 350) {
                        window.ReactNativeWebView.postMessage(JSON.stringify({
                            type: 'prediction',
                            label: 'Aguardando mão...',
                            confidence: 0
                        }));
                        lastInferenceTime = now;
                    }
                }

                // Desenha o Modelo Alvo PiP
                if (window.currentTarget && window.currentTarget.points) {
                    drawTargetPip(window.currentTarget.points, window.currentTarget.label);
                }
            }

            // Captura robusta com fallback inteligente de constraints
            async function getCameraStream() {
                setStatus('Solicitando câmera...', 'info');

                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    try {
                        return await navigator.mediaDevices.getUserMedia({
                            video: { 
                                facingMode: { ideal: '${facingMode}' },
                                width: { ideal: 640 },
                                height: { ideal: 480 }
                            },
                            audio: false
                        });
                    } catch(err1) {
                        console.warn("Primeira tentativa getUserMedia falhou, tentando video simples:", err1);
                        try {
                            return await navigator.mediaDevices.getUserMedia({
                                video: true,
                                audio: false
                            });
                        } catch(err2) {
                            throw err2;
                        }
                    }
                }

                const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
                if (legacy) {
                    return new Promise((resolve, reject) => {
                        legacy.call(navigator, { video: true, audio: false }, resolve, reject);
                    });
                }

                throw new Error('Câmera indisponível no WebView (mediaDevices não encontrado).');
            }

            async function init() {
                try {
                    const stream = await getCameraStream();
                    video.srcObject = stream;
                    video.setAttribute('playsinline', '');
                    video.setAttribute('webkit-playsinline', '');

                    await new Promise((resolve) => {
                        video.onloadedmetadata = () => {
                            video.play().then(resolve).catch(resolve);
                        };
                        setTimeout(resolve, 800);
                    });

                    setStatus('📷 Câmera ativa • Carregando IA...', 'info');

                    function resizeCanvas() {
                        canvas.width = window.innerWidth;
                        canvas.height = window.innerHeight;
                    }
                    resizeCanvas();
                    window.addEventListener('resize', resizeCanvas);

                    hands = new Hands({
                        locateFile: (file) => "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" + file
                    });

                    hands.setOptions({
                        maxNumHands: 1,
                        modelComplexity: 1,
                        minDetectionConfidence: 0.5,
                        minTrackingConfidence: 0.5
                    });

                    hands.onResults(onResults);

                    let isProcessing = false;
                    async function frameLoop() {
                        if (video.readyState >= 2 && !isProcessing) {
                            isProcessing = true;
                            try {
                                await hands.send({ image: video });
                            } catch(e) {}
                            isProcessing = false;
                        }
                        requestAnimationFrame(frameLoop);
                    }
                    requestAnimationFrame(frameLoop);

                    setStatus('🟢 Detector Ativo! Aguardando mão...', 'success');

                    // Carrega o classificador TFLite
                    setTimeout(async () => {
                        try {
                            tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
                            const res = await fetch("data:application/octet-stream;base64,${modelBase64}");
                            const buffer = await res.arrayBuffer();
                            tfliteModel = await tflite.loadTFLiteModel(buffer);
                            setStatus('🟢 IA 100% Pronta!', 'success');
                        } catch(modelErr) {
                            console.error("Erro modelo:", modelErr);
                        }
                    }, 150);

                } catch(e) {
                    const msg = e.name ? (e.name + ': ' + e.message) : e.toString();
                    setStatus('🔴 ' + msg, 'error');
                }
            }

            init();
        </script>
    </body>
    </html>
    `;

    return (
        <View style={styles.container}>
            <WebView
                ref={webViewRef}
                originWhitelist={['*']}
                source={{ html: htmlContent, baseUrl: 'https://localhost' }}
                style={styles.webView}
                allowsInlineMediaPlayback={true}
                mediaPlaybackRequiresUserAction={false}
                mediaCapturePermissionGrantType="grant"
                javaScriptEnabled={true}
                domStorageEnabled={true}
                androidHardwareAccelerationDisabled={false}
                androidLayerType="hardware"
                mixedContentMode="always"
                scrollEnabled={false}
                bounces={false}
                onMessage={(event) => {
                    try {
                        const data = JSON.parse(event.nativeEvent.data);
                        if (data.type === 'status' || data.type === 'prediction' || data.type === 'error') {
                            onHandsDetected(data);
                        }
                    } catch(e) {}
                }}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, width: '100%', height: '100%', backgroundColor: '#000' },
    webView: { flex: 1, width: '100%', height: '100%', backgroundColor: '#000' }
});
