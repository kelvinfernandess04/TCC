import React from 'react';
import { WebView } from 'react-native-webview';
import { modelBase64 } from './modelBase64';
import { labels } from './labels';
import { calibratedSeeds } from './seedsCalibradas';

export default function VisionProcessor({ facingMode, onHandsDetected }) {
    
    // Injectable HTML containing the entire AI processing engine
    const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
        <style>
            body, html { margin: 0; padding: 0; width: 100%; height: 100%; background: #000; overflow: hidden; }
            video { 
                width: 100%; height: 100%; 
                object-fit: cover; 
                position: absolute; top: 0; left: 0; z-index: 1;
                transform: ${facingMode === 'user' ? 'scaleX(-1)' : 'scaleX(1)'}; 
            }
            canvas { 
                width: 100%; height: 100%; 
                position: absolute; top: 0; left: 0; z-index: 2; pointer-events: none;
                transform: ${facingMode === 'user' ? 'scaleX(-1)' : 'scaleX(1)'};
            }
        </style>
        <!-- TensorFlow.js Core & TFLite -->
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js"></script>
        
        <!-- MediaPipe Holistic -->
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"></script>
    </head>
    <body>
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            let holistic;
            let tfliteModel;
            const classLabels = ${JSON.stringify(labels)};
            const calibratedSeedsData = ${JSON.stringify(calibratedSeeds)};
            let lastInferenceTime = Date.now();

            const HAND_CONNECTIONS = [
                [0,1],[1,2],[2,3],[3,4],
                [0,5],[5,6],[6,7],[7,8],
                [5,9],[9,10],[10,11],[11,12],
                [9,13],[13,14],[14,15],[15,16],
                [13,17],[17,18],[18,19],[19,20],
                [0,17]
            ];
            
            function base64ToArrayBuffer(base64) {
                var binary_string = window.atob(base64);
                var len = binary_string.length;
                var bytes = new Uint8Array(len);
                for (var i = 0; i < len; i++) {
                    bytes[i] = binary_string.charCodeAt(i);
                }
                return bytes.buffer;
            }

            async function init() {
                try {
                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'status', message: 'Acessando câmera...' }));
                    
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: '${facingMode}' }
                    });
                    video.srcObject = stream;
                    
                    await new Promise((resolve) => {
                        video.onloadedmetadata = () => {
                            video.play();
                            canvas.width = window.innerWidth;
                            canvas.height = window.innerHeight;
                            window.addEventListener('resize', () => {
                                canvas.width = window.innerWidth;
                                canvas.height = window.innerHeight;
                            });
                            resolve();
                        };
                    });

                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'status', message: 'Carregando Modelos de IA...' }));

                    tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
                    const modelBuffer = base64ToArrayBuffer("${modelBase64}");
                    tfliteModel = await tflite.loadTFLiteModel(modelBuffer);

                    holistic = new Holistic({locateFile: (file) => {
                        return "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/" + file;
                    }});

                    holistic.setOptions({
                        modelComplexity: 1,
                        smoothLandmarks: true,
                        minDetectionConfidence: 0.5,
                        minTrackingConfidence: 0.5
                    });

                    holistic.onResults(onResults);

                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'status', message: 'Modelos Prontos.' }));
                    
                    async function processVideo() {
                        await holistic.send({image: video});
                        requestAnimationFrame(processVideo);
                    }
                    processVideo();

                } catch(e) {
                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'error', message: e.message }));
                }
            }

            // Agente 2: Normalização Espacial Abstrata 3D (Pulso na Origem, Escala e Invariância Angular)
            function normalizeLandmarks3D(landmarks) {
                const wrist = landmarks[0];
                const ptsTrans = [];
                for (let i = 0; i < 21; i++) {
                    ptsTrans.push({
                        x: landmarks[i].x - wrist.x,
                        y: landmarks[i].y - wrist.y,
                        z: (landmarks[i].z || 0.0) - (wrist.z || 0.0)
                    });
                }

                // Escala invariante: Distância Pulso(0) -> Base MCP Dedo Médio(9)
                const p9 = ptsTrans[9];
                const scale = Math.sqrt(p9.x * p9.x + p9.y * p9.y + p9.z * p9.z) || 1.0;
                const ptsScaled = ptsTrans.map(p => ({
                    x: p.x / scale,
                    y: p.y / scale,
                    z: p.z / scale
                }));

                // Base Ortonormal Local da Palma (Invariância Total a Rotação Global do Pulso)
                const uy = { x: ptsScaled[9].x, y: ptsScaled[9].y, z: ptsScaled[9].z };
                const normUy = Math.sqrt(uy.x*uy.x + uy.y*uy.y + uy.z*uy.z) || 1.0;
                uy.x /= normUy; uy.y /= normUy; uy.z /= normUy;

                const vArch = {
                    x: ptsScaled[5].x - ptsScaled[17].x,
                    y: ptsScaled[5].y - ptsScaled[17].y,
                    z: ptsScaled[5].z - ptsScaled[17].z
                };

                let uz = {
                    x: vArch.y * uy.z - vArch.z * uy.y,
                    y: vArch.z * uy.x - vArch.x * uy.z,
                    z: vArch.x * uy.y - vArch.y * uy.x
                };
                const normUz = Math.sqrt(uz.x*uz.x + uz.y*uz.y + uz.z*uz.z) || 1.0;
                uz.x /= normUz; uz.y /= normUz; uz.z /= normUz;

                let ux = {
                    x: uy.y * uz.z - uy.z * uz.y,
                    y: uy.z * uz.x - uy.x * uz.z,
                    z: uy.x * uz.y - uy.y * uz.x
                };
                const normUx = Math.sqrt(ux.x*ux.x + ux.y*ux.y + ux.z*ux.z) || 1.0;
                ux.x /= normUx; ux.y /= normUx; ux.z /= normUx;

                const ptsLocal = [];
                for (let i = 0; i < 21; i++) {
                    const p = ptsScaled[i];
                    ptsLocal.push({
                        x: p.x * ux.x + p.y * ux.y + p.z * ux.z,
                        y: p.x * uy.x + p.y * uy.y + p.z * uy.z,
                        z: p.x * uz.x + p.y * uz.y + p.z * uz.z
                    });
                }

                return { ptsLocal, scale };
            }

            // Agente 4: Classificador em Tempo Real por Seeds Calibradas e Matriz de Tolerância
            function classifyHandWithSeeds(landmarks) {
                if (!calibratedSeedsData || !calibratedSeedsData.classes) return null;

                const { ptsLocal } = normalizeLandmarks3D(landmarks);

                let bestClass = 'DESCONHECIDO';
                let bestSeedName = '';
                let minDistance = Infinity;
                let bestTolerancePassed = false;

                const classes = calibratedSeedsData.classes;
                for (const clsName in classes) {
                    const clsInfo = classes[clsName];
                    const weights = clsInfo.discriminative_joint_weights || [];
                    const weightSum = weights.reduce((a, b) => a + b, 0) || 21.0;

                    for (const subName in clsInfo.sub_seeds) {
                        const seed = clsInfo.sub_seeds[subName];
                        const seedLms = seed.landmarks_3d;
                        const thresholds = (seed.tolerance_matrix && seed.tolerance_matrix.joint_thresholds) || [];

                        let weightedSumSq = 0;
                        let passedCount = 0;

                        for (let i = 0; i < 21; i++) {
                            const dx = ptsLocal[i].x - seedLms[i].x;
                            const dy = ptsLocal[i].y - seedLms[i].y;
                            const dz = ptsLocal[i].z - seedLms[i].z;
                            const distSq = dx*dx + dy*dy + dz*dz;
                            const w = (weights[i] !== undefined) ? weights[i] : 1.0;
                            weightedSumSq += w * distSq;

                            if (thresholds[i] !== undefined && Math.sqrt(distSq) <= thresholds[i]) {
                                passedCount++;
                            }
                        }

                        const weightedEuc = Math.sqrt(weightedSumSq / weightSum);

                        if (weightedEuc < minDistance) {
                            minDistance = weightedEuc;
                            bestClass = clsName;
                            bestSeedName = subName;
                            bestTolerancePassed = (passedCount >= 17);
                        }
                    }
                }

                const confidence = Math.max(0.0, Math.min(1.0, 1.0 / (1.0 + minDistance * 2.8)));
                const cleanLabel = bestClass.replace(/^classe_/, '');

                return {
                    class: bestClass,
                    label: cleanLabel,
                    seed: bestSeedName,
                    confidence: confidence,
                    tolerancePassed: bestTolerancePassed,
                    distance: minDistance
                };
            }

            // Normalização 2D legada mantida para compatibilidade
            function normalizeLandmarks(landmarks) {
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
                return norm;
            }

            function getScaledCoords(landmark) {
                const vw = video.videoWidth;
                const vh = video.videoHeight;
                const cw = canvas.width;
                const ch = canvas.height;
                const scale = Math.max(cw / vw, ch / vh);
                const scaledW = vw * scale;
                const scaledH = vh * scale;
                const offsetX = (cw - scaledW) / 2;
                const offsetY = (ch - scaledH) / 2;
                return {
                    x: (landmark.x * scaledW) + offsetX,
                    y: (landmark.y * scaledH) + offsetY
                };
            }

            function drawHand(landmarks) {
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                for (const [start, end] of HAND_CONNECTIONS) {
                    const p1 = getScaledCoords(landmarks[start]);
                    const p2 = getScaledCoords(landmarks[end]);
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
                ctx.fillStyle = '#FFFFFF';
                for (let i = 0; i < landmarks.length; i++) {
                    const p = getScaledCoords(landmarks[i]);
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }

            async function onResults(results) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                const handLandmarks = results.rightHandLandmarks || results.leftHandLandmarks;
                
                if (handLandmarks) {
                    drawHand(handLandmarks);
                    
                    // Classificação Primária por Seeds Calibradas e Tolerâncias
                    const seedPred = classifyHandWithSeeds(handLandmarks);
                    const now = Date.now();
                    
                    if (seedPred && now - lastInferenceTime > 100) {
                        window.ReactNativeWebView.postMessage(JSON.stringify({
                            type: 'prediction',
                            label: seedPred.label,
                            seedName: seedPred.seed,
                            confidence: seedPred.confidence,
                            tolerancePassed: seedPred.tolerancePassed,
                            distance: seedPred.distance
                        }));
                        lastInferenceTime = now;
                    } else if (tfliteModel && now - lastInferenceTime > 100) {
                        // Fallback TFLite legado se seeds não estiverem disponíveis
                        const flatArr = normalizeLandmarks(handLandmarks);
                        const inputTensor = tf.tensor2d(flatArr, [1, 42], 'float32');
                        const output = tfliteModel.predict(inputTensor);
                        const outputTensor = output instanceof tf.Tensor ? output : output[0];
                        const outputData = outputTensor.dataSync();
                        let maxProb = 0, maxIndex = 0;
                        outputData.forEach((prob, idx) => {
                            if (prob > maxProb) { maxProb = prob; maxIndex = idx; }
                        });
                        window.ReactNativeWebView.postMessage(JSON.stringify({
                            type: 'prediction',
                            label: classLabels[maxIndex],
                            confidence: maxProb
                        }));
                        lastInferenceTime = now;
                        inputTensor.dispose();
                        if (output instanceof tf.Tensor) output.dispose();
                    }
                } else {
                    const now = Date.now();
                    if (now - lastInferenceTime > 300) {
                        window.ReactNativeWebView.postMessage(JSON.stringify({
                            type: 'prediction',
                            label: 'Aguardando mão...',
                            confidence: 0
                        }));
                        lastInferenceTime = now;
                    }
                }
            }

            init();
        </script>
    </body>
    </html>
    `;

    return (
        <WebView
            originWhitelist={['*']}
            source={{ html: htmlContent, baseUrl: 'https://localhost' }}
            style={{ flex: 1, backgroundColor: '#000' }}
            allowsInlineMediaPlayback={true}
            mediaPlaybackRequiresUserAction={false}
            mediaCapturePermissionGrantType="grant"
            javaScriptEnabled={true}
            domStorageEnabled={true}
            onMessage={(event) => {
                console.log("[WebView native message received]");
                try {
                    const data = JSON.parse(event.nativeEvent.data);
                    console.log(`[WebView Data Parsing] Type: ${data.type}`);
                    if (data.type === 'status' || data.type === 'prediction' || data.type === 'error') {
                        onHandsDetected(data);
                    }
                } catch(e) {
                    console.error("[WebView Parsing Error]", e);
                }
            }}
        />
    );
}
