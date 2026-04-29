import React from 'react';
import { WebView } from 'react-native-webview';
import { modelBase64 } from './modelBase64';
import { labels } from './labels';

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
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
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

            function drawHand(landmarks) {
                // Desenhar conexões
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                for (const [start, end] of HAND_CONNECTIONS) {
                    ctx.beginPath();
                    ctx.moveTo(landmarks[start].x * canvas.width, landmarks[start].y * canvas.height);
                    ctx.lineTo(landmarks[end].x * canvas.width, landmarks[end].y * canvas.height);
                    ctx.stroke();
                }

                // Desenhar pontos
                ctx.fillStyle = '#FFFFFF';
                for (let i = 0; i < landmarks.length; i++) {
                    ctx.beginPath();
                    ctx.arc(landmarks[i].x * canvas.width, landmarks[i].y * canvas.height, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }

            async function onResults(results) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                const handLandmarks = results.rightHandLandmarks || results.leftHandLandmarks;
                
                if (handLandmarks) {
                    drawHand(handLandmarks);
                    
                    if (tfliteModel) {
                        const flatArr = normalizeLandmarks(handLandmarks);
                        const inputTensor = tf.tensor2d(flatArr, [1, 42], 'float32');
                        const output = tfliteModel.predict(inputTensor);
                        
                        const outputTensor = output instanceof tf.Tensor ? output : output[0];
                        const outputData = outputTensor.dataSync();
                        
                        let maxProb = 0, maxIndex = 0;
                        outputData.forEach((prob, idx) => {
                            if (prob > maxProb) { maxProb = prob; maxIndex = idx; }
                        });

                        const now = Date.now();
                        if (now - lastInferenceTime > 100) {
                            window.ReactNativeWebView.postMessage(JSON.stringify({
                                type: 'prediction',
                                label: classLabels[maxIndex],
                                confidence: maxProb
                            }));
                            lastInferenceTime = now;
                        }
                        
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
