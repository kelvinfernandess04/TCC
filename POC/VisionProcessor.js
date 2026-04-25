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
        <!-- TensorFlow.js Core & WebGL backend -->
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
        <!-- TensorFlow.js TFLite backend -->
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js"></script>
        <!-- MediaPipe dependencies -->
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands"></script>
    </head>
    <body>
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            let detector;
            let tfliteModel;
            const classLabels = ${JSON.stringify(labels)};
            let lastInferenceTime = Date.now();
            
            // Utility: Convert Base64 string back into ArrayBuffer for TFLite
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
                    // Send loading status
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

                    // Initialize TFLite WebAssembly runtime
                    // tf-tflite requires Wasm binaries, we load them via CDN
                    tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
                    
                    const modelBuffer = base64ToArrayBuffer("${modelBase64}");
                    tfliteModel = await tflite.loadTFLiteModel(modelBuffer);

                    // Initialize MediaPipe Hand Pose
                    const model = handPoseDetection.SupportedModels.MediaPipeHands;
                    const detectorConfig = {
                      runtime: 'mediapipe',
                      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
                    };
                    detector = await handPoseDetection.createDetector(model, detectorConfig);

                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'status', message: 'Modelos Prontos. Iniciando...' }));
                    predictLoop();

                } catch(e) {
                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'error', message: e.message }));
                }
            }

            // Centralized Normalization mimicking Python logic (Bounding Box Min-Max)
            function normalizeLandmarks(keypoints) {
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                
                // Find boundaries
                for (let i = 0; i < 21; i++) {
                    minX = Math.min(minX, keypoints[i].x);
                    maxX = Math.max(maxX, keypoints[i].x);
                    minY = Math.min(minY, keypoints[i].y);
                    maxY = Math.max(maxY, keypoints[i].y);
                }
                
                const width = Math.max(maxX - minX, 1e-6);
                const height = Math.max(maxY - minY, 1e-6);
                const size = Math.max(width, height);
                
                const normKeypoints = [];
                for (let i = 0; i < 21; i++) {
                    const nx = (keypoints[i].x - minX) / size;
                    const ny = (keypoints[i].y - minY) / size;
                    normKeypoints.push(nx, ny);
                }
                
                return normKeypoints;
            }

            function drawPoints(keypoints) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#00FF00';
                for (let i = 0; i < keypoints.length; i++) {
                    ctx.beginPath();
                    ctx.arc(keypoints[i].x, keypoints[i].y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }

            async function predictLoop() {
                try {
                    if (detector && tfliteModel) {
                        const hands = await detector.estimateHands(video, {flipHorizontal: false});
                        
                        if (hands.length > 0) {
                            const keypoints = hands[0].keypoints;
                            drawPoints(keypoints);
                            
                            if (keypoints.length === 21) {
                                const flatArr = normalizeLandmarks(keypoints);
                                
                                // Inference
                                const inputTensor = tf.tensor2d(flatArr, [1, flatArr.length], 'float32');
                                const output = tfliteModel.predict(inputTensor);
                                
                                // tflite.predict can return an array, object or tensor depending on the model
                                const outputTensor = output instanceof tf.Tensor 
                                    ? output 
                                    : (Array.isArray(output) ? output[0] : Object.values(output)[0]);
                                
                                const outputData = outputTensor.dataSync();
                                // Find highest probability class
                                let maxProb = 0;
                                let maxIndex = 0;
                                outputData.forEach((prob, idx) => {
                                    if (prob > maxProb) {
                                        maxProb = prob;
                                        maxIndex = idx;
                                    }
                                });
                                
                                // To avoid spamming React Native bridge, send updates at most 15 times a second
                                const now = Date.now();
                                if (now - lastInferenceTime > 66) {
                                    window.ReactNativeWebView.postMessage(JSON.stringify({
                                        type: 'prediction',
                                        classIndex: maxIndex,
                                        label: classLabels[maxIndex],
                                        confidence: maxProb,
                                        inferenceTimeMs: now - lastInferenceTime
                                    }));
                                    lastInferenceTime = now;
                                }
                                
                                inputTensor.dispose();
                                if (output instanceof tf.Tensor) {
                                    output.dispose();
                                } else if (Array.isArray(output)) {
                                    output.forEach(t => t.dispose());
                                } else {
                                    Object.values(output).forEach(t => t.dispose());
                                }
                            }
                        } else {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            const now = Date.now();
                            if (now - lastInferenceTime > 200) {
                                window.ReactNativeWebView.postMessage(JSON.stringify({
                                    type: 'prediction',
                                    label: 'Nenhuma mão detectada',
                                    confidence: 0
                                }));
                                lastInferenceTime = now;
                            }
                        }
                    }
                } catch(e) {
                    window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'error', message: 'Draw/Predict falhou: ' + e.message }));
                }
                
                requestAnimationFrame(predictLoop);
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
