import React, { useEffect, useRef, useState } from 'react';
import { IonButton, IonContent } from '@ionic/react';
import { DrawingUtils, FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import * as onnx from 'onnxruntime-web';
import './style.css';

const PoseLandmarkerComponent: React.FC = () => {
    const [poseLandmarker, setPoseLandmarker] = useState<PoseLandmarker | null>(null);
    const [webcamRunning, setWebcamRunning] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [repetitionCounter, setRepetitionCounter] = useState(0);
    
    useEffect(() => {
        const createPoseLandmarker = async () => {
            const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
            const landmarker = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                    delegate: "GPU"
                },
                runningMode: "IMAGE",
                numPoses: 2
            });
            setPoseLandmarker(landmarker);
        };
        createPoseLandmarker();
    }, []);

    const enableCam = async () => {
			if (!poseLandmarker) {
					console.log("Wait! poseLandmarker not loaded yet.");
					return;
			}
			if (webcamRunning) {
					setWebcamRunning(false);
					return;
			}
			setWebcamRunning(true);

			try {
					if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
							console.error("getUserMedia is not supported on this browser");
							return;
					}

					const constraints = {
							video: {
									facingMode: { ideal: "environment" } // Use the rear camera on mobile devices
							}
					};
					const stream = await navigator.mediaDevices.getUserMedia(constraints);
					if (videoRef.current) {
							videoRef.current.srcObject = stream;
							videoRef.current.addEventListener("loadeddata", predictWebcam);
					}
			} catch (error) {
					console.error("Error accessing webcam: ", error);
			}
		};

    const predictWebcam = async () => {
        if (!poseLandmarker || !videoRef.current || !canvasRef.current) return;
        
        const canvasCtx = canvasRef.current.getContext("2d");
        const drawingUtils = new DrawingUtils(canvasCtx);
        
        const detect = async () => {
            if (!poseLandmarker || !videoRef.current) return;
            
            poseLandmarker.setOptions({ runningMode: "VIDEO" });
            poseLandmarker.detectForVideo(videoRef.current, performance.now(), (result) => {
                if (canvasCtx) {
                    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                    result.landmarks.forEach((landmark) => {
                        drawingUtils.drawLandmarks(landmark, { radius: 5 });
                        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
                    });
                }
            });
            if (webcamRunning) {
                requestAnimationFrame(detect);
            }
        };
        detect();
    };

    return (
        <IonContent>
            <div className="video-container">
                <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: 'auto' }}></video>
                <canvas ref={canvasRef} width={480} height={360} />
            </div>
            <IonButton onClick={enableCam}>{webcamRunning ? "Disable" : "Enable"} Camera</IonButton>
            <div>Repetitions: {repetitionCounter}</div>
        </IonContent>
    );
};

export default PoseLandmarkerComponent;
