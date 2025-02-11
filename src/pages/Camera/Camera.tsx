import { IonButton, IonContent, IonPage } from '@ionic/react';
import { useState, useRef, useEffect } from 'react';
import {DrawingUtils, FilesetResolver, PoseLandmarker} from '@mediapipe/tasks-vision';
import * as onnx from 'onnxruntime-web';

const CameraComponent: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState<string | null>(null);
	const [normParams, setNormParams] = useState<{} | null>(null)
	const [poseLandmarker, setPoseLandmarker] = useState<PoseLandmarker | undefined>(undefined);
	const [runningMode, setRunningMode] = useState<string>("VIDEO")
  const recordedChunks = useRef<Blob[]>([]);

	const landmarkNames = [
    "nose", "left eye (inner)", "left eye", "left eye (outer)",
    "right eye (inner)", "right eye", "right eye (outer)",
    "left ear", "right ear", "mouth (left)", "mouth (right)",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left pinky", "right pinky",
    "left index", "right index", "left thumb", "right thumb",
    "left hip", "right hip", "left knee", "right knee",
    "left ankle", "right ankle", "left heel", "right heel",
    "left foot index", "right foot index"
];
const selectedAngles = [
    "left wrist-left elbow-left shoulder", "right wrist-right elbow-right shoulder",
    "left elbow-left shoulder-left hip", "right elbow-right shoulder-right hip",
    "left shoulder-left hip-left knee", "right shoulder-right hip-right knee",
    "left hip-left knee-left ankle", "right hip-right knee-right ankle"
];

console.log("WASM Support:", typeof WebAssembly !== "undefined");


// const loadONNX = async () => {
// 	debugger
//   try {
		
//     const onnxImport = await import('onnxruntime-web');
		
// 		console.log(onnx)
// 		// setOnnx(onnxImport)
//     console.log("ONNX Runtime Web loaded:", onnx);
//   } catch (error) {
//     console.error("Failed to load ONNX Runtime:", error);
//   }
// };

	const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    const res = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 2
    });
		
		setPoseLandmarker(res)
	};

	const loadNormParams = async () =>  {
    const response = await fetch('/models/classic_push_up/norm_params_CVAE.csv');
    const csvText = await response.text();
    // Split rows and filter out empty lines
    const rows = csvText.split('\n').filter(row => row.trim() !== '');
    // Split header to get column names
    const headers = rows[0].split(',').map(header => header.trim());
    // Parse the data rows correctly
    let normParams = rows.slice(1).map(row => {
        const values = row.split(',').map(value => value.trim()); // Remove extra spaces and parse numbers
        return {
            angle: values[0], // The first column is the angle name
            mean: parseFloat(values[1]), // Second column is the mean
            std: parseFloat(values[2])   // Third column is the standard deviation
        };
    });
    // Reduce the parsed data into a dictionary with angle names as keys
    const norm = normParams.reduce((obj, row) => {
        obj[row.angle] = { mean: row.mean, std: row.std };
        return obj;
    }, {});

		setNormParams(norm)
	}



	useEffect(() => {
		createPoseLandmarker();
		loadNormParams();
		// loadONNX();
	}, [])
	
  // const requestPermissionsClick = async () => {
  //   try {
  //     const permStatus = await Camera.requestPermissions({
  //       permissions: ['', 'photos'],
  //     });

  //     if (permStatus. !== 'granted') {
  //       alert('Camera permission is required.');
  //       return;
  //     }
  //   } catch (error) {
  //     console.error('Permission request error:', error);
  //   }
  // };

	const calculateAngle = (v1, v2) => {
    const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const magnitude1 = Math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2);
    const magnitude2 = Math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2);
    const radianAngles = Math.acos(dotProduct / (magnitude1 * magnitude2));
    return radianAngles * (180 / Math.PI);
}

	const createVector = (p1, p2) => {
    return { x: p2.x - p1.x, y: p2.y - p1.y, z: p2.z - p1.z };
	}

	const getAngles = (landmarks, selectedAngles, landmarkNames) => {
    const angleIndices = selectedAngles.map((angle: string) => {
        return angle.split('-').map((name: string) => landmarkNames.indexOf(name.trim()));
    });
    const angles = angleIndices.map(indices => {
        const [p1, p2, p3] = indices.map(i => landmarks[i]);
        const v1 = createVector(p2, p1); // Vector from p1 to p2
        const v2 = createVector(p2, p3); // Vector from p3 to p2
        return calculateAngle(v1, v2);
    });
    const anglesArray = new Float32Array(angles);
    return anglesArray;
	}

	const normalizeInput = (angles, np, inputOrder) => {
    const normalizedAngles = inputOrder.map(angleName => {
        const angleValue = angles[angleName];
        const { mean, std } = np[angleName];
        return (angleValue - mean) / std;
    });
    return new Float32Array(normalizedAngles);
	}

	const calculateReconstructionError = (input, reconstructed) => {
    // Calculate mean squared error along each sample
    const errors = new Float32Array(input.length);
    let sumOfErrors = 0;
    for (let i = 0; i < input.length; i++) {
        const error = Math.pow(input[i] - reconstructed[i], 2);
        errors[i] = error;
        sumOfErrors += error;
    }
    // Calculate the mean reconstruction error
    return sumOfErrors / errors.length;
}

	const inferClass = async (session, angles) => {
		const threshold = 1;
    // Map selected angles to a dictionary for normalization
    const anglesDict = selectedAngles.reduce((dict, angle, index) => {
        dict[angle] = angles[index];
        return dict;
    }, {});
		
    // Normalize the angles
    const normalizedInput = normalizeInput(anglesDict, normParams, selectedAngles);
    // Create input tensor
    const inputTensor = new onnx.Tensor('float32', normalizedInput, [1, selectedAngles.length]);
    // Define possible classes
    const possibleClasses = [0, 1];
    // Store reconstruction errors for each class
    const errors = possibleClasses.map(() => []);
    for (const cls of possibleClasses) {
        // Create class label tensor
        const y = new Float32Array(inputTensor.dims[0]).fill(cls);
        const yTensor = new onnx.Tensor('float32', y, [inputTensor.dims[0]]);
        // Run the model with normalized input and class label y
        const feeds = { input: inputTensor, y: yTensor };
        const results = await session.run(feeds);
        // Get the reconstructed output
        const reconstructed = results.output.cpuData;
        // Calculate reconstruction error
        const reconstructionError = calculateReconstructionError(normalizedInput, reconstructed);
        errors[cls].push(reconstructionError);
      }
    // Find the class with the minimum reconstruction error
    const avgErrors = errors.map((errList) => errList.reduce((a, b) => a + b, 0) / errList.length);
    const predictedClass = avgErrors.indexOf(Math.min(...avgErrors));
    const minError = Math.min(...avgErrors);
    const anomalies = minError > threshold;
    return { predictedClass, anomalies };
}


	const predictWebcam = async () => {
		if (!poseLandmarker || !videoRef.current || !canvasRef.current) return;

		const videoHeight = "360px";
		const videoWidth = "480px";
		const canvasCtx: CanvasRenderingContext2D  | null = canvasRef.current.getContext("2d");
		if(!canvasCtx) return
		
		let previousClass: string | null = null;
		let consecutivePreviousClass: number = 0;
		let repetitionCounter: number = 0;
		const drawingUtils = new DrawingUtils(canvasCtx);

    const session = await onnx.InferenceSession.create("models/classic_push_up/CVAE.onnx");
    // Now let's start detecting the stream.
    let startTimeMs = performance.now();
		let lastVideoTime = -1;

    if (lastVideoTime !== videoRef.current.currentTime) {
        lastVideoTime = videoRef.current.currentTime;
        poseLandmarker.detectForVideo(videoRef.current, startTimeMs, async (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            try {
                const angles = getAngles(result.landmarks[0], selectedAngles, landmarkNames);
                // Run AE prediction
                // const prediction = await predict(session, angles, threshold);

                // Run CVAE prediction
                const prediction = await inferClass(session, angles);
                const currentClass = prediction.predictedClass === 1 ? "UP" : "DOWN";
								console.log(prediction);
								console.log(consecutivePreviousClass);
								console.log(repetitionCounter)
                // Check for transition from "DOWN" to "UP" with at least k consecutive "DOWN"s
                if (previousClass === "DOWN") {
                    if (currentClass === "DOWN") {
                        consecutivePreviousClass++;
                        console.log("Consecutive DOWNs: ", consecutivePreviousClass, consecutivePreviousClass >= 20);
                    }
                    else if (currentClass === "UP" && consecutivePreviousClass >= 20) {
                        repetitionCounter++;
                        consecutivePreviousClass = 0;
                    }
                }
                // Update the previous class for the next iteration
                previousClass = currentClass;

                const outputDiv = document.getElementById("output");
                outputDiv.innerHTML = `
                    <h3>Prediction</h3>
                    <span style="color: ${prediction.anomalies ? 'red' : 'green'}">
                        ${prediction.anomalies ? "BAD" : "OK"}
                    </span>
                    <!-- For CVAE -->
                    <br><br>
                    Number of repetitions: ${repetitionCounter}
                    <br>
                    <span style="visibility: ${prediction.anomalies ? 'hidden' : 'visible'}">
                        ${currentClass}
                    </span>
                `;
            } catch (error) {
                console.error("Error testing ONNX model:", error);
            }
            for (const landmark of result.landmarks) {
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => data && DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }
            canvasCtx.restore();
        });
    }
    // Call this function again to keep predicting when the browser is ready.

		// 		canvasRef.current.style.height = videoHeight;
    // videoRef.current.style.height = videoHeight;
    // canvasRef.current.style.width = videoWidth;
    // videoRef.current.style.width = videoWidth;
		window.requestAnimationFrame(predictWebcam);

	}

  // Start camera
  const startCamera = async () => {
    try {
			navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener("loadeddata", predictWebcam);
				setStream(stream);
    });
  
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
			window.cancelAnimationFrame(animationFrameRef.current);
    }
  };

  // Start recording
  const startRecording = () => {
    if (stream) {
      recordedChunks.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = event => {
        if (event.data.size > 0) {
          recordedChunks.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(recordedChunks.current, { type: 'video/mp4' });
        const videoUrl = URL.createObjectURL(blob);
        setVideoBlob(videoUrl);
      };

      recorder.start();
      mediaRecorder.current = recorder;
      setRecording(true);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      setRecording(false);
    }
  };

	const testCamera = async () => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ video: true });
			console.log('Camera access granted:', stream);
		} catch (error) {
			console.error('Error accessing camera:', error);
			alert(`Camera error: ${error.message}`);
		}
	};
	
	useEffect(() => {
		testCamera();
	}, []);

  useEffect(() => {
    return () => {
      stopCamera(); // Stop camera when component unmounts
    };
  }, []);

  return (
    <IonPage>
      <IonContent className="ion-padding">
				<div style={{paddingTop: 100}}></div>
				{/* <IonButton onClick={requestPermissionsClick}>Premissionnn</IonButton> */}
        <IonButton onClick={startCamera}>Start</IonButton>
        <IonButton color="danger" onClick={stopCamera}>Stop</IonButton> 

        {/* {stream && !recording && <IonButton color="success" onClick={startRecording}>Start Recording</IonButton>} */}
        {recording && <IonButton color="warning" onClick={stopRecording}>Stop Recording</IonButton>}

        {/* Video Preview (Live Camera) */}
        <div style={{position: 'relative', width: '400px', height: '533'}}>
          <video ref={videoRef} autoPlay playsInline width={400} height={533} />
					<canvas className='output_canvas' ref={canvasRef} width={400} height={533} style={{position: 'absolute', top: 0}} />
        </div>
				<div id='output'></div>
      </IonContent>
    </IonPage>
  );
};

export default CameraComponent;