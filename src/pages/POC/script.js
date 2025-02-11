import {DrawingUtils, FilesetResolver, PoseLandmarker} from '@mediapipe/tasks-vision';
import * as onnx from 'onnxruntime-web';
import './style.css';

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 2
    });
};
createPoseLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
// Check if webcam access is supported.
const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
// If webcam supported, add event listener to button for when user wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}


/**
 * Calculates the angle in degrees between two 3D vectors.
 *
 * @param {Object} v1 - The first 3D vector with properties x, y, and z.
 * @param {Object} v2 - The second 3D vector with properties x, y, and z.
 * @return {number} The angle in degrees between the two vectors.
 */
function calculateAngle(v1, v2) {
    const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const magnitude1 = Math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2);
    const magnitude2 = Math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2);
    const radianAngles = Math.acos(dotProduct / (magnitude1 * magnitude2));
    return radianAngles * (180 / Math.PI);
}

/**
 * Creates a vector by calculating the difference between two points in 3D space.
 *
 * @param {Object} p1 - The starting point of the vector, with properties x, y, and z.
 * @param {Object} p2 - The ending point of the vector, with properties x, y, and z.
 * @return {Object} A vector object with properties x, y, and z representing the differences between corresponding coordinates of p2 and p1.
 */
function createVector(p1, p2) {
    return { x: p2.x - p1.x, y: p2.y - p1.y, z: p2.z - p1.z };
}

/**
 * Calculates angles between specified landmark points based on the given selections and landmark names.
 *
 * @param {Array} landmarks - An array of points representing landmark positions, with each point being an object or array containing coordinates.
 * @param {Array} selectedAngles - An array of strings, where each string represents a triplet of landmark names separated by dashes to define an angle (e.g., "A-B-C").
 * @param {Array} landmarkNames - An array of strings representing the names of all available landmarks, used to match the names in `selectedAngles`.
 * @return {Float32Array} A typed array containing the calculated angles in radians for each specified triplet in the `selectedAngles` array.
 */
function getAngles(landmarks, selectedAngles, landmarkNames) {
    const angleIndices = selectedAngles.map(angle => {
        return angle.split('-').map(name => landmarkNames.indexOf(name.trim()));
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

/**
 * Asynchronously loads normalization parameters from a CSV file.
 * Parses the CSV file content and transforms it into a dictionary
 * with angle names as keys and their corresponding mean and standard
 * deviation as values.
 *
 * @param {string} filePath - The path or URL to the CSV file containing normalization parameters.
 * @return {Promise<Object>} A promise that resolves to an object where each key is an angle name, and the value is an object containing the `mean` and `std` properties.
 */
async function loadNormParams(filePath) {
    const response = await fetch(filePath);
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
    normParams = normParams.reduce((obj, row) => {
        obj[row.angle] = { mean: row.mean, std: row.std };
        return obj;
    }, {});
    return normParams;
}

/**
 * Normalizes the specified input angles using the provided normalization parameters and input order.
 *
 * @param {Object} angles - An object containing angle names as keys and their corresponding values.
 * @param {Object} normParams - An object where keys are the angle names and values are objects containing `mean` and `std` for normalization.
 * @param {Array<string>} inputOrder - An array specifying the order of angle names to normalize.
 * @return {Float32Array} A Float32Array containing the normalized angle values, ordered as per the inputOrder.
 */
function normalizeInput(angles, normParams, inputOrder) {
    const normalizedAngles = inputOrder.map(angleName => {
        const angleValue = angles[angleName];
        const { mean, std } = normParams[angleName];
        return (angleValue - mean) / std;
    });
    return new Float32Array(normalizedAngles);
}

/**
 * Predicts anomalies in the provided angle data using a pre-trained ONNX model session.
 *
 * @param {onnx.InferenceSession} session - The ONNX inference session used for making predictions.
 * @param {Array<number>} angles - An array of angles to be normalized and used as input for prediction.
 * @param {number} threshold - The threshold value used to determine if an anomaly is detected.
 * @return {Promise<Object>} A promise that resolves to an object containing:
 *                            - `anomalies` (boolean): Whether anomalies were detected.
 *                            - `reconstructionError` (number): The calculated reconstruction error.
 * @throws Will throw an error if an issue occurs during the prediction process.
 */
async function predict(session, angles, threshold) {
    try {
        // Map selected angles to a dictionary for normalization
        const anglesDict = selectedAngles.reduce((dict, angle, index) => {
            dict[angle] = angles[index];
            return dict;
        }, {});
        // Normalize the angles
        const normalizedInput = normalizeInput(anglesDict, normParams, selectedAngles);
        // Create input tensor
        const inputTensor = new onnx.Tensor('float32', normalizedInput, [1, selectedAngles.length]);
        // Run inference
        const feeds = { 'input': inputTensor };
        const results = await session.run(feeds);
        // Get the reconstructed output
        const reconstructed = results.output.cpuData;
        // Calculate reconstruction error
        const reconstructionError = calculateReconstructionError(normalizedInput, reconstructed);
        // Determine anomalies
        const anomalies = reconstructionError > threshold;
        return { anomalies, reconstructionError };
    } catch (error) {
        console.error('Error in prediction:', error);
        throw error;
    }
}

/**
 * Computes the mean reconstruction error between the original input array and the reconstructed array.
 * The reconstruction error is calculated as the mean squared error for all elements in the given arrays.
 *
 * @param {Float32Array|Array<number>} input - The original input array of numerical values.
 * @param {Float32Array|Array<number>} reconstructed - The reconstructed array of numerical values to compare against the input.
 * @return {number} The mean reconstruction error as a single numeric value.
 */
function calculateReconstructionError(input, reconstructed) {
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

/**
 * Infers the class label and determines if anomalies are present based on reconstruction errors calculated using a machine learning model.
 *
 * @param {Object} session - The ONNX session instance used to run the inference model.
 * @param {Array<number>} angles - An array of angles used as input features for the model inference.
 * @return {Object} Returns an object containing the predicted class (`predictedClass`) and a boolean indicating if anomalies are detected (`anomalies`).
 */
async function inferClass(session, angles) {
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

// Downward-facing dog
//const normParamsFile = './models/downward_facing_dog/norm_params_AE.csv';
//const onnxPath = "./models/downward_facing_dog/AE.onnx";
// Push-up
//const normParamsFile = './models/classic_push_up/norm_params_AE.csv';
//const onnxPath = "./models/classic_push_up/AE.onnx";
const normParamsFile = './models/classic_push_up/norm_params_CVAE.csv';
const onnxPath = "./models/classic_push_up/CVAE.onnx";

// Load normalization parameters
const normParams = await loadNormParams(normParamsFile);
console.log("Normalization parameters:", normParams);
// Create the session with the model path
const session = await onnx.InferenceSession.create(onnxPath);
console.log("Successfully loaded model.");
let lastVideoTime = -1;

// For AE
//const threshold = 0.96;
// For CVAE
const threshold = 1.0;
let previousClass = null;
let repetitionCounter = 0;
let consecutivePreviousClass = 0;

async function predictWebcam() {
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await poseLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, async (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            try {
                const angles = getAngles(result.landmarks[0], selectedAngles, landmarkNames);

                // Run AE prediction
                // const prediction = await predict(session, angles, threshold);

                // Run CVAE prediction
                const prediction = await inferClass(session, angles);
                const currentClass = prediction.predictedClass === 1 ? "UP" : "DOWN";
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

                // Display prediction on the webpage
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
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }
            canvasCtx.restore();
        });
    }
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}