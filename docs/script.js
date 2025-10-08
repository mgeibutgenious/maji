let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders'];

async function setupCamera() {
  const webcamElement = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcamElement.srcObject = stream;

  return new Promise((resolve) => {
    webcamElement.onloadedmetadata = () => {
      resolve(webcamElement);
    };
  });
}

async function loadModel() {
  model = await tf.loadLayersModel('tfjs_model/model.json');
  console.log("Model loaded");
}

async function predictLoop(video) {
  const resultP = document.getElementById('result');
  const tensor = tf.browser.fromPixels(video)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();
  
  const prediction = model.predict(tensor);
  const predictions = await prediction.data();
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const confidence = predictions[maxIndex] * 100;

  resultP.innerText = `Class: ${classLabels[maxIndex]} (${confidence.toFixed(2)}%)`;

  tf.dispose([tensor, prediction]);

  requestAnimationFrame(() => predictLoop(video));
}

async function start() {
  await loadModel();
  const video = await setupCamera();
  predictLoop(video);
}

start();
