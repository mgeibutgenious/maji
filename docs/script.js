let model, webcam;

const classLabels = ['Big Lot', 'C Press', 'Snyders']; // Customize this

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

async function predictLoop() {
  const webcamElement = document.getElementById('webcam');
  const tfImg = tf.browser.fromPixels(webcamElement).resizeBilinear([224, 224]).toFloat().div(255.0).expandDims(0);
  const prediction = await model.predict(tfImg).data();
  const maxIdx = prediction.indexOf(Math.max(...prediction));
  const label = classLabels[maxIdx];
  const confidence = (prediction[maxIdx] * 100).toFixed(2);

  document.getElementById('label').innerText = `Prediction: ${label} (${confidence}%)`;

  tfImg.dispose();
  requestAnimationFrame(predictLoop);
}

async function init() {
  await setupCamera();
  model = await tf.loadLayersModel('model/model.json');
  document.getElementById('label').innerText = 'Model Loaded';
  predictLoop();
}

init();
