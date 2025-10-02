let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders'];

// ====== Settings ======
const INPUT_SIZE = 224;   // must match your trained model
const TARGET_FPS = 10;    // throttle to avoid freeze/overheat (try 8–15)
const MODEL_PATH = './tfjs_model/model.json';

// ====== Camera setup (rear camera preferred) ======
async function setupCamera() {
  const video = document.getElementById('webcam');

  // Ensure mobile autoplay compatibility
  video.setAttribute('playsinline', '');
  video.muted = true; // some browsers require muted for autoplay

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 640 },
      height: { ideal: 640 }
    },
    audio: false
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = async () => {
      await video.play();
      resolve({ video, stream });
    };
  });
}

// ====== Load model and warm up ======
async function loadModel() {
  // Prefer WebGL, fall back to WASM if needed
  try {
    await tf.setBackend('webgl');
  } catch {
    await tf.setBackend('wasm');
  }
  await tf.ready();

  model = await tf.loadLayersModel(MODEL_PATH);
  console.log('Model loaded');

  // Warm up once so shaders compile before the live loop
  tf.tidy(() => {
    const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    model.predict(dummy);
  });
}

// ====== Draw a centered square crop into a reusable canvas ======
const canvas = document.createElement('canvas');
canvas.width = INPUT_SIZE;
canvas.height = INPUT_SIZE;
const ctx = canvas.getContext('2d');

function drawSquareCropToCanvas(video) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return false;

  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;
  // Cover the square canvas with a center crop from the video
  ctx.drawImage(video, sx, sy, side, side, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return true;
}

// ====== Prediction loop (throttled & leak-free) ======
async function predictLoop(video) {
  const resultP = document.getElementById('result');
  const frameInterval = 1000 / TARGET_FPS;
  let lastTime = 0;

  const loop = async (ts) => {
    try {
      // Throttle FPS
      if (ts - lastTime < frameInterval) {
        requestAnimationFrame(loop);
        return;
      }
      lastTime = ts;

      if (!drawSquareCropToCanvas(video)) {
        requestAnimationFrame(loop);
        return;
      }

      let probs;
      // All tensor work inside tidy → auto-dispose each frame
      tf.tidy(() => {
        const img = tf.browser.fromPixels(canvas)         // [224,224,3]
          .toFloat()
          .div(255)
          .expandDims(0);                                 // [1,224,224,3]
        const pred = model.predict(img);                  // [1,C]
        probs = pred.dataSync();                          // copy to CPU
      });

      // Display confidences
      const lines = classLabels.map((label, i) =>
        `${label}: ${(probs[i] * 100).toFixed(2)}%`
      );
      resultP.textContent = lines.join('\n');
    } catch (err) {
      console.error('Inference error:', err);
      // Try backend fallback once if WebGL fails
      if (tf.getBackend() === 'webgl') {
        try {
          await tf.setBackend('wasm');
          await tf.ready();
          console.warn('Switched backend to WASM after error.');
        } catch (e) {
          console.error('Backend switch failed:', e);
        }
      }
    } finally {
      requestAnimationFrame(loop);
    }
  };

  requestAnimationFrame(loop);
}

// ====== Start ======
async function start() {
  await loadModel();
  const { video, stream } = await setupCamera();
  await predictLoop(video);

  // Optional cleanup on page hide/unload
  const cleanup = () => {
    stream.getTracks().forEach(t => t.stop());
    if (model) model.dispose();
    tf.engine().reset();
  };
  window.addEventListener('pagehide', cleanup);
  window.addEventListener('beforeunload', cleanup);
}

start();
