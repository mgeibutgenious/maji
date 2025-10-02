let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders'];

// ====== Settings ======
const INPUT_SIZE = 224;   // must match your trained model
const TARGET_FPS = 20;    // throttle FPS (try 8–15)
const MODEL_PATH = './tfjs_model/model.json';

// ====== Camera setup (rear camera preferred) ======
async function setupCamera() {
  const video = document.getElementById('webcam');

  // Ensure mobile autoplay compatibility
  video.setAttribute('playsinline', '');
  video.muted = true; // required for autoplay on mobile

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
  try {
    await tf.setBackend('webgl');
  } catch {
    await tf.setBackend('wasm');
  }
  await tf.ready();

  model = await tf.loadLayersModel(MODEL_PATH);
  console.log('✅ Model loaded');

  // Warm up model
  tf.tidy(() => {
    const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    model.predict(dummy);
  });
}

// ====== Square crop via canvas ======
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
  ctx.drawImage(video, sx, sy, side, side, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return true;
}

// ====== Prediction loop ======
async function predictLoop(video) {
  const resultP = document.getElementById('result');
  const frameInterval = 1000 / TARGET_FPS;
  let lastTime = 0;

  const loop = async (ts) => {
    try {
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
      tf.tidy(() => {
        const img = tf.browser.fromPixels(canvas)
          .toFloat()
          .div(255)
          .expandDims(0);
        const pred = model.predict(img);
        probs = pred.dataSync();
      });

      // === Format output ===
      const probsArr = Array.from(probs);
      const maxIdx = probsArr.indexOf(Math.max(...probsArr));

      let html = '';
      for (let i = 0; i < classLabels.length; i++) {
        const pct = (probsArr[i] * 100).toFixed(2) + '%';
        if (i === maxIdx) {
          html += `<div style="font-weight:700; color:#14833b;">${classLabels[i]}: ${pct}</div>`;
        } else {
          html += `<div>${classLabels[i]}: ${pct}</div>`;
        }
      }
      resultP.innerHTML = html;

    } catch (err) {
      console.error('Inference error:', err);
      if (tf.getBackend() === 'webgl') {
        try {
          await tf.setBackend('wasm');
          await tf.ready();
          console.warn('⚠️ Switched backend to WASM after error.');
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

  const cleanup = () => {
    stream.getTracks().forEach(t => t.stop());
    if (model) model.dispose();
    tf.engine().reset();
  };
  window.addEventListener('pagehide', cleanup);
  window.addEventListener('beforeunload', cleanup);
}

start();
