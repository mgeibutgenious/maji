let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders'];

// ===== Settings =====
const INPUT_SIZE = 224;            // must match your model
const TARGET_FPS = 10;             // throttle to keep devices stable
const MODEL_PATH = './tfjs_model/model.json';

// Reusable canvas for square crop
const canvas = document.createElement('canvas');
canvas.width = INPUT_SIZE;
canvas.height = INPUT_SIZE;
const ctx = canvas.getContext('2d');

const statusEl = document.getElementById('status');

// ===== Camera =====
async function setupCamera() {
  const video = document.getElementById('webcam');
  video.setAttribute('playsinline', '');
  video.muted = true;

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

// ===== Model =====
async function loadModel() {
  // Backend choice
  try { await tf.setBackend('webgl'); }
  catch { await tf.setBackend('wasm'); }
  await tf.ready();

  model = await tf.loadLayersModel(MODEL_PATH);
  statusEl.textContent = 'Model loaded. Warming up…';

  // Warm up to compile shaders
  tf.tidy(() => {
    const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    model.predict(dummy);
  });

  statusEl.textContent = 'Ready.';
}

// ===== Square center crop into canvas =====
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

// ===== UI Update: progress bars & vertical text =====
function updateUI(probs) {
  const probsArr = Array.from(probs);
  const maxIdx = probsArr.indexOf(Math.max(...probsArr));

  for (let i = 0; i < classLabels.length; i++) {
    const pct = Math.max(0, Math.min(100, probsArr[i] * 100));
    // bar fill width + highlight on max
    const fill = document.getElementById(`bar-${i}`);
    fill.style.width = pct.toFixed(1) + '%';
    if (i === maxIdx) fill.classList.add('highlight');
    else fill.classList.remove('highlight');

    // percentage text + bold on max
    const pctEl = document.getElementById(`pct-${i}`);
    pctEl.textContent = pct.toFixed(1) + '%';
    pctEl.classList.toggle('bold', i === maxIdx);
  }

  // Optional vertical text list (kept for accessibility/clarity)
  const textEl = document.getElementById('resultText');
  const lines = classLabels.map((label, i) => {
    const p = (probsArr[i] * 100).toFixed(2) + '%';
    return (i === maxIdx) ? `${label}: ${p}  ✅` : `${label}: ${p}`;
  });
  textEl.textContent = lines.join('\n');
}

// ===== Prediction loop (throttled + tidy to avoid leaks) =====
async function predictLoop(video) {
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
          .expandDims(0);          // [1,224,224,3]
        const pred = model.predict(img);
        probs = pred.dataSync();   // copy to CPU; pred & img auto-disposed at tidy end
      });

      updateUI(probs);

    } catch (err) {
      console.error('Inference error:', err);
      // Backend fallback if WebGL dies
      if (tf.getBackend() === 'webgl') {
        try {
          await tf.setBackend('wasm');
          await tf.ready();
          console.warn('Switched backend to WASM.');
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

// ===== Start =====
async function start() {
  await loadModel();
  const { video, stream } = await setupCamera();
  await predictLoop(video);

  // Cleanup on page hide/unload
  const cleanup = () => {
    stream.getTracks().forEach(t => t.stop());
    if (model) model.dispose();
    tf.engine().reset();
  };
  window.addEventListener('pagehide', cleanup);
  window.addEventListener('beforeunload', cleanup);

  // Optional: simple diagnostics every ~2s
  let frameCount = 0, lastReport = performance.now();
  const diag = () => {
    frameCount++;
    const now = performance.now();
    if (now - lastReport > 2000) {
      const mem = tf.engine().memory();
      console.log(
        `fps≈${(frameCount/((now-lastReport)/1000)).toFixed(1)}`,
        `tensors=${mem.numTensors}`,
        `bytes=${(mem.numBytes/1e6).toFixed(2)}MB`,
        `backend=${tf.getBackend()}`
      );
      frameCount = 0;
      lastReport = now;
    }
    requestAnimationFrame(diag);
  };
  requestAnimationFrame(diag);
}

start();
