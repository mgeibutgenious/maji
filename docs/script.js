let model;
const classLabels = ['Big Lot', 'C Press', 'Snyders']; // your original order
const INPUT_SIZE = 224;                                 // matches your code

let videoEl = null;
let running = false;
let frame = 0;

// Prefer WebGL → WASM → CPU (doesn't change math)
async function selectBackend() {
  try { await tf.setBackend('webgl'); } catch (_) {}
  if (tf.getBackend() !== 'webgl') {
    try { await tf.setBackend('wasm'); } catch (_) {}
  }
  await tf.ready();
  const b = document.getElementById('backend');
  if (b) b.textContent = `backend: ${tf.getBackend()}`;
}

async function setupCamera() {
  const webcamElement = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
  });
  webcamElement.srcObject = stream;
  await new Promise((resolve) => (webcamElement.onloadedmetadata = () => resolve(webcamElement)));
  await webcamElement.play();
  videoEl = webcamElement;
  return webcamElement;
}

async function loadModel() {
  model = await tf.loadLayersModel('tfjs_model/model.json');
  // Warmup (compiles kernels/shaders, avoids first-frame hiccup)
  tf.tidy(() => {
    const z = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    const y = model.predict(z);
    if (Array.isArray(y)) y.forEach(t => t.dispose()); else y.dispose?.();
  });
  console.log("Model loaded");
}

// EXACT same preprocessing & math you used originally:
// fromPixels(video) -> resizeNearestNeighbor([224,224]) -> toFloat() -> div(255.0) -> expandDims()
function makeInputFromVideo(video) {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(video)
      .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE]) // default alignCorners=false
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims(); // [1, H, W, 3]
    return tensor;
  });
}

function renderPanel(predArray) {
  // Build the same string as your original in #result (for consistency)
  const resultP = document.getElementById('result');
  let text = '';
  for (let i = 0; i < classLabels.length; i++) {
    text += `${classLabels[i]}: ${(predArray[i] * 100).toFixed(2)}%\n`;
  }
  resultP.innerText = text.trimEnd();

  // Also render the vertical list with top class highlighted
  const items = classLabels.map((label, i) => ({ label, p: predArray[i] ?? 0 }));
  let bestIdx = 0, bestVal = -Infinity;
  for (let i = 0; i < items.length; i++) if (items[i].p > bestVal) { bestVal = items[i].p; bestIdx = i; }

  const predsEl = document.getElementById('predictions');
  predsEl.innerHTML = items.map((it, i) => `
    <div class="row ${i === bestIdx ? 'best' : ''}">
      <div>${it.label}</div>
      <div>${(it.p * 100).toFixed(1)}%</div>
    </div>
  `).join('');
}

async function predictLoop() {
  if (!running) return;
  await tf.nextFrame(); // keep UI responsive
  frame++;

  try {
    const input = makeInputFromVideo(videoEl);
    const prediction = model.predict(input);
    const predArray = await prediction.data(); // raw values from your model (already probabilities in your setup)
    renderPanel(predArray);

    tf.dispose([input, prediction]);
  } catch (e) {
    console.error(e);
    const errEl = document.getElementById('err');
    if (errEl) errEl.textContent = String(e?.message || e);
  }

  requestAnimationFrame(predictLoop);
}

function stop() {
  running = false;
  const s = document.getElementById('status');
  if (s) s.textContent = '停止中';
}

function disposeAll() {
  stop();
  if (model) { model.dispose(); model = undefined; }
  if (videoEl && videoEl.srcObject) {
    for (const track of videoEl.srcObject.getVideoTracks()) track.stop();
    videoEl.srcObject = null;
  }
  tf.engine().disposeVariables();
  tf.backend().dispose?.();
}

// Start/Stop buttons (optional)
document.getElementById('startBtn')?.addEventListener('click', async () => {
  document.getElementById('err').textContent = '';
  if (!videoEl) await setupCamera();
  if (!model) await loadModel();
  running = true;
  const s = document.getElementById('status'); if (s) s.textContent = '推論中…';
  requestAnimationFrame(predictLoop);
});

document.getElementById('stopBtn')?.addEventListener('click', stop);

// Auto-start (same behavior as your original)
window.addEventListener('DOMContentLoaded', async () => {
  try {
    await selectBackend();
    await setupCamera();
    await loadModel();
    running = true;
    const s = document.getElementById('status'); if (s) s.textContent = '推論中…';
    requestAnimationFrame(predictLoop);
  } catch (e) {
    console.error(e);
    const errEl = document.getElementById('err');
    if (errEl) errEl.textContent = '初期化エラー: ' + (e?.message || e);
  }
});

// Clean up on exit
window.addEventListener('pagehide', disposeAll);
window.addEventListener('beforeunload', disposeAll);
