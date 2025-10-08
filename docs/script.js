// ====== CONFIG (edit these to match your model) ======
const MODEL_URL    = 'tfjs_model/model.json';          // path to your TF.js model
const CLASS_LABELS = ['Big Lot', 'C Press', 'Snyders']; // MUST match training order
const INPUT_SIZE   = 224;                               // model input size, e.g., 224

// If your training used extra preprocessing, flip these (default: same as your original – none):
const DIVIDE_BY_255 = false;  // true ONLY if you trained with images scaled to 0–1
const RGB_TO_BGR    = false;  // true ONLY if you trained with BGR (OpenCV) order

let model, video, running = false, frame = 0;

// Prefer WebGL → WASM → CPU (doesn't change math)
async function selectBackend() {
  try { await tf.setBackend('webgl'); } catch (_) {}
  if (tf.getBackend() !== 'webgl') {
    try { await tf.setBackend('wasm'); } catch (_) {}
  }
  await tf.ready();
}

async function setupCamera() {
  video = document.getElementById('webcam');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false
  });
  video.srcObject = stream;
  await new Promise(res => (video.onloadedmetadata = () => res()));
  await video.play();
  return video;
}

// EXACT pipeline like your original:
// fromPixels(video) -> resizeNearestNeighbor -> toFloat -> (optional BGR,/255) -> expandDims
function preprocessFromVideo(video) {
  return tf.tidy(() => {
    let img = tf.browser.fromPixels(video);                   // [H,W,3] uint8 RGB
    img = tf.image.resizeNearestNeighbor(img, [INPUT_SIZE, INPUT_SIZE]); // alignCorners=false by default
    let f = img.toFloat();                                     // [H,W,3] float32 (0..255)
    if (RGB_TO_BGR) f = f.reverse(-1);                         // optional channel swap
    if (DIVIDE_BY_255) f = f.div(255);                         // optional scaling
    return f.expandDims(0);                                    // [1,H,W,3]
  });
}

// Get raw outputs as-is (no softmax/normalization)
async function getRawOutputs(tensorOrArray) {
  const t = Array.isArray(tensorOrArray) ? tensorOrArray[0] : tensorOrArray; // [1,C] or [C]
  const flat = t.rank === 2 ? t.squeeze([0]) : t; // [C]
  const arr = await flat.data();                  // raw numbers
  if (t !== flat) flat.dispose();
  return arr;
}

function renderResult(values) {
  const items = CLASS_LABELS.map((label, i) => ({ label, v: values[i] ?? 0 }));
  let bestIdx = 0, bestVal = -Infinity;
  for (let i = 0; i < items.length; i++) if (items[i].v > bestVal) { bestVal = items[i].v; bestIdx = i; }

  // Keep your original style: show label + “%” (purely visual; math unchanged)
  const best = items[bestIdx];
  document.getElementById('result').textContent =
    `${best.label}（${(best.v * 100).toFixed(1)}%）`;
}

// Warm up once to compile kernels/shaders
async function loadModel() {
  model = await tf.loadLayersModel(MODEL_URL);
  tf.tidy(() => {
    const z = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    const y = model.predict(z);
    if (Array.isArray(y)) y.forEach(t => t.dispose()); else y.dispose?.();
  });
}

async function loop() {
  if (!running) return;
  await tf.nextFrame(); // keep UI responsive
  frame++;

  try {
    const input = preprocessFromVideo(video);
    const out = model.predict(input);
    const values = await getRawOutputs(out); // RAW
    input.dispose();
    if (Array.isArray(out)) out.forEach(t => t.dispose()); else out.dispose?.();

    // Update UI (every frame or throttle if you like)
    renderResult(values);
  } catch (e) {
    console.error(e);
    document.getElementById('result').textContent = 'エラー: ' + (e?.message || e);
  }

  requestAnimationFrame(loop);
}

function stop() {
  running = false;
}

function disposeAll() {
  stop();
  if (model) { model.dispose(); model = undefined; }
  if (video && video.srcObject) {
    for (const track of video.srcObject.getVideoTracks()) track.stop();
    video.srcObject = null;
  }
  tf.engine().disposeVariables();
  tf.backend().dispose?.();
}

// Auto-start on load (same behavior as your original)
window.addEventListener('DOMContentLoaded', async () => {
  await selectBackend();
  await setupCamera();
  await loadModel();
  running = true;
  requestAnimationFrame(loop);
});

// Clean up on page exit
window.addEventListener('pagehide', disposeAll);
window.addEventListener('beforeunload', disposeAll);
