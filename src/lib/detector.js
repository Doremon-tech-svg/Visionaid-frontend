// Client-side inference — no server round-trip, works offline.
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const MODEL_URL = '/yolov8n.onnx';
const INPUT_SIZE = 640;
const CONF_THRESH = 0.4;
const IOU_THRESH = 0.45;

const COCO_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

// ---------------------------------------------------------------------------
// OUTDOOR MODE
// ---------------------------------------------------------------------------
// COCO (what yolov8n.onnx was trained on) has real classes for car, bicycle,
// motorcycle, bus, truck, person, traffic light, stop sign, fire hydrant,
// bench — all genuinely outdoor-relevant, kept below.
// COCO has NO "pothole", "wall", or "tree" class — no pretrained model
// ships with those, so faking a detection for them would just be wrong
// data pretending to be sensed data. Two honest options, both wired in:
//   1) OUTDOOR_PRIORITY below re-sorts/boosts the real classes that matter
//      most for walking outdoors, so cars/bikes never get buried under
//      indoor clutter classes.
//   2) A SECOND, optional model slot (loadPotholeDetector below) for a
//      pothole/road-damage detector. Roboflow Universe has several free
//      pretrained "pothole detection" YOLOv8 models you can export to
//      .onnx (opset 12) and drop in as /public/pothole.onnx — the app
//      will pick it up automatically and merge its boxes in as
//      class: "pothole". Walls/trees are static and huge in frame; the
//      cleanest fix for those is the same export path with a small
//      "path-obstruction" dataset — genuinely out of scope to fake here.
// ---------------------------------------------------------------------------
const OUTDOOR_PRIORITY = new Set([
  "person","car","bicycle","motorcycle","bus","truck","traffic light",
  "stop sign","fire hydrant","bench","dog",
]);

const REAL_HEIGHT_M = {
  person:1.7, chair:0.9, couch:0.85, "dining table":0.75, bottle:0.25, cup:0.12,
  laptop:0.25, tv:0.55, "cell phone":0.15, book:0.22, backpack:0.45, handbag:0.30,
  dog:0.5, cat:0.3, car:1.5, bicycle:1.05, motorcycle:1.25, clock:0.3, remote:0.15,
  umbrella:0.9,
  // outdoor additions
  bus:3.0, truck:2.6, "traffic light":3.2, "stop sign":2.1, "fire hydrant":0.65, bench:0.85,
};
const DEFAULT_HEIGHT_M = 0.5;

let FOCAL_PX = Number(localStorage.getItem('visionaid_focal_px')) || 700;
export function setFocalPx(px) { FOCAL_PX = px; localStorage.setItem('visionaid_focal_px', String(px)); }
export function getFocalPx() { return FOCAL_PX; }

let session = null;
let potholeSession = null;
const POTHOLE_MODEL_URL = '/pothole.onnx'; // optional, see note above

export async function loadDetector() {
  if (session) return session;
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    return session;
  } catch (e) {
    console.error('On-device model failed to load:', e.message);
    console.error('Check: (1) yolov8n.onnx exists in frontend/public/, (2) it was exported with opset=12, (3) network tab shows the fetch for /yolov8n.onnx succeeding.');
    throw e;
  }
}

// Best-effort — silently no-ops if pothole.onnx isn't present. Call this once
// after loadDetector() resolves; check isPotholeDetectorReady() before use.
export async function loadPotholeDetector() {
  if (potholeSession) return potholeSession;
  try {
    potholeSession = await ort.InferenceSession.create(POTHOLE_MODEL_URL, { executionProviders: ['wasm'] });
    return potholeSession;
  } catch {
    console.warn('No pothole.onnx found in /public — pothole/road-damage detection disabled. This is optional; the app works fine without it.');
    return null;
  }
}
export function isPotholeDetectorReady() { return potholeSession !== null; }
export function isDetectorReady() { return session !== null; }

function preprocess(videoEl) {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d');

  const scale = Math.min(INPUT_SIZE / videoEl.videoWidth, INPUT_SIZE / videoEl.videoHeight);
  const nw = videoEl.videoWidth * scale;
  const nh = videoEl.videoHeight * scale;
  const dx = (INPUT_SIZE - nw) / 2;
  const dy = (INPUT_SIZE - nh) / 2;

  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(videoEl, dx, dy, nw, nh);

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const area = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < area; i++) {
    float32[i] = data[i * 4] / 255;
    float32[area + i] = data[i * 4 + 1] / 255;
    float32[2 * area + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]), scale, dx, dy };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

function nms(boxes) {
  boxes.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const box of boxes) {
    if (!kept.some(k => k.cls === box.cls && iou(k.bbox, box.bbox) > IOU_THRESH)) kept.push(box);
  }
  return kept;
}

function position(cx, w) {
  const t = w / 3;
  return cx < t ? 'left' : cx < 2 * t ? 'center' : 'right';
}
function proximity(area, frameArea) {
  const r = area / frameArea;
  return r > 0.25 ? 'very close' : r > 0.08 ? 'near' : 'far';
}

function postprocess(output, scale, dx, dy, frameW, frameH) {
  const data = output.data;
  const numBoxes = output.dims[2];
  const boxes = [];

  for (let i = 0; i < numBoxes; i++) {
    let bestCls = -1, bestScore = 0;
    for (let c = 0; c < 80; c++) {
      const s = data[(4 + c) * numBoxes + i];
      if (s > bestScore) { bestScore = s; bestCls = c; }
    }
    if (bestScore < CONF_THRESH) continue;

    const cx = data[i], cy = data[numBoxes + i], bw = data[2 * numBoxes + i], bh = data[3 * numBoxes + i];
    const x1 = (cx - bw / 2 - dx) / scale;
    const y1 = (cy - bh / 2 - dy) / scale;
    const x2 = (cx + bw / 2 - dx) / scale;
    const y2 = (cy + bh / 2 - dy) / scale;

    boxes.push({ cls: bestCls, confidence: bestScore, bbox: [x1, y1, x2, y2] });
  }

  const frameArea = frameW * frameH;
  return nms(boxes).slice(0, 14).map(b => {
    const className = COCO_CLASSES[b.cls] || 'object';
    const boxH = Math.max(1, b.bbox[3] - b.bbox[1]);
    const realH = REAL_HEIGHT_M[className] ?? DEFAULT_HEIGHT_M;
    const distance_m = Math.round((realH * FOCAL_PX) / boxH * 10) / 10;
    const boxArea = Math.max(0, b.bbox[2] - b.bbox[0]) * Math.max(0, b.bbox[3] - b.bbox[1]);
    const cx = (b.bbox[0] + b.bbox[2]) / 2;

    return {
      class: className,
      confidence: Math.round(b.confidence * 1000) / 1000,
      bbox: b.bbox,
      position: position(cx, frameW),
      proximity: proximity(boxArea, frameArea),
      distance_m,
      outdoorPriority: OUTDOOR_PRIORITY.has(className),
    };
  });
}

// Ground-plane hazard (pothole model output is a single class, no known real
// height — distance can't use the height-ratio trick above). Approximation:
// lower in frame = closer (standard ground-plane perspective), scaled 0–15m.
// This is a rough estimate for a demo, not survey-grade — label it as such.
function postprocessGroundHazard(output, scale, dx, dy, frameW, frameH) {
  const data = output.data;
  const numBoxes = output.dims[2];
  const numCls = output.dims[1] - 4;
  const boxes = [];

  for (let i = 0; i < numBoxes; i++) {
    let bestCls = 0, bestScore = 0;
    for (let c = 0; c < numCls; c++) {
      const s = data[(4 + c) * numBoxes + i];
      if (s > bestScore) { bestScore = s; bestCls = c; }
    }
    if (bestScore < CONF_THRESH) continue;
    const cx = data[i], cy = data[numBoxes + i], bw = data[2 * numBoxes + i], bh = data[3 * numBoxes + i];
    const x1 = (cx - bw / 2 - dx) / scale, y1 = (cy - bh / 2 - dy) / scale;
    const x2 = (cx + bw / 2 - dx) / scale, y2 = (cy + bh / 2 - dy) / scale;
    boxes.push({ cls: bestCls, confidence: bestScore, bbox: [x1, y1, x2, y2] });
  }

  return nms(boxes).slice(0, 6).map(b => {
    const cx = (b.bbox[0] + b.bbox[2]) / 2;
    const yBottomNorm = Math.min(1, b.bbox[3] / frameH);
    const distance_m = Math.round(Math.max(0.5, 15 * (1 - yBottomNorm)) * 10) / 10;
    const boxArea = Math.max(0, b.bbox[2] - b.bbox[0]) * Math.max(0, b.bbox[3] - b.bbox[1]);
    return {
      class: 'pothole',
      confidence: Math.round(b.confidence * 1000) / 1000,
      bbox: b.bbox,
      position: position(cx, frameW),
      proximity: proximity(boxArea, frameW * frameH),
      distance_m,
      outdoorPriority: true,
      approximate: true,
    };
  });
}

export async function detectLocal(videoEl) {
  if (!session) throw new Error('Detector not loaded — call loadDetector() first');
  const { tensor, scale, dx, dy } = preprocess(videoEl);
  const results = await session.run({ images: tensor });
  const output = results[Object.keys(results)[0]];
  let dets = postprocess(output, scale, dx, dy, videoEl.videoWidth, videoEl.videoHeight);

  if (potholeSession) {
    try {
      const { tensor: t2, scale: s2, dx: d2, dy: y2 } = preprocess(videoEl);
      const r2 = await potholeSession.run({ images: t2 });
      const out2 = r2[Object.keys(r2)[0]];
      dets = dets.concat(postprocessGroundHazard(out2, s2, d2, y2, videoEl.videoWidth, videoEl.videoHeight));
    } catch (e) {
      console.warn('Pothole model inference failed:', e.message);
    }
  }

  // outdoor-relevant + closer objects surface first
  return dets.sort((a, b) => (b.outdoorPriority - a.outdoorPriority) || (a.distance_m - b.distance_m));
}

const CLASS_HI = {
  person:"व्यक्ति", chair:"कुर्सी", bottle:"बोतल", cup:"कप", laptop:"लैपटॉप", "cell phone":"मोबाइल फोन",
  book:"किताब", "dining table":"मेज़", door:"दरवाज़ा", tv:"टीवी", dog:"कुत्ता", cat:"बिल्ली", car:"कार",
  bicycle:"साइकिल", motorcycle:"मोटरसाइकिल", backpack:"बैग", handbag:"हैंडबैग", clock:"घड़ी", remote:"रिमोट",
  couch:"सोफ़ा", bus:"बस", truck:"ट्रक", "traffic light":"ट्रैफिक लाइट", "stop sign":"स्टॉप साइन",
  "fire hydrant":"फायर हाइड्रेंट", bench:"बेंच", pothole:"गड्ढा",
};
const POSITION_PHRASE = {
  en: { left: "to the left", center: "in front", right: "to the right" },
  hi: { left: "बाईं ओर", center: "सामने", right: "दाईं ओर" },
};

export function generateDescription(detections, lang = "en") {
  lang = lang === "hi" ? "hi" : "en";
  if (!detections.length) {
    return lang === "hi" ? "आस-पास कुछ स्पष्ट नहीं दिखा। धीरे चलें।" : "No objects detected clearly. Please move slowly.";
  }
  const parts = detections.slice(0, 5).map((d) => {
    const posWord = POSITION_PHRASE[lang][d.position];
    if (lang === "hi") {
      const cls = CLASS_HI[d.class] || d.class;
      return `${cls} ${posWord}, लगभग ${d.distance_m} मीटर दूर`;
    }
    const article = /^[aeiou]/i.test(d.class) ? "an" : "a";
    return `${article} ${d.class} ${posWord}, about ${d.distance_m} meters away`;
  });
  return lang === "hi" ? parts.join("; ") + "।" : "There is " + parts.join(", and ") + ".";
}