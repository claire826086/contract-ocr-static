// app.js - PaddleOCR det 模型測試（多前處理自動嘗試 + 進度條 + 結果可視化）

// === ORT 設定（CDN 直連 .mjs/.wasm；iOS 關閉 threads） ===
if (window.ort) {
  ort.env.wasm.wasmPaths = {
    mjs:  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.mjs",
    wasm: "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.wasm",
  };
  ort.env.wasm.numThreads = 1;
} else {
  console.warn("onnxruntime-web 未載入");
}

// === DOM ===
const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn");
const result    = document.getElementById("result");

// 進度條
const progressBar = document.createElement("progress");
progressBar.max = 100;
progressBar.value = 0;
progressBar.style.width = "100%";
progressBar.style.display = "none";
result.insertAdjacentElement("beforebegin", progressBar);

// === 狀態 ===
let detSession = null;
let imageElement = null;

// 你的 det 模型（放在 GitHub Pages）
const DET_URL = "https://claire826086.github.io/contract-ocr-static/models/det.onnx";

// === 圖片選擇 + 預覽 ===
fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.onload = () => {
      imageElement = preview;
      result.textContent = "✅ 圖片已載入";
    };
  };
  reader.readAsDataURL(file);
});

// === 下載模型（顯示進度） ===
async function fetchWithProgress(url) {
  const resp = await fetch(url, { cache: "force-cache" });
  if (!resp.ok) throw new Error("下載失敗：" + resp.status);

  const total = +resp.headers.get("Content-Length");
  const reader = resp.body.getReader();
  let received = 0;
  const chunks = [];

  progressBar.style.display = "block";
  progressBar.value = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) {
      progressBar.value = Math.round((received / total) * 100);
    } else {
      progressBar.removeAttribute("value");
    }
  }

  progressBar.style.display = "none";

  // 合併 chunks → ArrayBuffer
  const size = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out.buffer;
}

// === 核心：多種前處理嘗試 ===
const PREPROCESS_VARIANTS = [
  { name: "RGB_ImageNet",  space: "RGB", mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225] },
  { name: "BGR_ImageNet",  space: "BGR", mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225] },
  { name: "RGB_div255",    space: "RGB", mean: [0,0,0],             std: [1,1,1]             },
  { name: "BGR_div255",    space: "BGR", mean: [0,0,0],             std: [1,1,1]             },
  { name: "RGB_0.5norm",   space: "RGB", mean: [0.5,0.5,0.5],       std: [0.5,0.5,0.5]       },
  { name: "BGR_0.5norm",   space: "BGR", mean: [0.5,0.5,0.5],       std: [0.5,0.5,0.5]       },
];

const TARGET_SIZE = 640;                  // 你的 det 模型輸入大小
const PASS_THRESHOLD = 0.2;               // 判定「不是全 0」的門檻（可視需求調整）
const SHOW_FIRST_N = 100;                 // 顯示前 N 個輸出數值

function makeInputTensor(img, variant) {
  const target = TARGET_SIZE;

  // 等比縮放 + letterbox
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = target;
  canvas.height = target;

  const scale = Math.min(target / img.naturalWidth, target / img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const dx = Math.floor((target - w) / 2);
  const dy = Math.floor((target - h) / 2);

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, target, target);
  ctx.drawImage(img, dx, dy, w, h);

  const data = ctx.getImageData(0, 0, target, target).data; // RGBA
  const out = new Float32Array(3 * target * target);

  const mean = variant.mean;
  const std  = variant.std;
  const RGB = (variant.space === "RGB");

  // HWC → CHW
  let p = 0;
  for (let y = 0; y < target; y++) {
    for (let x = 0; x < target; x++) {
      const i = (y * target + x) * 4;
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      const c0 = RGB ? r : b;
      const c2 = RGB ? b : r;

      out[0 * target * target + p] = (c0 - mean[0]) / std[0];
      out[1 * target * target + p] = (g  - mean[1]) / std[1];
      out[2 * target * target + p] = (c2 - mean[2]) / std[2];
      p++;
    }
  }
  return new ort.Tensor("float32", out, [1, 3, target, target]);
}

function summarizeTensor(tensor) {
  const arr = tensor.data;
  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / arr.length;
  return { min, max, mean };
}

function vizDetMap(tensor, whereEl) {
  const [n, c, h, w] = tensor.dims;
  const arr = tensor.data; // 期望長度 h*w
  const viz = document.createElement("canvas");
  viz.width = w; viz.height = h;
  const ctx2 = viz.getContext("2d");
  const imgData = ctx2.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    let v = arr[i];
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    const g = Math.round(v * 255);
    imgData.data[i*4 + 0] = g;
    imgData.data[i*4 + 1] = g;
    imgData.data[i*4 + 2] = g;
    imgData.data[i*4 + 3] = 255;
  }
  ctx2.putImageData(imgData, 0, 0);
  whereEl.insertAdjacentElement("afterend", viz);
  return viz;
}

ocrBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
  if (!window.ort) {
    result.textContent = "❌ 無法載入 onnxruntime-web（請檢查 <script src> 與 CSP）";
    return;
  }

  result.textContent = "🔄 載入模型中...";
  try {
    if (!detSession) {
      const buffer = await fetchWithProgress(DET_URL);
      detSession = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
    }

    // 嘗試多種前處理
    let best = null; // {variantName, outputs, stats}
    for (const variant of PREPROCESS_VARIANTS) {
      const inputTensor = makeInputTensor(imageElement, variant);
      const feeds = {};
      feeds[detSession.inputNames[0]] = inputTensor;

      const outputs = await detSession.run(feeds);
      const firstName = Object.keys(outputs)[0];
      const detMap = outputs[firstName];
      const stats = summarizeTensor(detMap);

      // 顯示每次嘗試的摘要（方便你在頁面觀察）
      console.log(`[${variant.name}] min=${stats.min} max=${stats.max} mean=${stats.mean}`);
      if (!best || stats.max > (best.stats?.max ?? -Infinity)) {
        best = { variantName: variant.name, outputs, stats, firstName };
      }
      // 一旦過門檻就接受
      if (stats.max >= PASS_THRESHOLD) {
        best.hit = true;
        break;
      }
    }

    if (!best) {
      result.textContent = "❌ 未知錯誤：沒有任何輸出";
      return;
    }

    // 顯示最好的那次
    const detMap = best.outputs[best.firstName];
    const arr = detMap.data;
    const head = Array.from(arr.slice(0, SHOW_FIRST_N));
    result.textContent =
      `前處理方案：${best.variantName}  ${best.hit ? "(命中門檻)" : "(未達門檻，已選最大值)"}\n` +
      `輸出名稱: ${best.firstName}\n` +
      `形狀: [${detMap.dims.join(", ")}], 型別: ${detMap.type}\n` +
      `min=${best.stats.min.toFixed(6)}, max=${best.stats.max.toFixed(6)}, mean=${best.stats.mean.toFixed(6)}\n\n` +
      `前 ${SHOW_FIRST_N} 筆:\n` +
      JSON.stringify(head, null, 2);

    // 可視化灰階圖
    vizDetMap(detMap, preview);

  } catch (err) {
    console.error(err);
    result.textContent = "❌ 錯誤：" + (err?.message || err);
    progressBar.style.display = "none";
  }
});
