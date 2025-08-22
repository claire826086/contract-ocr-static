// app.js - PaddleOCR det 模型測試版（CDN 直連 + 進度條 + 完整輸出）

// === ORT 設定：指定 cdnjs 的 .mjs/.wasm；iOS 關閉多執行緒 ===
if (window.ort) {
  ort.env.wasm.wasmPaths = {
    mjs:  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.mjs",
    wasm: "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.wasm",
  };
  ort.env.wasm.numThreads = 1; // iOS 不支援 threads
} else {
  console.warn("onnxruntime-web 未載入");
}

// === DOM ===
const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn");
const result    = document.getElementById("result");

// === 進度條（放在 result 之前） ===
const progressBar = document.createElement("progress");
progressBar.max = 100;
progressBar.value = 0;
progressBar.style.width = "100%";
progressBar.style.display = "none";
result.insertAdjacentElement("beforebegin", progressBar);

// === 狀態 ===
let detSession = null;
let imageElement = null;

// 你的模型（GitHub Pages，避免 CORS）
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

  const total = +resp.headers.get("Content-Length"); // 有些伺服器不回這個
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
      // 沒有 Content-Length 時，顯示不確定狀態
      progressBar.removeAttribute("value");
    }
  }

  progressBar.style.display = "none";
  // 合併 chunks
  const size = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out.buffer;
}

// === 按下「開始 OCR」 ===
ocrBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
  if (!window.ort) {
    result.textContent = "❌ 無法載入 onnxruntime-web（請檢查 <script src> 與 CSP）";
    return;
  }

  result.textContent = "🔄 載入模型中...";
  try {
    if (!detSession) {
      // 先把模型下載到 ArrayBuffer（顯示進度），再交給 ORT
      const modelBuffer = await fetchWithProgress(DET_URL);
      detSession = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
      });
    }

    result.textContent = "✅ 模型載入完成，開始推論...";
    const inputTensor = imageToTensor(imageElement);

    // 自動對齊模型輸入名稱
    const feeds = {};
    feeds[detSession.inputNames[0]] = inputTensor;

    const outputs = await detSession.run(feeds);

    // === 將結果「完整」顯示到頁面 ===
    // outputs 是一個 Map-like 物件：{name: ort.Tensor}
    const printable = {};
    for (const [name, tensor] of Object.entries(outputs)) {
      // 注意：完整 data 可能很大；你要求完整，我就全數轉成陣列
      printable[name] = {
        dims: tensor.dims,
        type: tensor.type,
        size: tensor.data.length,
        data: Array.from(tensor.data), // 這會很大，請知悉
      };
    }
    result.textContent = JSON.stringify(printable, null, 2);
  } catch (err) {
    console.error(err);
    result.textContent = "❌ 錯誤：" + (err?.message || err);
    progressBar.style.display = "none";
  }
});

// === 圖片轉 Tensor（640x640，RGB/255；等比置中，填黑） ===
function imageToTensor(img) {
  const target = 640;
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

  const data = ctx.getImageData(0, 0, target, target).data;
  const f32 = new Float32Array(target * target * 3);
  for (let i = 0; i < target * target; i++) {
    f32[i * 3 + 0] = data[i * 4 + 0] / 255;
    f32[i * 3 + 1] = data[i * 4 + 1] / 255;
    f32[i * 3 + 2] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", f32, [1, 3, target, target]);
}
