// app.js - PaddleOCR det 模型（det-only）測試版

// === ORT 設定：非 SIMD、單執行緒、避免 threaded 變體 ===
if (window.ort) {
  // 明確鎖定非 SIMD 的 .mjs/.wasm，避免裝置不支援造成簽章錯誤
  ort.env.wasm.wasmPaths = {
    wasm: "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm",
    mjs:  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.jsep.mjs"
  };
  ort.env.wasm.numThreads = 1;   // iOS 無 threads，先關閉
  ort.env.wasm.proxy = false;    // 可選：避免走 worker/代理路徑
} else {
  console.warn("onnxruntime-web 未載入（請確認 index.html 先載入 ort.min.js 再載入 app.js）");
}

// === DOM ===
const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn");
const result    = document.getElementById("result");

// === 狀態 ===
let detSession = null;
let imageElement = null;

// 你的模型（放在 GitHub Pages）
const DET_URL = "https://claire826086.github.io/contract-ocr-static/models/det.onnx";

// === 檔案選擇與預覽（確保 onload 後再設置 imageElement） ===
fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.onload = () => {
      imageElement = preview;
      result.innerText = "✅ 圖片已載入";
    };
  };
  reader.readAsDataURL(file);
});

// === 開始 OCR（det 測試：只跑推論，輸出請看 Console） ===
ocrBtn.addEventListener("click", async () => {
  if (!imageElement) {
    alert("請先上傳照片");
    return;
  }
  if (!window.ort) {
    result.innerText = "❌ 無法載入 onnxruntime-web（請檢查 <script src> 與 CSP）";
    return;
  }

  result.innerText = "🔄 載入模型中...";
  try {
    if (!detSession) {
      detSession = await ort.InferenceSession.create(
        DET_URL,
        { executionProviders: ['wasm'] } // 僅使用 WASM 後端
      );
    }

    result.innerText = "✅ 模型載入完成，開始推論...";
    const inputTensor = imageToTensor(imageElement);

    // 自動對齊模型的輸入名稱（避免硬寫 "x" 不相符）
    const feeds = {};
    feeds[detSession.inputNames[0]] = inputTensor;

    const outputs = await detSession.run(feeds);
    console.log("Det 模型輸出:", outputs);
    result.innerText = "✅ 推論完成！請開啟 Console 查看輸出。";
  } catch (err) {
    console.error(err);
    result.innerText = "❌ 錯誤：" + (err?.message || err);
  }
});

// === 影像轉 Tensor（簡化：等比縮放到 640x640，RGB/255） ===
function imageToTensor(img) {
  const target = 640;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = target;
  canvas.height = target;

  // 將圖等比縮放置中到 640x640（避免嚴重形變）
  const scale = Math.min(target / img.naturalWidth, target / img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const dx = Math.floor((target - w) / 2);
  const dy = Math.floor((target - h) / 2);

  ctx.fillStyle = "#000"; // 背景填黑
  ctx.fillRect(0, 0, target, target);
  ctx.drawImage(img, dx, dy, w, h);

  const data = ctx.getImageData(0, 0, target, target).data;
  const f32 = new Float32Array(target * target * 3);
  for (let i = 0; i < target * target; i++) {
    f32[i * 3 + 0] = data[i * 4 + 0] / 255; // R
    f32[i * 3 + 1] = data[i * 4 + 1] / 255; // G
    f32[i * 3 + 2] = data[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor("float32", f32, [1, 3, target, target]);
}
