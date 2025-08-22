// app.js - PaddleOCR det 模型測試版（CDN 直連 wasm/mjs）

if (window.ort) {
  // 指定 cdnjs 的 .mjs / .wasm
  ort.env.wasm.wasmPaths = {
    mjs:  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.mjs",
    wasm: "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.wasm"
  };
  ort.env.wasm.numThreads = 1;   // iOS 不支援多執行緒
} else {
  console.warn("onnxruntime-web 未載入");
}

const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn");
const result    = document.getElementById("result");

let detSession = null;
let imageElement = null;

// 你的模型放在 GitHub Pages
const DET_URL = "https://claire826086.github.io/contract-ocr-static/models/det.onnx";

// 圖片選擇 + 預覽
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

// 按下「開始 OCR」
ocrBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
  if (!window.ort) {
    result.innerText = "❌ 無法載入 onnxruntime-web";
    return;
  }

  result.innerText = "🔄 載入模型中...";
  try {
    if (!detSession) {
      detSession = await ort.InferenceSession.create(
        DET_URL,
        { executionProviders: ['wasm'] }
      );
    }

    result.innerText = "✅ 模型載入完成，開始推論...";
    const inputTensor = imageToTensor(imageElement);

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

// 把圖片轉成 Tensor
function imageToTensor(img) {
  const target = 640;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = target; canvas.height = target;

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
    f32[i*3+0] = data[i*4+0] / 255;
    f32[i*3+1] = data[i*4+1] / 255;
    f32[i*3+2] = data[i*4+2] / 255;
  }
  return new ort.Tensor("float32", f32, [1, 3, target, target]);
}
