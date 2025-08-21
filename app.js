// app.js - PaddleOCR det 模型測試版

// 安全保護
if (window.ort) {
  // 讓 ORT 自己去 CDN 抓對應的 .mjs/.wasm
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
} else {
  console.warn("onnxruntime-web 未載入");
}
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const ocrBtn = document.getElementById("ocrBtn");
const result = document.getElementById("result");
let detSession = null;
let imageElement = null;

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    imageElement = preview;
  };
  reader.readAsDataURL(file);
});

ocrBtn.addEventListener("click", async () => {
  if (!imageElement) {
    alert("請先上傳照片");
    return;
  }
  result.innerText = "🔄 載入模型中...";
  try {
    if (!detSession) {
      detSession = await ort.InferenceSession.create(
        "https://github.com/claire826086/contract-ocr-static/releases/download/v0.1/det.onnx",
        { executionProviders: ['wasm'] } // 只用 WASM，避免 webgpu/webnn 嘗試
      );
    }
    result.innerText = "✅ 模型載入完成，開始推論...";
    const inputTensor = imageToTensor(imageElement);
    const feeds = { "x": inputTensor };
    const outputs = await detSession.run(feeds);
    console.log("Det 模型輸出:", outputs);
    result.innerText = "✅ 推論完成！請打開 Console 查看輸出結果。";
  } catch (err) {
    console.error(err);
    result.innerText = "❌ 錯誤：" + err.message;
  }
});

function imageToTensor(img) {
  const targetSize = 640;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = targetSize;
  canvas.height = targetSize;
  ctx.drawImage(img, 0, 0, targetSize, targetSize);
  const imageData = ctx.getImageData(0, 0, targetSize, targetSize).data;
  const float32 = new Float32Array(targetSize * targetSize * 3);
  for (let i = 0; i < targetSize * targetSize; i++) {
    float32[i * 3 + 0] = imageData[i * 4 + 0] / 255;
    float32[i * 3 + 1] = imageData[i * 4 + 1] / 255;
    float32[i * 3 + 2] = imageData[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", float32, [1, 3, targetSize, targetSize]);
}
