// app.js — 目前保留 det 測試（ONNX），並加一顆「全文辨識（Tesseract）」快速看到文字

// ====== ONNX Runtime（det 測試仍可用）======
if (window.ort) {
  ort.env.wasm.wasmPaths = {
    mjs:  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.mjs",
    wasm: "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.wasm",
  };
  ort.env.wasm.numThreads = 1;
} else {
  console.warn("onnxruntime-web 未載入");
}

// ====== DOM ======
const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn");     // det 測試
const result    = document.getElementById("result");

// 動態加一顆「全文辨識（Tesseract）」按鈕
const fastBtn = document.createElement("button");
fastBtn.textContent = "全文辨識（快速測試：Tesseract）";
fastBtn.style.marginTop = "8px";
ocrBtn.insertAdjacentElement("afterend", fastBtn);

// 進度條
const progressBar = document.createElement("progress");
progressBar.max = 100;
progressBar.value = 0;
progressBar.style.width = "100%";
progressBar.style.display = "none";
result.insertAdjacentElement("beforebegin", progressBar);

// ====== 狀態 ======
let detSession = null;
let imageElement = null;

// 你的 det 模型（GitHub Pages）
const DET_URL = "https://claire826086.github.io/contract-ocr-static/models/det.onnx";

// ====== 檔案選擇 + 預覽 ======
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

// ====== det 模型（仍可）======
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

  const size = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out.buffer;
}

function imageToTensorSimple(img) {
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

ocrBtn.addEventListener("click", async () => {
  // det 測試：仍然會顯示 det 的輸出數值
  if (!imageElement) return alert("請先上傳照片");
  if (!window.ort) {
    result.textContent = "❌ 無法載入 onnxruntime-web";
    return;
  }
  result.textContent = "🔄 載入 det 模型中...";
  try {
    if (!detSession) {
      const buffer = await fetchWithProgress(DET_URL);
      detSession = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
    }
    result.textContent = "✅ 模型載入完成，開始 det 推論...";
    const feeds = {};
    feeds[detSession.inputNames[0]] = imageToTensorSimple(imageElement);
    const outputs = await detSession.run(feeds);
    result.textContent = JSON.stringify(outputs, (k,v) => {
      if (v?.data instanceof Float32Array) {
        return { dims: v.dims, type: v.type, size: v.data.length, data: Array.from(v.data.slice(0,200)) };
      }
      return v;
    }, 2);
  } catch (err) {
    console.error(err);
    result.textContent = "❌ 錯誤：" + (err?.message || err);
    progressBar.style.display = "none";
  }
});

// ====== 「快速全文辨識」：Tesseract.js ======
fastBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
  if (!window.Tesseract) {
    return result.textContent = "❌ Tesseract.js 未載入";
  }
  result.textContent = "🔄 Tesseract 下載語言與模型中...";
  progressBar.style.display = "block";
  progressBar.removeAttribute("value");

  try {
    const { data } = await Tesseract.recognize(preview, 'eng', {
      logger: m => {
        // m.status: 'loading tesseract core' | 'initializing api' | 'recognizing text'...
        // m.progress: 0~1
        if (typeof m.progress === 'number') {
          progressBar.value = Math.round(m.progress * 100);
          progressBar.max = 100;
        }
      }
    });
    progressBar.style.display = "none";
    result.textContent = data.text.trim() || "(無辨識結果)";

    // 如果要中文：把 'eng' 換 'chi_tra'，但需 CSP 放行語言包主機；語言包較大、速度較慢
  } catch (err) {
    console.error(err);
    progressBar.style.display = "none";
    result.textContent = "❌ Tesseract 錯誤：" + (err?.message || err);
  }
});
