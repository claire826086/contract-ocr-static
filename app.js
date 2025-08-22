// app.js - 表格 OCR 快速版（Tesseract 繁中 + 前處理 + 粗略輸出 CSV）
// 仍保留 det(onxx) 測試按鈕；多了一顆「表格 OCR（Tesseract）」按鈕

// ====== ORT（det 測試仍可用；如不需要可刪）======
if (window.ort) {
  ort.env.wasm.wasmPaths = {
    mjs:  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.mjs",
    wasm: "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.wasm",
  };
  ort.env.wasm.numThreads = 1;
}

// ====== DOM ======
const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn");     // det 測試用
const result    = document.getElementById("result");

// 新增一顆「表格 OCR（Tesseract）」按鈕
const tableBtn = document.createElement("button");
tableBtn.textContent = "表格 OCR（Tesseract 繁中）";
tableBtn.style.marginTop = "8px";
ocrBtn.insertAdjacentElement("afterend", tableBtn);

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

// 你的 det 模型（如果保留 det 測試）
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

// ====== 共用：下載顯示進度（for det.onnx）======
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

// ====== 圖片前處理（灰階→二值→放大 2x）======
function preprocessForOCR(img) {
  // 以較大的工作寬度做 OCR（提升效果）。若影像已很大，可用 1.5x
  const scale = 2.0;
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);

  const c1 = document.createElement("canvas");
  c1.width = w; c1.height = h;
  const g1 = c1.getContext("2d");
  g1.drawImage(img, 0, 0, w, h);

  // 轉灰階
  const imageData = g1.getImageData(0, 0, w, h);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = (d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114);
    d[i] = d[i+1] = d[i+2] = gray;
  }

  // Otsu 門檻（簡化版）
  const hist = new Array(256).fill(0);
  for (let i = 0; i < d.length; i += 4) hist[d[i]|0]++;
  const total = w * h;
  let sum = 0; for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, wF = 0, varMax = 0, threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > varMax) { varMax = between; threshold = t; }
  }
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] < threshold ? 0 : 255;
    d[i] = d[i+1] = d[i+2] = v;
  }
  g1.putImageData(imageData, 0, 0);

  return c1; // 回傳處理後的 Canvas（Tesseract 支援傳 canvas）
}

// ====== 粗略把 words 聚成「行」並輸出 CSV（非常簡化，先好用為主）======
function wordsToCSV(words) {
  // 依 y 中心點排序
  words.sort((a, b) => (a.bbox.y0 + a.bbox.y1)/2 - (b.bbox.y0 + b.bbox.y1)/2);

  const rows = [];
  const yTolerance = 14; // 可調：行的合併門檻（像素，取決於放大倍率）

  for (const w of words) {
    const y = (w.bbox.y0 + w.bbox.y1)/2;
    let row = rows.find(r => Math.abs(r.y - y) <= yTolerance);
    if (!row) { row = { y, items: [] }; rows.push(row); }
    row.items.push(w);
  }

  // 每行內再依 x 排序，合成文字；多空白用逗號「猜一格」
  const lines = rows.map(r => {
    r.items.sort((a,b)=>a.bbox.x0 - b.bbox.x0);
    // 粗略以「相鄰字間距」估計是否插入逗號（表格欄位）
    const cells = [];
    let lastX1 = null, buf = "";
    for (const w of r.items) {
      if (lastX1 != null) {
        const gap = w.bbox.x0 - lastX1;
        if (gap > 20) { // 門檻可調
          cells.push(buf.trim());
          buf = "";
        } else {
          buf += " ";
        }
      }
      buf += w.text;
      lastX1 = w.bbox.x1;
    }
    if (buf.trim()) cells.push(buf.trim());
    return cells.join(",");
  });

  return lines.join("\n");
}

// ====== 表格 OCR（Tesseract 繁中）======
tableBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
  if (!window.Tesseract) {
    return result.textContent = "❌ Tesseract.js 未載入";
  }

  result.textContent = "🔄 下載/初始化 Tesseract…（首次會較久）";
  progressBar.style.display = "block";
  progressBar.removeAttribute("value");

  const canvas = preprocessForOCR(imageElement);

  try {
    const { data } = await Tesseract.recognize(canvas, 'chi_tra+eng', {
      // 進度顯示
      logger: m => {
        if (typeof m.progress === 'number') {
          progressBar.max = 100;
          progressBar.value = Math.round(m.progress * 100);
        }
      },
      // 重要：PSM 影響版面分析（表格建議 6/4/11 視情況調）
      // psm 6：假設為單一均勻區塊的文字（常用於表格）
      // psm 4：單欄多塊
      // psm 11：稀疏文字
      // 這裡先用 6，辨識怪再試 4 或 11
      tessedit_pageseg_mode: 6,
      preserve_interword_spaces: 1
    });

    progressBar.style.display = "none";

    // 直接把辨識文字顯示
    const plainText = (data.text || "").trim();

    // 同時把 words 聚類成行，輸出粗略 CSV（先可用）
    const words = (data.words || []).map(w => ({
      text: w.text,
      bbox: { x0: w.bbox.x0, y0: w.bbox.y0, x1: w.bbox.x1, y1: w.bbox.y1 }
    }));
    const csv = wordsToCSV(words);

    result.textContent =
      "【全文】\n" + plainText + "\n\n" +
      "【粗略 CSV（測試版）】\n" + csv;

  } catch (err) {
    console.error(err);
    progressBar.style.display = "none";
    result.textContent = "❌ Tesseract 錯誤：" + (err?.message || err);
  }
});

// ====== det 測試（保留；如果不需要可刪）======
ocrBtn.addEventListener("click", async () => {
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
    result.textContent = "✅ 模型載入完成（det）→ 輸出張量見 Console";
    const feeds = {};
    feeds[detSession.inputNames[0]] = imageToTensorSimple(imageElement);
    const outputs = await detSession.run(feeds);
    console.log("Det 模型輸出:", outputs);
  } catch (err) {
    console.error(err);
    result.textContent = "❌ det 錯誤：" + (err?.message || err);
    progressBar.style.display = "none";
  }
});

// 簡化的 det 前處理（只為讓 det 跑通印輸出）
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
