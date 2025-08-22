// app.js — PP-OCR (det→postprocess→rows/cols) + (可選) rec → HTML/CSV
// 仍保留 det 測試按鈕；新增「表格抽取（Paddle）」主流程按鈕

// ====== ORT 設定 ======
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
const ocrBtn    = document.getElementById("ocrBtn"); // det 測試
const result    = document.getElementById("result");

// 主流程按鈕
const tableBtn = document.createElement("button");
tableBtn.textContent = "表格抽取（Paddle det→post→(rec)）";
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
let recSession = null;
let keys = null; // rec 字典
let imageElement = null;

// 你的 GitHub Pages 模型 URL（改成你的 repo）
const BASE = location.origin + location.pathname.replace(/\/[^/]*$/, "/");
const DET_URL = BASE + "models/det.onnx";
const REC_URL = BASE + "models/rec.onnx";
const KEYS_URL = BASE + "models/ppocr_keys_v1.txt";

// ====== 載圖預覽 ======
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

// ====== 工具：下載（帶進度，供 det.onnx/rec.onnx 用）======
async function fetchWithProgress(url, label="下載中") {
  result.textContent = `🔄 ${label}…`;
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
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out.buffer;
}

// ====== det 前處理（640×640，RGB/255，letterbox）======
const DET_SIZE = 640;
function imageToDetTensor(img) {
  const target = DET_SIZE;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = target; canvas.height = target;
  const scale = Math.min(target / img.naturalWidth, target / img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const dx = Math.floor((target - w) / 2);
  const dy = Math.floor((target - h) / 2);
  ctx.fillStyle = "#000"; ctx.fillRect(0,0,target,target);
  ctx.drawImage(img, dx, dy, w, h);
  const data = ctx.getImageData(0, 0, target, target).data;
  const f32 = new Float32Array(3 * target * target);
  let p = 0;
  for (let y=0; y<target; y++){
    for (let x=0; x<target; x++){
      const i = (y*target + x)*4;
      f32[0*target*target + p] = data[i  ] / 255;
      f32[1*target*target + p] = data[i+1] / 255;
      f32[2*target*target + p] = data[i+2] / 255;
      p++;
    }
  }
  return { tensor: new ort.Tensor("float32", f32, [1,3,target,target]), dx, dy, scale, w, h };
}

// ====== det 後處理（閾值→連通區→外接矩形，多邊形簡化）======
function probMapToBoxes(detTensor, thr=0.3, minArea=100) {
  // detTensor: [1,1,H,W] float in [0,1]
  const [_, __, H, W] = detTensor.dims;
  const src = detTensor.data; // length H*W
  // 二值化 → CCL（4-連通）
  const bin = new Uint8Array(H*W);
  for (let i=0;i<H*W;i++) bin[i] = src[i] >= thr ? 1 : 0;

  const labels = new Int32Array(H*W).fill(0);
  let cur = 0;
  const boxes = [];
  const qx = new Int32Array(H*W);
  const qy = new Int32Array(H*W);

  for (let y=0;y<H;y++){
    for (let x=0;x<W;x++){
      const idx = y*W + x;
      if (bin[idx]===0 || labels[idx]!==0) continue;
      cur++;
      let head=0, tail=0;
      qx[tail]=x; qy[tail]=y; tail++;
      labels[idx]=cur;

      let minx=x, miny=y, maxx=x, maxy=y, area=0;
      while(head<tail){
        const cx=qx[head], cy=qy[head]; head++;
        area++;
        // 4-neighbors
        const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
        for (const [dx,dy] of dirs){
          const nx=cx+dx, ny=cy+dy;
          if (nx<0||nx>=W||ny<0||ny>=H) continue;
          const nidx=ny*W+nx;
          if (bin[nidx]===1 && labels[nidx]===0){
            labels[nidx]=cur;
            qx[tail]=nx; qy[tail]=ny; tail++;
            if (nx<minx) minx=nx; if (ny<miny) miny=ny;
            if (nx>maxx) maxx=nx; if (ny>maxy) maxy=ny;
          }
        }
      }
      if (area>=minArea){
        boxes.push({x0:minx, y0:miny, x1:maxx+1, y1:maxy+1, area});
      }
    }
  }
  // 依上到下、左到右排序
  boxes.sort((a,b)=>{
    const cyA = (a.y0+a.y1)/2, cyB=(b.y0+b.y1)/2;
    if (Math.abs(cyA-cyB)>10) return cyA-cyB;
    return a.x0-b.x0;
  });
  return boxes;
}

// ====== 將 det 盒轉回原圖座標（反 letterbox）======
function detBoxToOriginal(box, meta){
  const {dx, dy, scale} = meta;
  return {
    x0: Math.max(0, Math.round((box.x0 - dx) / scale)),
    y0: Math.max(0, Math.round((box.y0 - dy) / scale)),
    x1: Math.max(0, Math.round((box.x1 - dx) / scale)),
    y1: Math.max(0, Math.round((box.y1 - dy) / scale)),
  };
}

// ====== 將框群組成 rows / cols（簡化、先可用）======
function groupToGrid(boxes, yTol=12, xTol=12){
  // 先按 y 聚成行
  const rows = [];
  for (const b of boxes){
    const cy = (b.y0+b.y1)/2;
    let row = rows.find(r => Math.abs(r.cy - cy) <= yTol);
    if (!row){ row = { cy, cells: [] }; rows.push(row); }
    row.cells.push(b);
  }
  rows.sort((a,b)=>a.cy-b.cy);
  // 每行內按 x 排序；若相鄰 gap > xTol，視為新欄位
  const grid = rows.map(r=>{
    r.cells.sort((a,b)=>a.x0-b.x0);
    // 合併近距離的 boxes
    const merged = [];
    for (const c of r.cells){
      const last = merged[merged.length-1];
      if (!last) { merged.push({...c}); continue; }
      if (c.x0 - last.x1 <= xTol){ // 視為同欄延伸
        last.x1 = Math.max(last.x1, c.x1);
        last.y0 = Math.min(last.y0, c.y0);
        last.y1 = Math.max(last.y1, c.y1);
      } else {
        merged.push({...c});
      }
    }
    return merged;
  });
  return grid; // 陣列（每列是一排 boxes）
}

// ====== rec：讀字典 + CTC 解碼 ======
async function loadKeys() {
  const txt = await (await fetch(KEYS_URL, {cache:"force-cache"})).text();
  // 每行一個字；有些第一行是 "blank" 或 "#"
  return txt.split(/\r?\n/).filter(s=>s.length>0);
}

function argmax(arr){ let idx=0, v=-Infinity; for (let i=0;i<arr.length;i++){ if(arr[i]>v){v=arr[i]; idx=i;} } return idx; }

// 簡易 CTC decoder：去重、去 blank
function ctcDecode(prob2D, keys, blankIndex=0){
  // prob2D: Float32Array (T * C) 或 (C * T)；這裡假設 T×C
  // 為簡化：這裡傳入的是經 argmax 的序列
  const T = prob2D.length;
  const out = [];
  let prev = -1;
  for (let t=0;t<T;t++){
    const k = prob2D[t];
    if (k===blankIndex || k===prev) { prev = k; continue; }
    prev = k;
    out.push(keys[k] ?? "");
  }
  return out.join("");
}

// 對單一裁切圖做 rec：resize 到 32×W（保持比例，寬上限 320）
async function recognizeCrop(canvas){
  if (!recSession || !keys) return null;

  const targetH = 32, maxW = 320;
  const scale = targetH / canvas.height;
  let newW = Math.max(16, Math.min(maxW, Math.round(canvas.width * scale)));

  const c = document.createElement("canvas");
  c.width = newW; c.height = targetH;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0,0,newW,targetH);
  g.drawImage(canvas, 0, 0, newW, targetH);

  const data = g.getImageData(0,0,newW,targetH).data;
  // normalize to [0,1], RGB → CHW
  const f32 = new Float32Array(3*targetH*newW);
  let p = 0;
  for (let y=0;y<targetH;y++){
    for (let x=0;x<newW;x++){
      const i=(y*newW+x)*4;
      f32[0*targetH*newW + p] = data[i  ]/255;
      f32[1*targetH*newW + p] = data[i+1]/255;
      f32[2*targetH*newW + p] = data[i+2]/255;
      p++;
    }
  }
  const input = new ort.Tensor("float32", f32, [1,3,targetH,newW]);
  const feeds = {}; feeds[recSession.inputNames[0]] = input;
  const out = await recSession.run(feeds);
  const name = Object.keys(out)[0];
  const logits = out[name]; // [1, T, C] or [T, C]
  const dims = logits.dims; // e.g., [1, T, C]
  const dataLogits = logits.data;
  let T, C, offset=0;
  if (dims.length===3){ T=dims[1]; C=dims[2]; offset=0; }
  else { T=dims[0]; C=dims[1]; }
  // argmax per time step
  const seq = new Array(T);
  for (let t=0;t<T;t++){
    const row = dataLogits.subarray(t*C + offset, t*C + offset + C);
    seq[t] = argmax(row);
  }
  return ctcDecode(seq, keys, /*blankIndex=*/0);
}

// ====== 視覺：畫框疊在預覽圖下方 ======
function drawBoxesOnImage(img, boxesOrig){
  const can = document.createElement("canvas");
  can.width = img.naturalWidth; can.height = img.naturalHeight;
  const g = can.getContext("2d");
  g.drawImage(img, 0, 0);
  g.lineWidth = Math.max(2, Math.round(img.naturalWidth/600));
  g.strokeStyle = "rgba(0,200,255,0.9)";
  for (const b of boxesOrig){
    g.strokeRect(b.x0, b.y0, b.x1-b.x0, b.y1-b.y0);
  }
  preview.insertAdjacentElement("afterend", can);
  return can;
}

// ====== 主流程：表格抽取 ======
tableBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");

  try {
    // 1) 準備 det session
    if (!detSession) {
      const buf = await fetchWithProgress(DET_URL, "下載 det.onnx");
      detSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    }
    // 2) (可選) 準備 rec session + keys（若檔案存在）
    if (!recSession) {
      try {
        const recHead = await fetch(REC_URL, {method:"HEAD"});
        if (recHead.ok) {
          const buf = await fetchWithProgress(REC_URL, "下載 rec.onnx");
          recSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
          keys = await loadKeys();
        }
      } catch {}
    }

    // 3) det 推論
    result.textContent = "🔎 det 推論中…";
    const meta = imageToDetTensor(imageElement);
    const feeds = {}; feeds[detSession.inputNames[0]] = meta.tensor;
    const detOut = await detSession.run(feeds);
    const detName = Object.keys(detOut)[0];
    const detMap = detOut[detName]; // [1,1,H,W], data in [0,1]（多數模型）

    // 4) 後處理：機率→框（640×640座標），再還原到原圖座標
    const boxes640 = probMapToBoxes(detMap, /*thr=*/0.3, /*minArea=*/80);
    const boxesOrig = boxes640.map(b => detBoxToOriginal(b, meta));

    // 5) 視覺化框
    const overlay = drawBoxesOnImage(imageElement, boxesOrig);

    // 6) 依 rows/cols 估算表格網格
    //    這裡使用簡化分群；若需要更嚴謹的「格線還原」，之後可加 Hough 線偵測。
    //    我們在原圖座標上分群，避免縮放造成誤差。
    const boxesForGrid = boxesOrig.map(b=>({x0:b.x0,y0:b.y0,x1:b.x1,y1:b.y1}));
    const grid = groupToGrid(boxesForGrid, /*yTol=*/Math.round(imageElement.naturalHeight/120), /*xTol=*/Math.round(imageElement.naturalWidth/180));

    // 7) 逐格裁切 →（若 rec 可用就辨識）→ 組 HTML/CSV
    const tmpCanvas = document.createElement("canvas");
    const tmpCtx = tmpCanvas.getContext("2d");

    const rowsText = [];
    for (const row of grid){
      const colsText = [];
      for (const b of row){
        const w = Math.max(1, b.x1-b.x0), h=Math.max(1, b.y1-b.y0);
        tmpCanvas.width = w; tmpCanvas.height = h;
        tmpCtx.drawImage(imageElement, b.x0, b.y0, w, h, 0, 0, w, h);
        let text = "";
        if (recSession && keys){
          try {
            text = await recognizeCrop(tmpCanvas) || "";
          } catch(e){
            text = ""; console.warn("rec 失敗：", e);
          }
        }
        colsText.push(text.trim());
      }
      rowsText.push(colsText);
    }

    // 8) 輸出 HTML 表格 + CSV（若沒 rec，就輸出空字串或座標）
    let html = "<table border='1' style='border-collapse:collapse'>\n";
    for (const row of rowsText){
      html += "  <tr>" + row.map(t=>`<td style="padding:4px 8px">${escapeHtml(t)}</td>`).join("") + "</tr>\n";
    }
    html += "</table>";

    const csv = rowsText.map(r=>r.map(s=>csvEscape(s)).join(",")).join("\n");

    result.textContent = "✅ 完成（下方為表格 HTML，其次是 CSV）\n\n";
    const preHtml = document.createElement("div");
    preHtml.innerHTML = html;
    result.insertAdjacentElement("afterend", preHtml);

    const preCsv = document.createElement("pre");
    preCsv.textContent = csv;
    preHtml.insertAdjacentElement("afterend", preCsv);

  } catch (err) {
    console.error(err);
    progressBar.style.display = "none";
    result.textContent = "❌ 錯誤：" + (err?.message || err);
  }
});

// ====== det 測試（保留）======
ocrBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
  if (!window.ort) { result.textContent = "❌ 無法載入 onnxruntime-web"; return; }
  result.textContent = "🔄 載入 det 模型中...";
  try {
    if (!detSession) {
      const buffer = await fetchWithProgress(DET_URL, "下載 det.onnx");
      detSession = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
    }
    const meta = imageToDetTensor(imageElement);
    const feeds = {}; feeds[detSession.inputNames[0]] = meta.tensor;
    const outputs = await detSession.run(feeds);
    console.log("Det 模型輸出:", outputs);
    result.textContent = "✅ det 輸出已印到 Console（下一步請用『表格抽取（Paddle…）』）";
  } catch (err) {
    console.error(err);
    progressBar.style.display = "none";
    result.textContent = "❌ det 錯誤：" + (err?.message || err);
  }
});

// ====== 小工具 ======
function escapeHtml(s){ return s.replace(/[&<>"']/g, m=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m])); }
function csvEscape(s){
  if (s==null) return "";
  s = String(s);
  if (s.includes('"') || s.includes(",") || s.includes("\n")) {
    return `"${s.replace(/"/g,'""')}"`;
  }
  return s;
}
