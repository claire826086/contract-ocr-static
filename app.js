// app.js — PP-OCR det(DB)→postprocess(unclip)→rotated crop→(cls)→rec→表格 + 下載 Excel
// 先在 index.html 載入：onnxruntime-web、xlsx（SheetJS）、@techstark/opencv-js

/******** ORT 設定 ********/
if (window.ort) {
  ort.env.wasm.wasmPaths = {
    mjs:  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.mjs",
    wasm: "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.22.0/ort-wasm-simd-threaded.jsep.wasm",
  };
  ort.env.wasm.numThreads = 1;
} else {
  console.warn("onnxruntime-web 未載入");
}

/******** DOM ********/
const fileInput = document.getElementById("fileInput");
const preview   = document.getElementById("preview");
const ocrBtn    = document.getElementById("ocrBtn"); // det 測試（保留）
const result    = document.getElementById("result");

// 主流程按鈕
const tableBtn = document.createElement("button");
tableBtn.textContent = "表格抽取（det→unclip→(cls)→rec）";
tableBtn.style.marginTop = "8px";
ocrBtn.insertAdjacentElement("afterend", tableBtn);

// 進度條
const progressBar = document.createElement("progress");
progressBar.max = 100; progressBar.value = 0;
progressBar.style.width = "100%"; progressBar.style.display = "none";
result.insertAdjacentElement("beforebegin", progressBar);

/******** 全域狀態 ********/
let detSession = null;
let recSession = null;
let clsSession = null;
let keys = null;      // rec 字典
let imageElement = null;

// 自動取當前 Pages 根路徑
const BASE = location.origin + location.pathname.replace(/\/[^/]*$/, "/");
const DET_URL  = BASE + "models/det.onnx";
const REC_URL  = BASE + "models/rec.onnx";
const CLS_URL  = BASE + "models/cls.onnx"; // 可選
const KEYS_URL = BASE + "models/ppocr_keys_v1.txt";

/******** 載圖預覽 ********/
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

/******** 小工具：下載（帶進度） ********/
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
    if (total) progressBar.value = Math.round((received / total) * 100);
    else progressBar.removeAttribute("value");
  }
  progressBar.style.display = "none";
  const size = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out.buffer;
}

/******** det 前處理（640×640，letterbox，RGB/255） ********/
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

/******** DB 後處理：用 OpenCV 取輪廓 + minAreaRect + unclip ********/
/* detOutMap: ort.Tensor [1,1,H,W] 值域[0,1]
   回傳 rotated rect 陣列：{cx,cy,w,h,angle} 在 640x640 座標系 */
function dbPostprocessToRBoxes(detOutMap, binThr=0.25, unclipRatio=1.6, minBox=8) {
  if (!window.cv || !cv.Mat) {
    console.warn("OpenCV 尚未就緒，fallback 以 axis-aligned boxes");
    // 簡化：回傳空
    return [];
  }
  const [_, __, H, W] = detOutMap.dims;
  const prob = detOutMap.data; // H*W
  // 轉成 Mat 單通道
  const mat = cv.matFromArray(H, W, cv.CV_32FC1, prob);
  // 閾值 → 二值
  let bin = new cv.Mat();
  cv.threshold(mat, bin, binThr, 1.0, cv.THRESH_BINARY);
  // 放大到 0~255
  let bin8 = new cv.Mat();
  bin.convertTo(bin8, cv.CV_8UC1, 255, 0);

  // 找輪廓
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(bin8, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  const rboxes = [];
  for (let i=0; i<contours.size(); i++){
    const cnt = contours.get(i);
    const rect = cv.minAreaRect(cnt); // {center:{x,y}, size:{width,height}, angle}
    let w = rect.size.width, h = rect.size.height;
    if (Math.min(w,h) < minBox) { cnt.delete(); continue; }

    // unclip：把短邊乘上 ratio
    const r = Math.sqrt(w*h) * (unclipRatio - 1);
    w = Math.max(1, w + r);
    h = Math.max(1, h + r);

    rboxes.push({
      cx: rect.center.x, cy: rect.center.y,
      w, h,
      angle: rect.angle // OpenCV 角度，順時針為正/負依版本，下面會處理
    });
    cnt.delete();
  }
  bin.delete(); bin8.delete(); contours.delete(); hierarchy.delete(); mat.delete();
  return rboxes;
}

/******** 將 det 座標還原到原圖 ********/
function rboxToOriginal(rb, meta) {
  const {dx, dy, scale} = meta; // 640座標 → 原圖
  return {
    cx: Math.round((rb.cx - dx) / scale),
    cy: Math.round((rb.cy - dy) / scale),
    w:  Math.round(rb.w / scale),
    h:  Math.round(rb.h / scale),
    angle: rb.angle
  };
}

/******** 旋轉裁切（依 minAreaRect） ********/
function cropRotatedRectFromImage(imgEl, rbOrig) {
  if (!window.cv || !cv.Mat) return null;
  const src = cv.imread(imgEl); // RGBA
  const center = new cv.Point(rbOrig.cx, rbOrig.cy);
  // OpenCV 的 angle：對於直立字串，minAreaRect 可能給 -90~0；這裡把盒子轉成水平
  const angle = rbOrig.angle; // 直接用；如果方向怪可以 +90 or -90 微調
  const size  = new cv.Size(rbOrig.w, rbOrig.h);

  // 先做旋轉
  const M = cv.getRotationMatrix2D(center, angle, 1.0);
  let rotated = new cv.Mat();
  cv.warpAffine(src, rotated, M, src.size(), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
  // 從旋轉後圖上擷取水平矩形
  let out = new cv.Mat();
  const roi = new cv.Rect(
    Math.max(0, Math.round(center.x - size.width/2)),
    Math.max(0, Math.round(center.y - size.height/2)),
    Math.max(1, Math.round(size.width)),
    Math.max(1, Math.round(size.height))
  );
  // 防越界
  const safeRect = new cv.Rect(
    Math.min(Math.max(roi.x, 0), Math.max(0, rotated.cols-1)),
    Math.min(Math.max(roi.y, 0), Math.max(0, rotated.rows-1)),
    Math.min(roi.width,  rotated.cols - Math.min(Math.max(roi.x,0), rotated.cols-1)),
    Math.min(roi.height, rotated.rows - Math.min(Math.max(roi.y,0), rotated.rows-1))
  );
  out = rotated.roi(safeRect);

  // 輸出到 canvas
  const can = document.createElement("canvas");
  can.width = out.cols; can.height = out.rows;
  cv.imshow(can, out);

  // 釋放
  src.delete(); rotated.delete(); out.delete(); M.delete();
  return can;
}

/******** 行/列分群：以文字塊中心排序（粗略） ********/
function groupToGridFromRBoxes(rboxesOrig, yTol=12, xTol=12){
  // 將 rbox 轉成 axis-aligned box 用於分群
  const boxes = rboxesOrig.map(rb => ({
    x0: Math.round(rb.cx - rb.w/2),
    y0: Math.round(rb.cy - rb.h/2),
    x1: Math.round(rb.cx + rb.w/2),
    y1: Math.round(rb.cy + rb.h/2),
    rb
  }));
  boxes.sort((a,b)=> (a.y0+a.y1)/2 - (b.y0+b.y1)/2 || a.x0-b.x0);

  const rows = [];
  for (const b of boxes){
    const cy = (b.y0+b.y1)/2;
    let row = rows.find(r => Math.abs(r.cy - cy) <= yTol);
    if (!row){ row = { cy, cells: [] }; rows.push(row); }
    row.cells.push(b);
  }
  rows.sort((a,b)=>a.cy-b.cy);
  return rows.map(r=>{
    r.cells.sort((a,b)=>a.x0-b.x0);
    // 合併相鄰
    const merged = [];
    for (const c of r.cells){
      const last = merged[merged.length-1];
      if (!last) { merged.push(c); continue; }
      if (c.x0 - last.x1 <= xTol){
        last.x1 = Math.max(last.x1, c.x1);
        last.rb.w = Math.max(last.rb.w, c.rb.w);
        last.rb.h = Math.max(last.rb.h, c.rb.h);
        last.rb.cx = (last.rb.cx + c.rb.cx)/2;
        last.rb.cy = (last.rb.cy + c.rb.cy)/2;
      } else {
        merged.push(c);
      }
    }
    return merged.map(m => m.rb); // 回傳此列的 rboxes
  });
}

/******** rec：讀字典 + CTC 解碼 ********/
async function loadKeys() {
  const txt = await (await fetch(KEYS_URL, {cache:"force-cache"})).text();
  return txt.split(/\r?\n/).map(s=>s.trim()).filter(s => s.length>0 && !s.startsWith("#"));
}

function ctcDecode(seq, keys, blankIndex=0){
  const out = [];
  let prev = -1;
  for (const k of seq){
    if (k === blankIndex || k === prev) { prev = k; continue; }
    prev = k; out.push(keys[k] ?? "");
  }
  return out.join("");
}

/******** cls：方向分類（可選） ********/
async function runClsIfReady(canvas){
  if (!clsSession) return canvas; // 未啟用就跳過
  // PP-OCR cls 預處理：RGB /255 → (x-0.5)/0.5，輸入常見 48×192
  const targetH = 48, targetW = 192;
  const c = document.createElement("canvas");
  c.width = targetW; c.height = targetH;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0,0,targetW,targetH);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  const scale = Math.min(targetW/canvas.width, targetH/canvas.height);
  const w = Math.round(canvas.width*scale), h = Math.round(canvas.height*scale);
  g.drawImage(canvas, Math.floor((targetW-w)/2), Math.floor((targetH-h)/2), w, h);

  const data = g.getImageData(0,0,targetW,targetH).data;
  const f32 = new Float32Array(3*targetH*targetW);
  let p = 0;
  for (let y=0;y<targetH;y++){
    for (let x=0;x<targetW;x++){
      const i=(y*targetW+x)*4;
      f32[0*targetH*targetW + p] = (data[i  ]/255-0.5)/0.5;
      f32[1*targetH*targetW + p] = (data[i+1]/255-0.5)/0.5;
      f32[2*targetH*targetW + p] = (data[i+2]/255-0.5)/0.5;
      p++;
    }
  }
  const input = new ort.Tensor("float32", f32, [1,3,targetH,targetW]);
  const feeds = {}; feeds[clsSession.inputNames[0]] = input;
  const out = await clsSession.run(feeds);
  const name = Object.keys(out)[0];
  const logits = out[name].data;
  // 2 類：0 正，1 180 或 90（依模型）；這裡簡化成若第1類較大就把圖旋轉180。
  if (logits.length>=2 && logits[1] > logits[0]) {
    const rot = document.createElement("canvas");
    rot.width = canvas.width; rot.height = canvas.height;
    const rg = rot.getContext("2d");
    rg.translate(rot.width/2, rot.height/2);
    rg.rotate(Math.PI); // 180deg
    rg.drawImage(canvas, -canvas.width/2, -canvas.height/2);
    return rot;
  }
  return canvas;
}

/******** 單格 rec（H=48、(x-0.5)/0.5、自動判 shape、blank 推斷） ********/
async function recognizeCanvas(canvas){
  if (!recSession || !keys) return "";

  // 先跑 cls（若有）
  const pre = await runClsIfReady(canvas);

  const targetH = 48, maxW = 512;
  const scale = targetH / pre.height;
  let newW = Math.max(16, Math.min(maxW, Math.round(pre.width * scale)));

  const c = document.createElement("canvas");
  c.width = newW; c.height = targetH;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, newW, targetH);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(pre, 0, 0, newW, targetH);

  const data = g.getImageData(0,0,newW,targetH).data;
  const f32 = new Float32Array(3*targetH*newW);
  let p = 0;
  for (let y=0;y<targetH;y++){
    for (let x=0;x<newW;x++){
      const i=(y*newW+x)*4;
      f32[0*targetH*newW + p] = (data[i  ]/255-0.5)/0.5;
      f32[1*targetH*newW + p] = (data[i+1]/255-0.5)/0.5;
      f32[2*targetH*newW + p] = (data[i+2]/255-0.5)/0.5;
      p++;
    }
  }
  const input = new ort.Tensor("float32", f32, [1,3,targetH,newW]);
  const feeds = {}; feeds[recSession.inputNames[0]] = input;
  const out = await recSession.run(feeds);
  const name = Object.keys(out)[0];
  const logits = out[name];
  const dims = logits.dims.slice();
  const A = logits.data;

  let T, C, step;
  if (dims.length === 3 && dims[0] === 1 && dims[1] > 1 && dims[2] > 1) { T=dims[1]; C=dims[2]; step=(t,c)=>A[t*C+c]; }
  else if (dims.length === 3 && dims[0] === 1 && dims[2] > 1 && dims[1] > 1) { C=dims[1]; T=dims[2]; step=(t,c)=>A[c*T+t]; }
  else if (dims.length === 2 && dims[0] > 1 && dims[1] > 1) { T=dims[0]; C=dims[1]; step=(t,c)=>A[t*C+c]; }
  else if (dims.length === 2 && dims[1] > 1 && dims[0] > 1) { C=dims[0]; T=dims[1]; step=(t,c)=>A[c*T+t]; }
  else { console.warn("未知 rec 輸出形狀", dims); return ""; }

  let blankIndex = 0;
  if (C === keys.length + 1) blankIndex = C - 1;

  const seq = new Array(T);
  for (let t=0; t<T; t++){
    let bestI = 0, bestV = -Infinity;
    for (let c2=0; c2<C; c2++){
      const v = step(t, c2);
      if (v > bestV){ bestV=v; bestI=c2; }
    }
    seq[t] = bestI;
  }
  return ctcDecode(seq, keys, blankIndex);
}

/******** 視覺化：把 rboxes 畫在預覽下方 ********/
function drawRBoxesOnImage(img, rboxesOrig){
  const can = document.createElement("canvas");
  can.width = img.naturalWidth; can.height = img.naturalHeight;
  const g = can.getContext("2d");
  g.drawImage(img, 0, 0);
  g.lineWidth = Math.max(2, Math.round(img.naturalWidth/600));
  g.strokeStyle = "rgba(0,200,255,0.9)";
  rboxesOrig.forEach(rb=>{
    g.save();
    g.translate(rb.cx, rb.cy);
    g.rotate(rb.angle * Math.PI / 180);
    g.strokeRect(-rb.w/2, -rb.h/2, rb.w, rb.h);
    g.restore();
  });
  preview.insertAdjacentElement("afterend", can);
  return can;
}

/******** 匯出 Excel ********/
function exportToXLSX(rowsText, filename="ocr_table.xlsx"){
  if (!window.XLSX) { alert("XLSX 函式庫未載入"); return; }
  const ws = XLSX.utils.aoa_to_sheet(rowsText);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  XLSX.writeFile(wb, filename);
}

/******** 主流程：det→unclip→rotated crop→(cls)→rec→表格 ********/
tableBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");

  try {
    // det
    if (!detSession) {
      const buf = await fetchWithProgress(DET_URL, "下載 det.onnx");
      detSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    }
    // rec + keys
    if (!recSession) {
      const head = await fetch(REC_URL, {method:"HEAD"});
      if (!head.ok) throw new Error("找不到 rec.onnx（可先只跑框）");
      const buf = await fetchWithProgress(REC_URL, "下載 rec.onnx");
      recSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
      keys = await loadKeys();
      console.log("keys.length =", keys.length);
    }
    // cls（可選）
    if (!clsSession) {
      try {
        const h = await fetch(CLS_URL, {method:"HEAD"});
        if (h.ok) {
          const cbuf = await fetchWithProgress(CLS_URL, "下載 cls.onnx");
          clsSession = await ort.InferenceSession.create(cbuf, { executionProviders: ["wasm"] });
        }
      } catch { /* ignore */ }
    }

    // det 推論
    result.textContent = "🔎 det 推論中…";
    const meta = imageToDetTensor(imageElement);
    const feeds = {}; feeds[detSession.inputNames[0]] = meta.tensor;
    const detOut = await detSession.run(feeds);
    const detName = Object.keys(detOut)[0];
    const detMap = detOut[detName]; // [1,1,H,W]

    // DB 後處理（640座標）→ 還原到原圖
    const rboxes640 = dbPostprocessToRBoxes(detMap, /*thr=*/0.25, /*unclip=*/1.6, /*minBox=*/8);
    const rboxesOrig = rboxes640.map(rb => rboxToOriginal(rb, meta));
    drawRBoxesOnImage(imageElement, rboxesOrig);

    // 依行列分群（簡化 grid）
    const grid = groupToGridFromRBoxes(
      rboxesOrig,
      Math.round(imageElement.naturalHeight/120),
      Math.round(imageElement.naturalWidth/180)
    );

    // 逐格：旋轉裁切 → (cls) → rec
    const rowsText = [];
    for (const row of grid){
      const cols = [];
      for (const rb of row){
        const crop = cropRotatedRectFromImage(imageElement, rb);
        let text = "";
        if (crop) {
          try { text = await recognizeCanvas(crop) || ""; } catch(e){ console.warn("rec 失敗:", e); }
        }
        cols.push(text.trim());
      }
      rowsText.push(cols);
    }

    // 顯示 HTML + 下載 Excel
    let html = "<table border='1' style='border-collapse:collapse'>\n";
    for (const row of rowsText){
      html += "  <tr>" + row.map(t=>`<td style="padding:4px 8px">${escapeHtml(t)}</td>`).join("") + "</tr>\n";
    }
    html += "</table>";

    result.textContent = "✅ 完成（下方為表格 HTML），可下載 Excel";
    const preHtml = document.createElement("div");
    preHtml.innerHTML = html;
    result.insertAdjacentElement("afterend", preHtml);

    const dlBtn = document.createElement("button");
    dlBtn.textContent = "下載 Excel（.xlsx）";
    dlBtn.style.marginTop = "8px";
    dlBtn.onclick = () => exportToXLSX(rowsText, "ocr_table.xlsx");
    preHtml.insertAdjacentElement("afterend", dlBtn);

  } catch (err) {
    console.error(err);
    progressBar.style.display = "none";
    result.textContent = "❌ 錯誤：" + (err?.message || err);
  }
});

/******** det 測試（保留） ********/
ocrBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");
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
    result.textContent = "✅ det 輸出已印到 Console（主流程請按『表格抽取』）";
  } catch (err) {
    console.error(err);
    progressBar.style.display = "none";
    result.textContent = "❌ det 錯誤：" + (err?.message || err);
  }
});

/******** 小工具 ********/
function escapeHtml(s){ return String(s||"").replace(/[&<>"']/g, m=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m])); }
