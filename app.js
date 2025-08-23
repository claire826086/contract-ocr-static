// app.js — det(DB) → postprocess(unclip/axis-fallback) → rotated crop → (cls 可選) → rec
// 並新增：rec.onnx 與 ppocr_keys_v1.txt 相容性檢查（C vs keys.length）
// 需於 index.html 先載入：onnxruntime-web、xlsx、opencv.js

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
const ocrBtn    = document.getElementById("ocrBtn");   // det 測試（保留）
const result    = document.getElementById("result");

// 主流程按鈕
const tableBtn = document.createElement("button");
tableBtn.textContent = "表格抽取（det→post→rec）";
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
let clsSession = null; // 可選
let keys = null;
let imageElement = null;
let recClassCount = null; // 由檢查流程寫入，用於提示/除錯

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

/******** 工具：進度下載 ********/
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

/******** 等待 OpenCV ready（最多 6s） ********/
function waitForOpenCV(maxWaitMs = 6000) {
  return new Promise((resolve) => {
    const start = performance.now();
    (function loop(){
      if (window.cv && cv.Mat) return resolve(true);
      if (performance.now() - start > maxWaitMs) return resolve(false);
      setTimeout(loop, 200);
    })();
  });
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

/******** 簡易（軸對齊）後處理：二值 + 連通區外接矩形 ********/
function probMapToBoxes_AABB(detTensor, thr=0.18, minArea=40) {
  const [_, __, H, W] = detTensor.dims;
  const src = detTensor.data;
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
  boxes.sort((a,b)=>{
    const cyA = (a.y0+a.y1)/2, cyB=(b.y0+b.y1)/2;
    if (Math.abs(cyA-cyB)>10) return cyA-cyB;
    return a.x0-b.x0;
  });
  // 轉成 rotated-box 形式（角度 0）
  return boxes.map(b=>({
    cx:(b.x0+b.x1)/2, cy:(b.y0+b.y1)/2,
    w: b.x1-b.x0, h: b.y1-b.y0, angle: 0
  }));
}

/******** DB 後處理（OpenCV）：輪廓 + minAreaRect + unclip ********/
function dbPostprocess_RBOX(detOutMap, binThr=0.22, unclipRatio=1.8, minBox=6) {
  const [_, __, H, W] = detOutMap.dims;
  const prob = detOutMap.data; // H*W
  const mat = cv.matFromArray(H, W, cv.CV_32FC1, prob);

  let bin = new cv.Mat();
  cv.threshold(mat, bin, binThr, 1.0, cv.THRESH_BINARY);
  let bin8 = new cv.Mat();
  bin.convertTo(bin8, cv.CV_8UC1, 255, 0);

  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(bin8, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  const rboxes = [];
  for (let i=0; i<contours.size(); i++){
    const cnt = contours.get(i);
    const rect = cv.minAreaRect(cnt);
    let w = rect.size.width, h = rect.size.height;
    if (Math.min(w,h) < minBox) { cnt.delete(); continue; }

    // unclip：用短邊近似
    const r = Math.sqrt(w*h) * (unclipRatio - 1);
    w = Math.max(1, w + r);
    h = Math.max(1, h + r);

    rboxes.push({
      cx: rect.center.x, cy: rect.center.y,
      w, h,
      angle: rect.angle // 角度單位：deg
    });
    cnt.delete();
  }
  bin.delete(); bin8.delete(); contours.delete(); hierarchy.delete(); mat.delete();
  return rboxes;
}

/******** 座標還原到原圖 ********/
function rboxToOriginal(rb, meta) {
  const {dx, dy, scale} = meta;
  return {
    cx: Math.round((rb.cx - dx) / scale),
    cy: Math.round((rb.cy - dy) / scale),
    w:  Math.round(rb.w / scale),
    h:  Math.round(rb.h / scale),
    angle: rb.angle
  };
}

/******** 旋轉裁切 ********/
function cropRotatedRectFromImage(imgEl, rbOrig) {
  if (!window.cv || !cv.Mat) return null;
  const src = cv.imread(imgEl); // RGBA
  const center = new cv.Point(rbOrig.cx, rbOrig.cy);
  const size  = new cv.Size(Math.max(1, rbOrig.w), Math.max(1, rbOrig.h));
  const angle = rbOrig.angle;

  const M = cv.getRotationMatrix2D(center, angle, 1.0);
  let rotated = new cv.Mat();
  cv.warpAffine(src, rotated, M, src.size(), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
  const roi = new cv.Rect(
    Math.max(0, Math.round(center.x - size.width/2)),
    Math.max(0, Math.round(center.y - size.height/2)),
    Math.max(1, Math.round(size.width)),
    Math.max(1, Math.round(size.height))
  );
  const safeRect = new cv.Rect(
    Math.min(Math.max(roi.x, 0), Math.max(0, rotated.cols-1)),
    Math.min(Math.max(roi.y, 0), Math.max(0, rotated.rows-1)),
    Math.min(roi.width,  rotated.cols - Math.min(Math.max(roi.x,0), rotated.cols-1)),
    Math.min(roi.height, rotated.rows - Math.min(Math.max(roi.y,0), rotated.rows-1))
  );
  const out = rotated.roi(safeRect);

  const can = document.createElement("canvas");
  can.width = out.cols; can.height = out.rows;
  cv.imshow(can, out);

  src.delete(); rotated.delete(); out.delete(); M.delete();
  return can;
}

/******** 行列分群（簡化） ********/
function gridFromRBoxes(rboxesOrig, yTol=12, xTol=12){
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
    return merged.map(m => m.rb);
  });
}

/******** keys + CTC ********/
async function loadKeys() {
  const txt = await (await fetch(KEYS_URL, {cache:"force-cache"})).text();
  return txt.split(/\r?\n/).map(s=>s.trim()).filter(s => s.length>0 && !s.startsWith("#"));
}
function ctcDecode(seq, keys, blankIndex=0){
  const out = []; let prev = -1;
  for (const k of seq){
    if (k === blankIndex || k === prev) { prev = k; continue; }
    prev = k; out.push(keys[k] ?? "");
  }
  return out.join("");
}

/******** cls（可選） ********/
async function runClsIfReady(canvas){
  if (!clsSession) return canvas;
  const targetH = 48, targetW = 192;
  const c = document.createElement("canvas");
  c.width = targetW; c.height = targetH;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0,0,targetW,targetH);
  const scale = Math.min(targetW/canvas.width, targetH/canvas.height);
  const w = Math.round(canvas.width*scale), h = Math.round(canvas.height*scale);
  g.drawImage(canvas, Math.floor((targetW-w)/2), Math.floor((targetH-h)/2), w, h);

  const data = g.getImageData(0,0,targetW,targetH).data;
  const f32 = new Float32Array(3*targetH*targetW);
  let p = 0;
  for (let y=0;y<targetH;y++)for (let x=0;x<targetW;x++){
    const i=(y*targetW+x)*4;
    f32[0*targetH*targetW+p]=(data[i  ]/255-0.5)/0.5;
    f32[1*targetH*targetW+p]=(data[i+1]/255-0.5)/0.5;
    f32[2*targetH*targetW+p]=(data[i+2]/255-0.5)/0.5;
    p++;
  }
  const input=new ort.Tensor("float32",f32,[1,3,targetH,targetW]);
  const feeds={}; feeds[clsSession.inputNames[0]]=input;
  const out=await clsSession.run(feeds);
  const name=Object.keys(out)[0]; const logits=out[name].data;
  if (logits.length>=2 && logits[1]>logits[0]) {
    const rot=document.createElement("canvas");
    rot.width=canvas.width; rot.height=canvas.height;
    const rg=rot.getContext("2d");
    rg.translate(rot.width/2, rot.height/2); rg.rotate(Math.PI); // 180deg
    rg.drawImage(canvas, -canvas.width/2, -canvas.height/2);
    return rot;
  }
  return canvas;
}

/******** rec（H=48，(x-0.5)/0.5，自動 shape/blank） ********/
async function recognizeCanvas(canvas){
  if (!recSession || !keys) return "";
  const pre = await runClsIfReady(canvas);

  const targetH = 48, maxW = 512;
  const scale = targetH / pre.height;
  let newW = Math.max(16, Math.min(maxW, Math.round(pre.width * scale)));

  const c = document.createElement("canvas");
  c.width = newW; c.height = targetH;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0,0,newW,targetH);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(pre, 0, 0, newW, targetH);

  const data = g.getImageData(0,0,newW,targetH).data;
  const f32 = new Float32Array(3*targetH*newW);
  let p = 0;
  for (let y=0;y<targetH;y++)for (let x=0;x<newW;x++){
    const i=(y*newW+x)*4;
    f32[0*targetH*newW+p]=(data[i  ]/255-0.5)/0.5;
    f32[1*targetH*newW+p]=(data[i+1]/255-0.5)/0.5;
    f32[2*targetH*newW+p]=(data[i+2]/255-0.5)/0.5;
    p++;
  }
  const input=new ort.Tensor("float32",f32,[1,3,targetH,newW]);
  const feeds={}; feeds[recSession.inputNames[0]]=input;
  const out=await recSession.run(feeds);
  const name=Object.keys(out)[0]; const logits=out[name];
  const dims=logits.dims.slice(); const A=logits.data;

  let T,C,step;
  if (dims.length===3 && dims[0]===1){
    if (dims[1]>1 && dims[2]>1){ T=dims[1]; C=dims[2]; step=(t,c)=>A[t*C+c]; }
    else { C=dims[1]; T=dims[2]; step=(t,c)=>A[c*T+t]; }
  } else if (dims.length===2){
    if (dims[0]>1 && dims[1]>1){ T=dims[0]; C=dims[1]; step=(t,c)=>A[t*C+c]; }
    else { C=dims[0]; T=dims[1]; step=(t,c)=>A[c*T+t]; }
  } else { console.warn("未知 rec 形狀", dims); return ""; }

  // 記錄 C（供除錯）
  recClassCount = C;

  // blank index：若 C = keys.length + 1，通常 blank 在最後
  let blankIndex = 0;
  if (C === keys.length + 1) blankIndex = C - 1;

  const seq = new Array(T);
  for (let t=0;t<T;t++){
    let bestI=0,bestV=-Infinity;
    for (let c2=0;c2<C;c2++){ const v=step(t,c2); if (v>bestV){bestV=v;bestI=c2;} }
    seq[t]=bestI;
  }
  return ctcDecode(seq, keys, blankIndex);
}

/******** 新增：檢查 rec.onnx 與 keys 是否相容 ********/
async function verifyRecModelCompatibility() {
  // 準備一張極小的白底圖跑 rec 取得輸出維度
  const test = document.createElement("canvas");
  const targetH = 48, testW = 64;
  test.width = testW; test.height = targetH;
  const g = test.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0,0,testW,targetH);
  const txt = await recognizeCanvas(test); // 這一步也會把 recClassCount 寫入（由 recognizeCanvas 內）

  // recClassCount 由 recognizeCanvas 計算得到的 C
  if (typeof recClassCount !== "number") {
    console.warn("無法取得 rec 的 class 維度 C");
    return { ok: false, C: null, reason: "無法取得 rec 的輸出維度" };
  }
  const K = keys.length;
  const ok = (recClassCount === K) || (recClassCount === K + 1);

  if (!ok) {
    console.error(`[模型不相容] rec 的類別數 C=${recClassCount}，但 keys.length=${K}（差值 ${recClassCount-K}）`);
    result.textContent = `❌ 模型不是中文或與字典不相容（C=${recClassCount}，keys=${K}）。請換中文 rec.onnx 或相符的 ppocr_keys_v1.txt。`;
  } else {
    console.log(`[模型相容] C=${recClassCount}，keys=${K}`);
  }
  return { ok, C: recClassCount, K, sampleText: txt };
}

/******** 視覺化 ********/
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
}

/******** 匯出 Excel ********/
function exportToXLSX(rowsText, filename="ocr_table.xlsx"){
  if (!window.XLSX) { alert("XLSX 函式庫未載入"); return; }
  const ws = XLSX.utils.aoa_to_sheet(rowsText);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  XLSX.writeFile(wb, filename);
}

/******** 主流程 ********/
tableBtn.addEventListener("click", async () => {
  if (!imageElement) return alert("請先上傳照片");

  try {
    // 載模型
    if (!detSession) {
      const buf = await fetchWithProgress(DET_URL, "下載 det.onnx");
      detSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    }
    if (!recSession) {
      const head = await fetch(REC_URL, {method:"HEAD"});
      if (!head.ok) throw new Error("找不到 rec.onnx（可先只跑框）");
      const buf = await fetchWithProgress(REC_URL, "下載 rec.onnx");
      recSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
      keys = await loadKeys();
      console.log("keys.length =", keys.length);

      // ★ 新增：檢查 rec/keys 相容性
      const chk = await verifyRecModelCompatibility();
      if (!chk.ok) {
        // 不相容時直接中止，避免你看到一堆亂碼
        return;
      }
    }
    if (!clsSession) {
      try {
        const h = await fetch(CLS_URL, {method:"HEAD"});
        if (h.ok) {
          const cbuf = await fetchWithProgress(CLS_URL, "下載 cls.onnx");
          clsSession = await ort.InferenceSession.create(cbuf, { executionProviders: ["wasm"] });
        }
      } catch {}
    }

    // det 推論
    result.textContent = "🔎 det 推論中…";
    const meta = imageToDetTensor(imageElement);
    const feeds = {}; feeds[detSession.inputNames[0]] = meta.tensor;
    const detOut = await detSession.run(feeds);
    const detName = Object.keys(detOut)[0];
    const detMap = detOut[detName]; // [1,1,H,W]

    // 嘗試：OpenCV RBOX → 若 0 框，再以寬鬆參數重試 → 再不行走 AABB
    let rboxes640 = [];
    const cvReady = await waitForOpenCV();
    if (cvReady) {
      rboxes640 = dbPostprocess_RBOX(detMap, 0.22, 1.8, 6);
      if (rboxes640.length === 0) rboxes640 = dbPostprocess_RBOX(detMap, 0.18, 2.2, 4);
    } else {
      console.warn("OpenCV 未就緒，改用軸對齊後處理");
    }
    if (rboxes640.length === 0) {
      rboxes640 = probMapToBoxes_AABB(detMap, 0.16, 30);
    }
    if (rboxes640.length === 0) {
      result.textContent = "⚠️ 未偵測到文字區塊（可嘗試拍更近、更直或調高亮度）";
      return;
    }

    const rboxesOrig = rboxes640.map(rb => rboxToOriginal(rb, meta));
    drawRBoxesOnImage(imageElement, rboxesOrig);

    // 分群成 grid
    const grid = gridFromRBoxes(
      rboxesOrig,
      Math.round(imageElement.naturalHeight/120),
      Math.round(imageElement.naturalWidth/180)
    );

    // 逐格裁切 → (cls) → rec
    result.textContent = "🔤 文字識別中…（格數：" + grid.reduce((a,r)=>a+r.length,0) + "）";
    const rowsText = [];
    for (const row of grid){
      const cols = [];
      for (const rb of row){
        let crop = null;
        if (cvReady) crop = cropRotatedRectFromImage(imageElement, rb);
        if (!crop) { // 沒 OpenCV 就用 canvas 直接切 AABB
          const c = document.createElement("canvas");
          c.width = Math.max(1, Math.round(rb.w));
          c.height= Math.max(1, Math.round(rb.h));
          const g = c.getContext("2d");
          g.drawImage(
            imageElement,
            Math.max(0, Math.round(rb.cx - rb.w/2)),
            Math.max(0, Math.round(rb.cy - rb.h/2)),
            Math.round(rb.w), Math.round(rb.h),
            0, 0, Math.round(rb.w), Math.round(rb.h)
          );
          crop = c;
        }
        let text = "";
        try { text = await recognizeCanvas(crop) || ""; } catch(e){ console.warn("rec 失敗:", e); }
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

    result.textContent = "✅ 完成（下方為表格），可下載 Excel";
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
