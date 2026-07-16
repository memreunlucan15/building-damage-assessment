/* Deprem Hasar Tespiti — arayüz mantığı (vanilla JS, bağımlılık yok) */
"use strict";

const state = {
  cfg: null,
  staged: [],      // {file, name, kind: 'img'|'raw', url|null}
  results: [],     // sunucu yanıtındaki results öğeleri
  threshold: 0.40,
};

const $ = (sel) => document.querySelector(sel);
const IMG_TYPES = ["image/png", "image/jpeg"];
const OK_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mat"];

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

let toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 4000);
}

/* ---------------- başlangıç ---------------- */
async function init() {
  try {
    const [cfgR, samR] = await Promise.all([fetch("/api/config"), fetch("/api/samples")]);
    state.cfg = await cfgR.json();
    renderBadges();
    const slider = $("#threshold");
    slider.value = state.cfg.threshold.toFixed(2);
    state.threshold = state.cfg.threshold;
    $("#threshold-val").textContent = state.cfg.threshold.toFixed(2);
    $("#max-files").textContent = state.cfg.max_files;
    if (!state.cfg.model_ready) $("#model-warning").classList.remove("hidden");
    const sam = await samR.json();
    if (sam.available) renderSamples(sam.samples);
  } catch {
    toast("Sunucu yapılandırması alınamadı.");
  }
  wireEvents();
}

function renderBadges() {
  const c = state.cfg;
  const dev = c.device === "cuda" ? `CUDA — ${c.device_name}` : "CPU";
  $("#badges").innerHTML = [
    `Cihaz: ${esc(dev)}`,
    `Model: ${c.n_fold_files} fold ensemble${c.tta ? " + TTA" : ""}`,
    `Varsayılan eşik: ${c.threshold.toFixed(2)}`,
  ].map((b) => `<span class="badge">${b}</span>`).join("");
}

function renderSamples(samples) {
  const strip = $("#samples-strip");
  strip.innerHTML = "";
  for (const s of samples) {
    const div = document.createElement("div");
    div.className = "sample";
    div.title = `${s.id} — analiz etmek için tıklayın`;
    div.innerHTML = `<img src="${s.thumb}" alt="${esc(s.id)}">
      <span class="tag ${s.cls}">${s.cls === "damaged" ? "hasarlı" : "sağlam"}</span>`;
    div.addEventListener("click", () => analyze([s.ref]));
    strip.appendChild(div);
  }
  $("#samples-card").classList.remove("hidden");
}

/* ---------------- dosya hazırlama ---------------- */
function addFiles(fileList) {
  const maxN = state.cfg ? state.cfg.max_files : 16;
  for (const f of fileList) {
    if (state.staged.length >= maxN) {
      toast(`En fazla ${maxN} dosya yükleyebilirsiniz.`);
      break;
    }
    const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
    if (!OK_EXT.includes(ext)) {
      toast(`Desteklenmeyen dosya türü: ${f.name}`);
      continue;
    }
    const kind = IMG_TYPES.includes(f.type) ? "img" : "raw";
    state.staged.push({
      file: f, name: f.name, kind,
      url: kind === "img" ? URL.createObjectURL(f) : null,
    });
  }
  renderStaged();
}

function removeStaged(i) {
  const it = state.staged[i];
  if (it && it.url) URL.revokeObjectURL(it.url);
  state.staged.splice(i, 1);
  renderStaged();
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function renderStaged() {
  const ul = $("#staged");
  ul.innerHTML = "";
  state.staged.forEach((it, i) => {
    const li = document.createElement("li");
    const preview = it.kind === "img"
      ? `<img class="preview" src="${it.url}" alt="">`
      : `<span class="file-icon">${it.name.toLowerCase().endsWith(".mat") ? "MAT" : "TIF"}</span>`;
    const cropBtn = it.kind === "img"
      ? `<button class="btn small" data-crop="${i}">Kırp</button>`
      : `<button class="btn small" disabled title="Kırpma yalnızca PNG/JPG dosyaları için">Kırp</button>`;
    li.innerHTML = `${preview}
      <span class="fname" title="${esc(it.name)}">${esc(it.name)}</span>
      <span class="fsize">${fmtSize(it.file.size)}</span>
      ${cropBtn}
      <button class="btn small danger-ghost" data-rm="${i}">Kaldır</button>`;
    ul.appendChild(li);
  });
  $("#analyze").disabled = state.staged.length === 0;
  $("#clear-staged").classList.toggle("hidden", state.staged.length === 0);
}

/* ---------------- kırpma modalı ---------------- */
const crop = { idx: -1, img: null, scale: 1, natW: 0, natH: 0, rect: null, dragging: false, start: null };

function openCrop(i) {
  const it = state.staged[i];
  if (!it || it.kind !== "img") return;
  crop.idx = i;
  crop.rect = null;
  crop.dragging = false;
  const img = new Image();
  img.onload = () => {
    crop.img = img;
    crop.natW = img.naturalWidth;
    crop.natH = img.naturalHeight;
    crop.scale = Math.min(720 / crop.natW, 520 / crop.natH, 1);
    const cv = $("#crop-canvas");
    cv.width = Math.round(crop.natW * crop.scale);
    cv.height = Math.round(crop.natH * crop.scale);
    drawCrop();
    $("#crop-size").textContent = "—";
    $("#crop-apply").disabled = true;
    $("#crop-modal").classList.remove("hidden");
  };
  img.src = it.url;
}

function closeCrop() {
  $("#crop-modal").classList.add("hidden");
  crop.idx = -1;
  crop.img = null;
}

function drawCrop() {
  const cv = $("#crop-canvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(crop.img, 0, 0, cv.width, cv.height);
  if (!crop.rect) return;
  const { x, y, w, h } = crop.rect;
  ctx.fillStyle = "rgba(15,23,42,.5)";              // seçim dışını karart
  ctx.fillRect(0, 0, cv.width, y);
  ctx.fillRect(0, y + h, cv.width, cv.height - y - h);
  ctx.fillRect(0, y, x, h);
  ctx.fillRect(x + w, y, cv.width - x - w, h);
  ctx.strokeStyle = "#fff";
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  ctx.setLineDash([]);
}

function canvasPos(e) {
  const cv = $("#crop-canvas");
  const r = cv.getBoundingClientRect();
  return {
    x: Math.min(Math.max(e.clientX - r.left, 0), cv.width),
    y: Math.min(Math.max(e.clientY - r.top, 0), cv.height),
  };
}

function updateCropRect(p) {
  const s = crop.start;
  crop.rect = {
    x: Math.min(s.x, p.x), y: Math.min(s.y, p.y),
    w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y),
  };
  drawCrop();
  const nw = Math.round(crop.rect.w / crop.scale);
  const nh = Math.round(crop.rect.h / crop.scale);
  $("#crop-size").textContent = `${nw}×${nh} px`;
  $("#crop-apply").disabled = !(nw >= 32 && nh >= 32);
}

function applyCrop() {
  const { rect, scale, img, idx } = crop;
  if (!rect || idx < 0) return;
  const sx = Math.max(0, Math.round(rect.x / scale));
  const sy = Math.max(0, Math.round(rect.y / scale));
  const sw = Math.min(crop.natW - sx, Math.round(rect.w / scale));
  const sh = Math.min(crop.natH - sy, Math.round(rect.h / scale));
  const off = document.createElement("canvas");
  off.width = sw;
  off.height = sh;
  off.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  off.toBlob((blob) => {
    if (!blob) { toast("Kırpma başarısız oldu."); return; }
    const it = state.staged[idx];
    const base = it.name.replace(/\.[^.]+$/, "");
    if (it.url) URL.revokeObjectURL(it.url);
    it.file = new File([blob], `${base}_kirpilmis.png`, { type: "image/png" });
    it.name = it.file.name;
    it.url = URL.createObjectURL(it.file);
    renderStaged();
    closeCrop();
  }, "image/png");
}

/* ---------------- analiz ---------------- */
async function analyze(sampleRefs = null) {
  const btn = $("#analyze");
  const fd = new FormData();
  if (sampleRefs) {
    for (const r of sampleRefs) fd.append("sample_refs", r);
  } else {
    if (!state.staged.length) return;
    for (const it of state.staged) fd.append("files", it.file, it.name);
  }
  fd.append("threshold", String(state.threshold));
  fd.append("gradcam", $("#gradcam-toggle").checked ? "1" : "0");

  const oldTxt = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Analiz ediliyor…";
  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    let data = null;
    try { data = await r.json(); } catch { /* gövde JSON değil */ }
    if (!r.ok) {
      toast((data && data.error) || "Sunucuya ulaşılamadı.");
      return;
    }
    state.results = data.results;
    renderResults();
    if (!sampleRefs) {
      state.staged.forEach((it) => it.url && URL.revokeObjectURL(it.url));
      state.staged = [];
      renderStaged();
    }
  } catch {
    toast("Sunucuya ulaşılamadı.");
  } finally {
    btn.textContent = oldTxt;
    btn.disabled = state.staged.length === 0;
  }
}

/* ---------------- sonuç görünümü ---------------- */
function labelOf(r) {
  return r.p_damaged >= state.threshold ? "damaged" : "intact";
}

function renderResults() {
  const cards = $("#cards");
  cards.innerHTML = "";
  state.results.forEach((r, i) => {
    if (!r.ok) {
      const div = document.createElement("div");
      div.className = "result-card error-card";
      div.innerHTML = `<div class="rname" title="${esc(r.name)}"><strong>${esc(r.name)}</strong></div>
        <div class="err">${esc(r.error)}</div>`;
      cards.appendChild(div);
      return;
    }
    const div = document.createElement("div");
    div.className = "result-card";
    div.dataset.idx = i;
    const camBtn = r.cam
      ? `<button class="camswitch" data-cam="${i}">Grad-CAM</button>` : "";
    div.innerHTML = `
      <div class="imgwrap"><img src="${r.thumb}" data-showing="thumb" alt="">${camBtn}</div>
      <div class="body">
        <span class="rname" title="${esc(r.name)}">${esc(r.name)}</span>
        <div class="verdict"></div>
        <div class="pbar"><div style="width:${(r.p_damaged * 100).toFixed(1)}%"></div></div>
        ${r.true_cls ? `<span class="truth">Gerçek etiket: ${r.true_cls === "damaged" ? "hasarlı" : "sağlam"}</span>` : ""}
      </div>`;
    cards.appendChild(div);
  });
  relabel();
  $("#results-card").classList.remove("hidden");
  $("#results-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* Eşik değişince yalnızca etiketler yeniden hesaplanır — ağ çağrısı YOK. */
function relabel() {
  const tbody = $("#results-table tbody");
  tbody.innerHTML = "";
  let d = 0, s = 0, err = 0;
  state.results.forEach((r) => {
    const tr = document.createElement("tr");
    if (!r.ok) {
      err++;
      tr.innerHTML = `<td>${esc(r.name)}</td><td colspan="3" class="t-dmg">${esc(r.error)}</td>`;
      tbody.appendChild(tr);
      return;
    }
    const lab = labelOf(r);
    lab === "damaged" ? d++ : s++;
    tr.innerHTML = `<td>${esc(r.name)}</td>
      <td>${r.p_damaged.toFixed(4)}</td>
      <td class="${lab === "damaged" ? "t-dmg" : "t-int"}">${lab === "damaged" ? "HASARLI" : "SAĞLAM"}</td>
      <td>${r.true_cls ? (r.true_cls === "damaged" ? "hasarlı" : "sağlam") : "—"}</td>`;
    tbody.appendChild(tr);
  });
  $("#summary").textContent =
    `${d + s} görüntü: ${d} hasarlı, ${s} sağlam` + (err ? `, ${err} hata` : "") +
    ` (eşik ${state.threshold.toFixed(2)})`;

  document.querySelectorAll("#cards .result-card:not(.error-card)").forEach((card) => {
    const r = state.results[Number(card.dataset.idx)];
    const lab = labelOf(r);
    card.querySelector(".verdict").innerHTML =
      `<span class="label ${lab}">${lab === "damaged" ? "HASARLI" : "SAĞLAM"}</span>
       <span class="pval">p = ${r.p_damaged.toFixed(3)}</span>`;
    card.querySelector(".pbar > div").className = lab;
  });
}

/* ---------------- CSV ---------------- */
function csvField(v) {
  const s = String(v);
  return /[;"\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function downloadCsv() {
  const rows = [["dosya", "kaynak", "p_hasarli", "esik", "tahmin", "gercek_etiket"]];
  for (const r of state.results) {
    if (!r.ok) continue;
    rows.push([
      r.name,
      r.source === "sample" ? "örnek" : "yükleme",
      r.p_damaged.toFixed(4),
      state.threshold.toFixed(2),
      labelOf(r) === "damaged" ? "HASARLI" : "SAĞLAM",
      r.true_cls ? (r.true_cls === "damaged" ? "hasarlı" : "sağlam") : "—",
    ]);
  }
  if (rows.length === 1) { toast("İndirilecek başarılı sonuç yok."); return; }
  const csv = "\ufeff" + rows.map((row) => row.map(csvField).join(";")).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "tahminler.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

/* ---------------- olaylar ---------------- */
function wireEvents() {
  const dz = $("#dropzone");
  const fi = $("#file-input");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  fi.addEventListener("change", () => { addFiles(fi.files); fi.value = ""; });

  $("#staged").addEventListener("click", (e) => {
    const rm = e.target.closest("[data-rm]");
    if (rm) { removeStaged(Number(rm.dataset.rm)); return; }
    const cr = e.target.closest("[data-crop]");
    if (cr) openCrop(Number(cr.dataset.crop));
  });

  $("#clear-staged").addEventListener("click", () => {
    state.staged.forEach((it) => it.url && URL.revokeObjectURL(it.url));
    state.staged = [];
    renderStaged();
  });

  $("#analyze").addEventListener("click", () => analyze());

  const slider = $("#threshold");
  slider.addEventListener("input", () => {
    state.threshold = parseFloat(slider.value);
    $("#threshold-val").textContent = state.threshold.toFixed(2);
    if (state.results.length) relabel();
  });

  $("#cards").addEventListener("click", (e) => {
    const b = e.target.closest("[data-cam]");
    if (!b) return;
    const r = state.results[Number(b.dataset.cam)];
    const img = b.parentElement.querySelector("img");
    const showingThumb = img.dataset.showing === "thumb";
    img.src = showingThumb ? r.cam : r.thumb;
    img.dataset.showing = showingThumb ? "cam" : "thumb";
    b.textContent = showingThumb ? "Orijinal" : "Grad-CAM";
  });

  $("#csv").addEventListener("click", downloadCsv);
  $("#clear-results").addEventListener("click", () => {
    state.results = [];
    $("#results-card").classList.add("hidden");
  });

  // kırpma modalı
  const cv = $("#crop-canvas");
  cv.addEventListener("mousedown", (e) => {
    crop.dragging = true;
    crop.start = canvasPos(e);
    updateCropRect(crop.start);
  });
  cv.addEventListener("mousemove", (e) => { if (crop.dragging) updateCropRect(canvasPos(e)); });
  window.addEventListener("mouseup", () => { crop.dragging = false; });
  $("#crop-apply").addEventListener("click", applyCrop);
  $("#crop-cancel").addEventListener("click", closeCrop);
  $("#crop-modal").addEventListener("click", (e) => {
    if (e.target === $("#crop-modal")) closeCrop();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#crop-modal").classList.contains("hidden")) closeCrop();
  });
}

document.addEventListener("DOMContentLoaded", init);
