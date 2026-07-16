/* Geniş sahne görüntüleyici: pan/zoom canvas, kutu seçimi/ekleme, ısı haritası.
   app.js'in globalleri ($, esc, toast) paylaşılır; kendi durumu IIFE içindedir.
   NOT: bu dosyada kaçış dizisi (backslash-escape) KULLANILMAZ — CSV/PNG sunucuda üretilir. */
(() => {
  "use strict";

  const S = {
    cfg: null,            // /api/config yanıtındaki scene bloğu + threshold
    scene: null,          // /api/scene yanıtı
    boxes: [],
    threshold: 0.40,
    selected: null,       // seçili kutu id'si
    img: null,            // display JPEG Image nesnesi
    heatImg: null,
    heatOn: true,
    heatAlpha: 0.55,
    view: { scale: 1, tx: 0, ty: 0, fit: 1 },
    mode: "pan",          // "pan" | "add"
    drag: null,
    addRect: null,
    file: null,
    busy: false,
    timer: null,
  };

  const OK_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];

  /* ---------------- sekmeler ---------------- */
  function wireTabs() {
    const single = $("#tab-single");
    const sceneTab = $("#tab-scene");
    const bSingle = $("#tab-btn-single");
    const bScene = $("#tab-btn-scene");
    bSingle.addEventListener("click", () => {
      single.classList.remove("hidden");
      sceneTab.classList.add("hidden");
      bSingle.classList.add("active");
      bScene.classList.remove("active");
    });
    bScene.addEventListener("click", () => {
      sceneTab.classList.remove("hidden");
      single.classList.add("hidden");
      bScene.classList.add("active");
      bSingle.classList.remove("active");
      resizeCanvas();
    });
  }

  /* ---------------- başlangıç ---------------- */
  async function init() {
    wireTabs();
    try {
      const r = await fetch("/api/config");
      const cfg = await r.json();
      S.cfg = cfg.scene || {};
      S.threshold = cfg.threshold;
      const t = $("#scene-threshold");
      t.value = cfg.threshold.toFixed(2);
      $("#scene-threshold-val").textContent = cfg.threshold.toFixed(2);
      const w = $("#scene-window");
      w.min = S.cfg.window_min;
      w.max = S.cfg.window_max;
      w.step = S.cfg.window_step;
      w.value = S.cfg.window_default;
      $("#scene-window-val").textContent = S.cfg.window_default;
      if (!S.cfg.detector_importable) {
        const radio = $("#mode-yolo");
        radio.disabled = true;
        radio.parentElement.title =
          "ultralytics kurulu değil — İzgara modunu kullanın (pip install ultralytics)";
        radio.parentElement.style.opacity = "0.5";
        $("#mode-grid").checked = true;
      }
      syncWindowVisibility();
    } catch {
      /* config alınamazsa varsayılanlar kalır */
    }
    wireUpload();
    wireControls();
    wireCanvas();
  }

  function currentMode() {
    return $("#mode-grid").checked ? "grid" : "yolo";
  }

  function syncWindowVisibility() {
    $("#scene-window-ctrl").style.display =
      currentMode() === "grid" ? "" : "none";
  }

  /* ---------------- yükleme ---------------- */
  function setFile(f) {
    if (!f) return;
    const name = f.name.toLowerCase();
    if (!OK_EXT.some((e) => name.endsWith(e))) {
      toast(`Desteklenmeyen dosya türü: ${f.name} (sahne için PNG/JPG/TIF)`);
      return;
    }
    S.file = f;
    const mb = (f.size / 1024 / 1024).toFixed(1);
    const el = $("#scene-staged");
    el.textContent = `Seçili sahne: ${f.name} (${mb} MB)`;
    el.classList.remove("hidden");
    $("#scene-analyze").disabled = false;
  }

  function wireUpload() {
    const dz = $("#scene-drop");
    const fi = $("#scene-file-input");
    dz.addEventListener("click", () => fi.click());
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer && e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    fi.addEventListener("change", () => { setFile(fi.files[0]); fi.value = ""; });
  }

  /* ---------------- analiz ---------------- */
  async function analyzeScene() {
    if (!S.file || S.busy) return;
    S.busy = true;
    const btn = $("#scene-analyze");
    const status = $("#scene-status");
    btn.disabled = true;
    status.classList.remove("hidden");
    const t0 = Date.now();
    S.timer = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      status.textContent = `Analiz ediliyor… (${s} sn) — sahne boyutuna göre 10–60 sn sürebilir.`;
    }, 500);
    status.textContent = "Analiz ediliyor…";

    const fd = new FormData();
    fd.append("file", S.file, S.file.name);
    fd.append("mode", currentMode());
    fd.append("threshold", String(S.threshold));
    fd.append("window", $("#scene-window").value);
    try {
      const r = await fetch("/api/scene", { method: "POST", body: fd });
      let data = null;
      try { data = await r.json(); } catch { /* JSON değil */ }
      if (!r.ok) {
        let msg = (data && data.error) || "Sunucuya ulaşılamadı.";
        if (data && data.detector_available === false) {
          msg += " — İzgara modunu deneyin.";
        }
        toast(msg);
        return;
      }
      onSceneLoaded(data);
    } catch {
      toast("Sunucuya ulaşılamadı.");
    } finally {
      clearInterval(S.timer);
      status.classList.add("hidden");
      btn.disabled = !S.file;
      S.busy = false;
    }
  }

  function onSceneLoaded(d) {
    S.scene = d;
    S.boxes = d.buildings.slice();
    S.selected = null;
    S.mode = "pan";
    $("#btn-add-box").classList.remove("active");
    $("#canvas-wrap").classList.remove("adding");

    S.img = new Image();
    S.img.onload = () => { fitView(); render(); };
    S.img.src = d.display.uri;

    S.heatImg = null;
    if (d.heatmap) {
      S.heatImg = new Image();
      S.heatImg.onload = () => render();
      S.heatImg.src = d.heatmap;
    }
    $("#heatmap-row").style.display = d.heatmap ? "" : "none";

    $("#scene-viewer-card").classList.remove("hidden");
    renderPanelDefault();
    relabel();
    const t = d.timings;
    const modeName = d.mode === "yolo" ? "YOLO" : "Izgara";
    toast(`${modeName} analizi bitti: ${t.total_s} sn (${t.n_windows} pencere, ${t.n_refined} rafine)`);
    $("#scene-viewer-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ---------------- görünüm/çizim ---------------- */
  const canvas = () => $("#scene-canvas");

  function resizeCanvas() {
    const cv = canvas();
    const wrap = $("#canvas-wrap");
    if (!cv || !wrap) return;
    const hh = window.innerWidth <= 900 ? 420 : 560;
    if (cv.width !== wrap.clientWidth || cv.height !== hh) {
      cv.width = wrap.clientWidth;
      cv.height = hh;
    }
    render();
  }

  function fitView() {
    const cv = canvas();
    const wrap = $("#canvas-wrap");
    cv.width = wrap.clientWidth;
    cv.height = window.innerWidth <= 900 ? 420 : 560;
    if (!S.img) return;
    const fit = Math.min(cv.width / S.img.width, cv.height / S.img.height);
    S.view.fit = fit;
    S.view.scale = fit;
    S.view.tx = (cv.width - S.img.width * fit) / 2;
    S.view.ty = (cv.height - S.img.height * fit) / 2;
  }

  function visibleBoxes() {
    return S.boxes.filter((b) => !(b.source === "grid" && b.p < S.threshold));
  }

  function render() {
    const cv = canvas();
    if (!cv || !S.img) return;
    const ctx = cv.getContext("2d");
    const v = S.view;
    const ds = S.scene.display.scale;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.setTransform(v.scale, 0, 0, v.scale, v.tx, v.ty);
    ctx.imageSmoothingEnabled = v.scale < 3;
    ctx.drawImage(S.img, 0, 0);
    if (S.heatImg && S.heatOn) {
      ctx.globalAlpha = S.heatAlpha;
      ctx.drawImage(S.heatImg, 0, 0, S.img.width, S.img.height);
      ctx.globalAlpha = 1;
    }
    const lw = 2 / v.scale;
    for (const b of visibleBoxes()) {
      const damaged = b.p >= S.threshold;
      ctx.strokeStyle = damaged ? "#dc2626" : "#16a34a";
      ctx.lineWidth = b.id === S.selected ? lw * 2 : lw;
      ctx.strokeRect(b.x * ds, b.y * ds, b.w * ds, b.h * ds);
      if (b.id === S.selected) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = lw;
        ctx.setLineDash([6 / v.scale, 4 / v.scale]);
        ctx.strokeRect(b.x * ds - lw * 2, b.y * ds - lw * 2,
                       b.w * ds + lw * 4, b.h * ds + lw * 4);
        ctx.setLineDash([]);
      }
    }
    if (S.addRect) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = lw;
      ctx.setLineDash([6 / v.scale, 4 / v.scale]);
      const r = S.addRect;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
    }
  }

  /* ekran -> display-görüntü pikseli */
  function toDisp(mx, my) {
    const v = S.view;
    return { x: (mx - v.tx) / v.scale, y: (my - v.ty) / v.scale };
  }
  /* display pikseli -> sahne pikseli */
  function toSceneCoord(p) {
    const ds = S.scene.display.scale;
    return { x: p.x / ds, y: p.y / ds };
  }

  function hitTest(mx, my) {
    if (!S.scene) return null;
    const p = toSceneCoord(toDisp(mx, my));
    const vis = visibleBoxes();
    for (let i = vis.length - 1; i >= 0; i--) {
      const b = vis[i];
      if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) return b;
    }
    return null;
  }

  /* ---------------- canvas etkileşimi ---------------- */
  function wireCanvas() {
    const cv = canvas();
    const tip = $("#scene-tooltip");

    cv.addEventListener("wheel", (e) => {
      if (!S.img) return;
      e.preventDefault();
      const f0 = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const v = S.view;
      const ns = Math.min(Math.max(v.scale * f0, v.fit * 0.5), 12);
      const f = ns / v.scale;
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      v.tx = mx - (mx - v.tx) * f;
      v.ty = my - (my - v.ty) * f;
      v.scale = ns;
      render();
    }, { passive: false });

    cv.addEventListener("mousedown", (e) => {
      if (!S.img) return;
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (S.mode === "add") {
        const d = toDisp(mx, my);
        S.drag = { kind: "draw", sx: d.x, sy: d.y };
        S.addRect = { x: d.x, y: d.y, w: 0, h: 0 };
      } else {
        S.drag = { kind: "pan", mx, my, moved: false };
        cv.classList.add("panning");
      }
    });

    cv.addEventListener("mousemove", (e) => {
      if (!S.img) return;
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (S.drag && S.drag.kind === "pan") {
        const dx = mx - S.drag.mx;
        const dy = my - S.drag.my;
        if (Math.abs(dx) + Math.abs(dy) > 4) S.drag.moved = true;
        S.view.tx += dx;
        S.view.ty += dy;
        S.drag.mx = mx;
        S.drag.my = my;
        render();
        return;
      }
      if (S.drag && S.drag.kind === "draw") {
        const d = toDisp(mx, my);
        S.addRect = {
          x: Math.min(S.drag.sx, d.x), y: Math.min(S.drag.sy, d.y),
          w: Math.abs(d.x - S.drag.sx), h: Math.abs(d.y - S.drag.sy),
        };
        render();
        return;
      }
      const b = hitTest(mx, my);
      if (b) {
        const lab = b.p >= S.threshold ? "HASARLI" : "SAĞLAM";
        const src = { yolo: "YOLO", grid: "ızgara", manual: "manuel" }[b.source];
        tip.textContent = `#${b.id} — p=${b.p.toFixed(2)} · ${lab} · ${src}`;
        tip.style.left = `${mx + 14}px`;
        tip.style.top = `${my + 14}px`;
        tip.classList.remove("hidden");
      } else {
        tip.classList.add("hidden");
      }
    });

    const endDrag = async (e) => {
      if (!S.drag) return;
      const drag = S.drag;
      S.drag = null;
      cv.classList.remove("panning");
      if (drag.kind === "pan") {
        if (!drag.moved && e && e.target === cv) {
          const rect = cv.getBoundingClientRect();
          const b = hitTest(e.clientX - rect.left, e.clientY - rect.top);
          selectBox(b ? b.id : null);
        }
        return;
      }
      // draw bitti -> sunucuya gönder
      const r = S.addRect;
      S.addRect = null;
      render();
      if (!r || r.w < 6 || r.h < 6) return;
      const sc = S.scene.display.scale;
      const payload = {
        x: Math.round(r.x / sc), y: Math.round(r.y / sc),
        w: Math.round(r.w / sc), h: Math.round(r.h / sc),
      };
      try {
        const resp = await fetch(`/api/scene/${S.scene.scene_id}/box`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) {
          toast(data.error || "Kutu eklenemedi.");
          return;
        }
        S.boxes.push(data.box);
        exitAddMode();
        selectBox(data.box.id);
        relabel();
      } catch {
        toast("Sunucuya ulaşılamadı.");
      }
    };
    cv.addEventListener("mouseup", endDrag);
    cv.addEventListener("mouseleave", () => { tip.classList.add("hidden"); });
    window.addEventListener("mouseup", (e) => { if (S.drag) endDrag(e); });

    window.addEventListener("keydown", (e) => {
      if ($("#tab-scene").classList.contains("hidden")) return;
      if (e.key === "Escape") {
        if (S.mode === "add") exitAddMode();
        else selectBox(null);
      }
      if (e.key === "Delete" && S.selected != null) deleteSelected();
    });

    window.addEventListener("resize", resizeCanvas);
  }

  function enterAddMode() {
    if (!S.scene) return;
    S.mode = "add";
    $("#btn-add-box").classList.add("active");
    $("#canvas-wrap").classList.add("adding");
  }
  function exitAddMode() {
    S.mode = "pan";
    S.addRect = null;
    $("#btn-add-box").classList.remove("active");
    $("#canvas-wrap").classList.remove("adding");
    render();
  }

  /* ---------------- seçim paneli ---------------- */
  function findBox(id) {
    return S.boxes.find((b) => b.id === id) || null;
  }

  function selectBox(id) {
    S.selected = id;
    render();
    if (id == null) renderPanelDefault();
    else renderPanel(findBox(id));
  }

  function renderPanelDefault() {
    $("#panel-content").innerHTML =
      `<p class="hint">Bir kutuya tıklayın: olasılık, kaynak ve Grad-CAM burada görünür.
        Fare tekerleği ile yakınlaşın, sürükleyerek gezinin.
        <strong>Kutu Ekle</strong> ile kaçırılan binayı kendiniz çizebilirsiniz.</p>`;
  }

  function chipPreviewDataUri(b) {
    // display görüntüsünden kutu+%15 pay kırp (panelde hızlı önizleme)
    const ds = S.scene.display.scale;
    const pad = Math.max(b.w, b.h) * 0.15;
    const x = Math.max(0, (b.x - pad) * ds);
    const y = Math.max(0, (b.y - pad) * ds);
    const w = Math.min(S.img.width - x, (b.w + 2 * pad) * ds);
    const h = Math.min(S.img.height - y, (b.h + 2 * pad) * ds);
    const off = document.createElement("canvas");
    const side = 180;
    off.width = side;
    off.height = side;
    off.getContext("2d").drawImage(S.img, x, y, w, h, 0, 0, side, side);
    return off.toDataURL("image/jpeg", 0.85);
  }

  function renderPanel(b) {
    if (!b) { renderPanelDefault(); return; }
    const damaged = b.p >= S.threshold;
    const srcName = { yolo: "YOLO tespiti", grid: "Izgara hücresi", manual: "Manuel kutu" }[b.source];
    const chip = chipPreviewDataUri(b);
    const rows = [
      ["Kutu", `#${b.id}`],
      ["Kaynak", srcName],
      ["p(hasarlı)", b.p.toFixed(4)],
      ["Konum", `${b.x}, ${b.y}`],
      ["Boyut", `${b.w} × ${b.h} px`],
    ];
    if (b.det_conf != null) rows.splice(2, 0, ["Tespit güveni", b.det_conf.toFixed(2)]);
    $("#panel-content").innerHTML = `
      <img id="panel-img" src="${chip}" alt="kesit">
      <div class="verdict" style="margin-top:10px">
        <span class="label ${damaged ? "damaged" : "intact"}">${damaged ? "HASARLI" : "SAĞLAM"}</span>
      </div>
      ${rows.map(([k, v]) => `<div class="prow"><span class="k">${k}</span><span>${esc(String(v))}</span></div>`).join("")}
      <div class="pbtns">
        <button id="btn-cam" class="btn small">Grad-CAM</button>
        <button id="btn-del" class="btn small danger-ghost">Kutuyu Sil</button>
      </div>`;
    $("#btn-del").addEventListener("click", deleteSelected);
    $("#btn-cam").addEventListener("click", async () => {
      const img = $("#panel-img");
      const btn = $("#btn-cam");
      if (img.dataset.showing === "cam") {
        img.src = chip;
        img.dataset.showing = "chip";
        btn.textContent = "Grad-CAM";
        return;
      }
      if (!b._cam) {
        btn.disabled = true;
        btn.textContent = "Hesaplanıyor…";
        try {
          const r = await fetch(`/api/scene/${S.scene.scene_id}/cam?box=${b.id}`);
          const data = await r.json();
          if (!r.ok) { toast(data.error || "CAM üretilemedi."); return; }
          b._cam = data.cam;
        } catch {
          toast("Sunucuya ulaşılamadı.");
          return;
        } finally {
          btn.disabled = false;
        }
      }
      img.src = b._cam;
      img.dataset.showing = "cam";
      btn.textContent = "Orijinal";
    });
  }

  async function deleteSelected() {
    const id = S.selected;
    if (id == null || !S.scene) return;
    try {
      const r = await fetch(`/api/scene/${S.scene.scene_id}/box/${id}`, { method: "DELETE" });
      if (!r.ok) {
        const data = await r.json().catch(() => null);
        toast((data && data.error) || "Kutu silinemedi.");
        return;
      }
      S.boxes = S.boxes.filter((b) => b.id !== id);
      selectBox(null);
      relabel();
    } catch {
      toast("Sunucuya ulaşılamadı.");
    }
  }

  /* ---------------- eşik/özet/indirme ---------------- */
  function relabel() {
    if (!S.scene) return;
    let dmg = 0, safe = 0;
    const vis = visibleBoxes();
    for (const b of vis) {
      if (b.p >= S.threshold) dmg++;
      else safe++;
    }
    const modeName = S.scene.mode === "yolo" ? "YOLO" : "Izgara";
    if (S.scene.mode === "grid") {
      $("#scene-summary").textContent =
        `${vis.length} işaretli hücre: ${dmg} hasarlı (eşik ${S.threshold.toFixed(2)}) — ${modeName}`;
    } else {
      $("#scene-summary").textContent =
        `${vis.length} bina: ${dmg} hasarlı, ${safe} sağlam (eşik ${S.threshold.toFixed(2)}) — ${modeName}`;
    }
    const sid = S.scene.scene_id;
    const t = S.threshold.toFixed(2);
    const heat = S.heatOn && S.heatImg ? 1 : 0;
    $("#btn-scene-png").href = `/api/scene/${sid}/annotated.png?threshold=${t}&heatmap=${heat}`;
    $("#btn-scene-csv").href = `/api/scene/${sid}/csv?threshold=${t}`;
    render();
    if (S.selected != null) renderPanel(findBox(S.selected));
  }

  function wireControls() {
    $("#scene-analyze").addEventListener("click", analyzeScene);
    $("#mode-yolo").addEventListener("change", syncWindowVisibility);
    $("#mode-grid").addEventListener("change", syncWindowVisibility);
    $("#scene-window").addEventListener("input", () => {
      $("#scene-window-val").textContent = $("#scene-window").value;
    });
    $("#scene-threshold").addEventListener("input", () => {
      S.threshold = parseFloat($("#scene-threshold").value);
      $("#scene-threshold-val").textContent = S.threshold.toFixed(2);
      relabel();
    });
    $("#btn-add-box").addEventListener("click", () => {
      if (S.mode === "add") exitAddMode();
      else enterAddMode();
    });
    $("#btn-fit").addEventListener("click", () => { fitView(); render(); });
    $("#heatmap-toggle").addEventListener("change", () => {
      S.heatOn = $("#heatmap-toggle").checked;
      relabel();
    });
    $("#heatmap-alpha").addEventListener("input", () => {
      S.heatAlpha = parseInt($("#heatmap-alpha").value, 10) / 100;
      render();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
