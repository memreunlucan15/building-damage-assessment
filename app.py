"""Deprem hasar tespiti web arayuzu (Flask).

Calistirma:  python app.py  ->  http://127.0.0.1:5000
Modeller ilk istekte (veya __main__ on-yuklemesinde) bir kez yuklenir.
Tum /api/* yanitlari JSON'dur; hatalar {"error": "<Turkce mesaj>"} bicimindedir.
"""
import re
import traceback

import torch
from flask import Flask, Response, jsonify, render_template, request

import config
import dataset
import detector
import inference
import scene
from ensemble import DEPLOY_THRESHOLD

MAX_FILES = 16
SAMPLES_PER_CLASS = 4
_SAMPLE_REF_RE = re.compile(r"^(damaged|intact)/([A-Za-z0-9_\-]+)$")


def _parse_threshold(values) -> float:
    """request.form veya request.args kabul eder (ikisi de .get destekler)."""
    raw = (values.get("threshold") or "").strip().replace(",", ".")
    if not raw:
        return DEPLOY_THRESHOLD
    try:
        t = float(raw)
    except ValueError:
        raise inference.UploadError("Geçersiz eşik değeri (0 ile 1 arasında olmalı).")
    if not 0.0 < t < 1.0:
        raise inference.UploadError("Geçersiz eşik değeri (0 ile 1 arasında olmalı).")
    return t


def _resolve_sample(ref: str):
    """'damaged/123' referansini dogrular -> (path, id, cls). Yol gezinmesine kapali."""
    m = _SAMPLE_REF_RE.match(ref or "")
    if not m:
        raise inference.UploadError(f"Geçersiz örnek referansı: {ref!r}")
    cls, sid = m.group(1), m.group(2)
    path = config.DATA_DIR / cls / f"{sid}_opt.mat"   # DATA_DIR cagri aninda okunur
    if not path.exists():
        raise inference.UploadError(f"Örnek bulunamadı: {ref}")
    return path, sid, cls


def _list_samples():
    """Sinif basina ilk SAMPLES_PER_CLASS ornegi kucuk onizlemeyle listeler."""
    out = []
    for cls in ("damaged", "intact"):
        d = config.DATA_DIR / cls
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*_opt.mat"))[:SAMPLES_PER_CLASS]:
            sid = path.stem[:-len("_opt")]
            pil = dataset.load_optical_image(path)
            out.append({"ref": f"{cls}/{sid}", "id": sid, "cls": cls,
                        "thumb": inference.thumb_data_uri(pil, 96)})
    return out


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024   # genis sahne yuklemeleri
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/config")
    def api_config():
        is_cuda = config.DEVICE.type == "cuda"
        n_folds = inference.fold_count()
        has_samples = any((config.DATA_DIR / c).is_dir()
                          and any((config.DATA_DIR / c).glob("*_opt.mat"))
                          for c in ("damaged", "intact"))
        return jsonify({
            "threshold": DEPLOY_THRESHOLD,
            "device": config.DEVICE.type,
            "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
            "n_fold_files": n_folds,
            "model_ready": n_folds > 0,
            "models_loaded": inference.models_loaded(),
            "img_size": config.IMG_SIZE,
            "max_files": MAX_FILES,
            "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            "tta": config.USE_TTA,
            "samples_available": has_samples,
            "scene": {
                "detector_importable": detector.importable(),
                "detector_weights": detector.weights_present(),
                "max_side": scene.MAX_SIDE,
                "max_mp": inference.MAX_PIXELS // 1_000_000,
                "window_min": scene.WINDOW_MIN,
                "window_max": scene.WINDOW_MAX,
                "window_default": scene.WINDOW_DEFAULT,
                "window_step": scene.WINDOW_STEP,
                "candidate_thr": scene.CANDIDATE_THR,
            },
        })

    @app.get("/api/samples")
    def api_samples():
        try:
            samples = _list_samples()
        except Exception:
            traceback.print_exc()
            samples = []
        return jsonify({"available": bool(samples), "samples": samples})

    @app.post("/api/predict")
    def api_predict():
        try:
            threshold = _parse_threshold(request.form)
        except inference.UploadError as e:
            return jsonify({"error": str(e)}), 400
        want_cam = request.form.get("gradcam", "1") != "0"

        # --- girdileri coz (torch yok; oge bazli hata batch'i durdurmaz) ---
        items = []
        for f in request.files.getlist("files"):
            item = {"name": f.filename or "dosya", "source": "upload",
                    "true_cls": None, "pil": None, "error": None}
            try:
                item["pil"] = inference.load_image_any(f.read(), f.filename or "")
            except inference.UploadError as e:
                item["error"] = str(e)
            items.append(item)
        for ref in request.form.getlist("sample_refs"):
            item = {"name": ref, "source": "sample",
                    "true_cls": None, "pil": None, "error": None}
            try:
                path, sid, cls = _resolve_sample(ref)
                item["name"] = f"{sid}_opt.mat"
                item["true_cls"] = cls
                item["pil"] = dataset.load_optical_image(path)
            except inference.UploadError as e:
                item["error"] = str(e)
            items.append(item)

        if not items:
            return jsonify({"error": "En az bir dosya veya örnek seçmelisiniz."}), 400
        if len(items) > MAX_FILES:
            return jsonify({"error": f"En fazla {MAX_FILES} görüntü analiz edilebilir."}), 400

        # --- tahmin + Grad-CAM (yalnizca cozulebilenler) ---
        ok_items = [it for it in items if it["pil"] is not None]
        if ok_items:
            pils = [it["pil"] for it in ok_items]
            try:
                probs = inference.predict_pils(pils)
                cams = inference.gradcam_pils(pils) if want_cam else [None] * len(pils)
            except FileNotFoundError:
                return jsonify({"error": "Model dosyaları bulunamadı (outputs/cv_opt/fold*.pt). "
                                         "Önce cross_validate_mm.py ile modelleri eğitin."}), 503
            for it, p, cam in zip(ok_items, probs, cams):
                it["p"] = float(p)
                it["cam"] = cam

        # --- yanit (orijinal sira korunur) ---
        results = []
        for it in items:
            if it["pil"] is None:
                results.append({"name": it["name"], "source": it["source"], "ok": False,
                                "p_damaged": None, "prediction": None,
                                "true_cls": it["true_cls"], "thumb": None, "cam": None,
                                "error": it["error"]})
                continue
            cam_uri = None
            if it.get("cam") is not None:
                cam_uri = inference.png_data_uri(inference.overlay_cam(it["pil"], it["cam"]))
            results.append({
                "name": it["name"], "source": it["source"], "ok": True,
                "p_damaged": round(it["p"], 6),
                "prediction": "damaged" if it["p"] >= threshold else "intact",
                "true_cls": it["true_cls"],
                "thumb": inference.thumb_data_uri(it["pil"]),
                "cam": cam_uri, "error": None,
            })
        return jsonify({"threshold": threshold, "results": results})

    # --- genis sahne rotalari ---
    def _get_scene_or_404(sid):
        entry = scene.STORE.get(sid)
        if entry is None:
            return None, (jsonify({"error": "Sahne bulunamadı veya bellekten çıkarıldı — "
                                            "görüntüyü yeniden analiz edin."}), 404)
        return entry, None

    @app.post("/api/scene")
    def api_scene():
        try:
            threshold = _parse_threshold(request.form)
        except inference.UploadError as e:
            return jsonify({"error": str(e)}), 400
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "Bir sahne görüntüsü yüklemelisiniz."}), 400
        mode = request.form.get("mode", "yolo")
        try:
            window = int(request.form.get("window", scene.WINDOW_DEFAULT))
        except ValueError:
            return jsonify({"error": "Geçersiz pencere boyutu."}), 400
        try:
            pil = inference.load_image_any(f.read(), f.filename, max_side=None)
            result = scene.analyze_scene(pil, mode=mode, threshold=threshold,
                                         window=window)
        except inference.UploadError as e:
            return jsonify({"error": str(e)}), 400
        except detector.DetectorUnavailable as e:
            return jsonify({"error": str(e), "detector_available": False}), 400
        except FileNotFoundError:
            return jsonify({"error": "Model dosyaları bulunamadı (outputs/cv_opt/fold*.pt). "
                                     "Önce cross_validate_mm.py ile modelleri eğitin."}), 503
        return jsonify(result)

    @app.post("/api/scene/<sid>/box")
    def api_scene_add_box(sid):
        entry, err = _get_scene_or_404(sid)
        if err:
            return err
        d = request.get_json(silent=True) or {}
        try:
            box = scene.classify_manual_box(entry, d.get("x"), d.get("y"),
                                            d.get("w"), d.get("h"))
        except inference.UploadError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError:
            return jsonify({"error": "Model dosyaları bulunamadı."}), 503
        return jsonify({"box": box})

    @app.delete("/api/scene/<sid>/box/<int:bid>")
    def api_scene_del_box(sid, bid):
        entry, err = _get_scene_or_404(sid)
        if err:
            return err
        n = len(entry["boxes"])
        entry["boxes"] = [b for b in entry["boxes"] if b["id"] != bid]
        if len(entry["boxes"]) == n:
            return jsonify({"error": "Kutu bulunamadı."}), 404
        return jsonify({"ok": True})

    @app.get("/api/scene/<sid>/cam")
    def api_scene_cam(sid):
        entry, err = _get_scene_or_404(sid)
        if err:
            return err
        try:
            bid = int(request.args.get("box", ""))
        except ValueError:
            return jsonify({"error": "Geçersiz kutu numarası."}), 400
        try:
            png = scene.cam_for_box(entry, bid)
        except KeyError:
            return jsonify({"error": "Kutu bulunamadı."}), 404
        except FileNotFoundError:
            return jsonify({"error": "Model dosyaları bulunamadı."}), 503
        return jsonify({"cam": inference.png_data_uri(png)})

    @app.get("/api/scene/<sid>/annotated.png")
    def api_scene_annotated(sid):
        entry, err = _get_scene_or_404(sid)
        if err:
            return err
        try:
            threshold = _parse_threshold(request.args)
        except inference.UploadError as e:
            return jsonify({"error": str(e)}), 400
        with_heat = request.args.get("heatmap", "1") != "0"
        png = scene.render_annotated(entry, threshold, with_heatmap=with_heat)
        return Response(png, mimetype="image/png", headers={
            "Content-Disposition":
                f"attachment; filename=sahne_{sid}_isaretli.png"})

    @app.get("/api/scene/<sid>/csv")
    def api_scene_csv(sid):
        entry, err = _get_scene_or_404(sid)
        if err:
            return err
        try:
            threshold = _parse_threshold(request.args)
        except inference.UploadError as e:
            return jsonify({"error": str(e)}), 400
        raw = scene.csv_bytes(entry, threshold)
        return Response(raw, mimetype="text/csv; charset=utf-8", headers={
            "Content-Disposition":
                f"attachment; filename=sahne_{sid}_kutular.csv"})

    # --- JSON hata yakalayicilari ---
    @app.errorhandler(413)
    def too_large(e):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": f"Yükleme çok büyük (limit {mb} MB)."}), 413

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Bulunamadı."}), 404
        return e

    @app.errorhandler(Exception)
    def server_error(e):
        if request.path.startswith("/api/"):
            traceback.print_exc()
            return jsonify({"error": f"Sunucu hatası: {type(e).__name__}"}), 500
        raise e

    return app


app = create_app()


if __name__ == "__main__":
    print("Deprem hasar tespiti arayuzu baslatiliyor...")
    print(f"  cihaz: {config.DEVICE.type} | fold dosyasi: {inference.fold_count()} "
          f"| esik: {DEPLOY_THRESHOLD:.2f}")
    try:
        inference.get_models()
        print("  ensemble yuklendi (5 fold, TTA acik).")
    except FileNotFoundError:
        print("  UYARI: outputs/cv_opt/fold*.pt bulunamadi; tahmin istekleri 503 dondurur.")
    det_ok, _ = detector.availability()
    print(f"  genis sahne: yolo {'hazir' if det_ok else 'devre disi'} | izgara hazir")
    print("  adres: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
