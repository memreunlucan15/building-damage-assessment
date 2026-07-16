"""Genis sahne API testleri: inference/detector siniri mock'lanir, agirlik yuklenmez."""
import codecs
import io

import numpy as np
import pytest
from PIL import Image

import detector
import inference
import scene


def _scene_png(w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(inference, "predict_pils_fast",
                        lambda pils, batch_size=256: np.full(len(pils), 0.6, np.float32))
    monkeypatch.setattr(inference, "predict_pils",
                        lambda pils: np.full(len(pils), 0.9, np.float32))
    monkeypatch.setattr(inference, "gradcam_pils",
                        lambda pils: [np.zeros((224, 224), np.float32)] * len(pils))
    import app as app_module
    a = app_module.create_app()
    a.testing = True
    return a.test_client()


def _post_scene(client, mode="grid", window="96", **extra):
    data = {"file": (_scene_png(), "sahne.png"), "mode": mode, "window": window}
    data.update(extra)
    return client.post("/api/scene", data=data, content_type="multipart/form-data")


def test_scene_grid_schema(client):
    r = _post_scene(client)
    assert r.status_code == 200
    d = r.get_json()
    for key in ("scene_id", "width", "height", "mode", "threshold", "window",
                "display", "detector_available", "buildings", "heatmap", "timings"):
        assert key in d
    assert d["mode"] == "grid" and d["window"] == 96
    assert 0 < d["display"]["scale"] <= 1.0
    assert d["display"]["uri"].startswith("data:image/jpeg;base64,")
    assert d["heatmap"].startswith("data:image/png;base64,")
    assert all(isinstance(b["p"], float) for b in d["buildings"])
    assert set(d["timings"]) >= {"detect_s", "classify_s", "total_s",
                                 "n_windows", "n_refined"}


def test_scene_yolo_unavailable_400(client, monkeypatch):
    def boom(pils, conf=0.25):
        raise detector.DetectorUnavailable(
            "YOLO bina tespiti için 'ultralytics' paketi kurulu değil")
    monkeypatch.setattr(detector, "predict_tiles", boom)
    r = _post_scene(client, mode="yolo")
    assert r.status_code == 400
    d = r.get_json()
    assert d["detector_available"] is False
    assert "ultralytics" in d["error"]


def test_scene_missing_file_400(client):
    r = client.post("/api/scene", data={"mode": "grid"},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_scene_bad_mode_and_window_400(client):
    assert _post_scene(client, mode="foo").status_code == 400
    assert _post_scene(client, window="31").status_code == 400
    assert _post_scene(client, window="abc").status_code == 400


def test_scene_too_large_400(client, monkeypatch):
    monkeypatch.setattr(scene, "MAX_SIDE", 100)
    r = _post_scene(client)
    assert r.status_code == 400
    assert "büyük" in r.get_json()["error"]


def test_manual_box_roundtrip_and_csv(client):
    sid = _post_scene(client).get_json()["scene_id"]
    r = client.post(f"/api/scene/{sid}/box",
                    json={"x": 10, "y": 10, "w": 40, "h": 40})
    assert r.status_code == 200
    box = r.get_json()["box"]
    assert box["source"] == "manual" and box["p"] == pytest.approx(0.9)

    csv_r = client.get(f"/api/scene/{sid}/csv?threshold=0.4")
    text = csv_r.data[len(codecs.BOM_UTF8):].decode("utf-8")
    assert f"{box['id']};manuel;10;10;40;40" in text


def test_manual_box_invalid_and_unknown_scene(client):
    sid = _post_scene(client).get_json()["scene_id"]
    r = client.post(f"/api/scene/{sid}/box",
                    json={"x": -5, "y": 10, "w": 40, "h": 40})
    assert r.status_code == 400
    r = client.post(f"/api/scene/{sid}/box", json={"x": 1, "y": 1, "w": 4, "h": 4})
    assert r.status_code == 400                      # cok kucuk
    r = client.post("/api/scene/yokyok/box", json={"x": 1, "y": 1, "w": 40, "h": 40})
    assert r.status_code == 404


def test_delete_box(client):
    d = _post_scene(client).get_json()
    sid = d["scene_id"]
    bid = client.post(f"/api/scene/{sid}/box",
                      json={"x": 10, "y": 10, "w": 40, "h": 40}).get_json()["box"]["id"]
    assert client.delete(f"/api/scene/{sid}/box/{bid}").status_code == 200
    assert client.delete(f"/api/scene/{sid}/box/{bid}").status_code == 404


def test_cam_endpoint(client):
    d = _post_scene(client).get_json()
    sid = d["scene_id"]
    bid = client.post(f"/api/scene/{sid}/box",
                      json={"x": 10, "y": 10, "w": 60, "h": 60}).get_json()["box"]["id"]
    r = client.get(f"/api/scene/{sid}/cam?box={bid}")
    assert r.status_code == 200
    assert r.get_json()["cam"].startswith("data:image/png;base64,")
    assert client.get(f"/api/scene/{sid}/cam?box=99999").status_code == 404
    assert client.get(f"/api/scene/{sid}/cam?box=abc").status_code == 400


def test_annotated_png_download(client):
    sid = _post_scene(client).get_json()["scene_id"]
    r = client.get(f"/api/scene/{sid}/annotated.png?threshold=0.4")
    assert r.status_code == 200
    assert r.mimetype == "image/png"
    assert r.data[:4] == b"\x89PNG"
    assert "attachment" in r.headers["Content-Disposition"]


def test_csv_download(client):
    d = _post_scene(client).get_json()
    sid = d["scene_id"]
    r = client.get(f"/api/scene/{sid}/csv?threshold=0.4")
    assert r.status_code == 200
    assert r.data.startswith(codecs.BOM_UTF8)
    text = r.data[len(codecs.BOM_UTF8):].decode("utf-8")
    lines = [l for l in text.splitlines() if l]
    assert lines[0].startswith("id;kaynak;x;y;genislik;yukseklik;p_hasarli")
    assert len(lines) == 1 + len(d["buildings"])


def test_store_eviction_404(client):
    sids = [_post_scene(client).get_json()["scene_id"] for _ in range(3)]
    r = client.get(f"/api/scene/{sids[0]}/csv")
    assert r.status_code == 404
    assert "yeniden analiz" in r.get_json()["error"]


def test_config_scene_block(client):
    d = client.get("/api/config").get_json()
    s = d["scene"]
    assert isinstance(s["detector_importable"], bool)
    assert s["window_min"] == 64 and s["window_max"] == 192
    assert s["window_default"] == 96
    assert s["max_side"] == 8000
