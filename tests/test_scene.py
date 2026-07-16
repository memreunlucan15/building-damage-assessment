"""scene.py testleri: geometri, iki asama, isi haritasi, depo, analiz (mock'lu)."""
import io

import numpy as np
import pytest
from PIL import Image

import detector
import inference
import scene


def _pil(w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), "RGB")


# ---------------- geometri ----------------
def test_tile_grid_covers_and_overlaps():
    w, h = 1500, 900
    tiles = scene.tile_grid(w, h)
    mask = np.zeros((h, w), dtype=bool)
    xs = sorted({t[0] for t in tiles})
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert x2 - x1 <= scene.TILE and y2 - y1 <= scene.TILE
        mask[y1:y2, x1:x2] = True
    assert mask.all()                              # tam kapsama
    assert xs[1] - xs[0] == 512                    # stride = 640*(1-0.2)
    assert any(t[2] == w for t in tiles)           # son tile kenara yaslanir


def test_tile_grid_small_image():
    assert scene.tile_grid(400, 300) == [(0, 0, 400, 300)]


def test_grid_windows_cover_stride_and_shape():
    wins, nx, ny = scene.grid_windows(1000, 500, 96)
    assert len(wins) == nx * ny
    xs = sorted({w[0] for w in wins})
    assert xs[1] - xs[0] == 48                     # stride = win//2
    assert wins[1][0] > wins[0][0] and wins[1][1] == wins[0][1]   # satir-major
    mask = np.zeros((500, 1000), dtype=bool)
    for x1, y1, x2, y2 in wins:
        assert x2 - x1 <= 96 and y2 - y1 <= 96
        mask[y1:y2, x1:x2] = True
    assert mask.all()


def test_grid_windows_smaller_scene():
    wins, nx, ny = scene.grid_windows(80, 60, 96)
    assert (nx, ny) == (1, 1)
    assert wins == [(0, 0, 80, 60)]


def test_check_grid_rejects():
    with pytest.raises(inference.UploadError, match="Pencere boyutu"):
        scene.check_grid(500, 500, 32)
    with pytest.raises(inference.UploadError, match="yoğun"):
        scene.check_grid(8000, 8000, 64)           # 249*249 pencere > 9000


def test_chip_rect_pads_and_squares():
    x1, y1, x2, y2 = scene.chip_rect((100, 100, 50, 30), 1000, 1000)
    assert (x2 - x1) == (y2 - y1) == 70            # max(50,30)*1.4
    assert x1 <= 100 and x2 >= 150 and y1 <= 100 and y2 >= 130


def test_chip_rect_translates_at_corner():
    x1, y1, x2, y2 = scene.chip_rect((2, 2, 50, 30), 1000, 1000)
    assert x1 >= 0 and y1 >= 0
    assert (x2 - x1) == (y2 - y1) == 70            # kare korunur, ceviri ile


def test_chip_rect_huge_box_clamps():
    x1, y1, x2, y2 = scene.chip_rect((0, 0, 500, 400), 300, 200)
    assert (x1, y1, x2, y2) == (0, 0, 300, 200)    # tum sahne


def test_nms_keep_merges():
    boxes = np.array([[0, 0, 100, 100], [20, 20, 100, 100], [300, 300, 50, 50]],
                     dtype=np.float32)
    scores = np.array([0.9, 0.5, 0.8], dtype=np.float32)
    keep = scene.nms_keep(boxes, scores, iou_thr=0.45)
    assert set(keep.tolist()) == {0, 2}
    assert scene.nms_keep(np.zeros((0, 4)), np.zeros(0), 0.5).shape == (0,)


# ---------------- iki asama ----------------
def test_two_stage_refines_only_candidates(monkeypatch):
    rects = [(i * 10, 0, i * 10 + 10, 10) for i in range(10)]
    fast_p = np.array([0.01, 0.5, 0.02, 0.02, 0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
                      dtype=np.float32)
    seen_full = []
    monkeypatch.setattr(inference, "predict_pils_fast",
                        lambda pils, batch_size=256: fast_p[:len(pils)])
    def fake_full(pils):
        seen_full.append(len(pils))
        return np.full(len(pils), 0.9, dtype=np.float32)
    monkeypatch.setattr(inference, "predict_pils", fake_full)

    p, stats = scene.two_stage_probs(_pil(200, 20), rects)
    assert stats["n_windows"] == 10 and stats["n_refined"] == 2
    assert sum(seen_full) == 2                     # yalnizca 2 aday rafine edildi
    assert p[1] == pytest.approx(0.9) and p[4] == pytest.approx(0.9)
    assert p[0] == pytest.approx(0.01)             # aday degil: fold-1 degeri kalir


def test_two_stage_chunks_calls(monkeypatch):
    rects = [(0, 0, 10, 10)] * 300
    calls = {"fast": 0}
    def fake_fast(pils, batch_size=256):
        calls["fast"] += 1
        assert all(p.size == (10, 10) for p in pils)   # crop boyutu dogru
        return np.zeros(len(pils), dtype=np.float32)
    monkeypatch.setattr(inference, "predict_pils_fast", fake_fast)
    monkeypatch.setattr(inference, "predict_pils",
                        lambda pils: np.zeros(len(pils), dtype=np.float32))
    scene.two_stage_probs(_pil(64, 64), rects, fast_batch=128)
    assert calls["fast"] == 3                      # 300/128 -> 3 parca


# ---------------- isi haritasi ----------------
def test_heatmap_png_alpha_ramp():
    grid = np.array([[0.0, 1.0], [0.5, scene.CANDIDATE_THR]], dtype=np.float32)
    png = scene.heatmap_png(grid, 64, 64)
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA" and img.size == (64, 64)
    a = np.asarray(img)[:, :, 3]
    assert a[2, 61] > 150                          # p=1 kosesi yogun
    assert a[2, 2] == 0                            # p=0 kosesi seffaf


# ---------------- depo ----------------
def test_scene_store_lru_evicts():
    store = scene.SceneStore(max_scenes=2)
    s1 = store.put({"n": 1})
    s2 = store.put({"n": 2})
    s3 = store.put({"n": 3})
    assert store.get(s1) is None                   # en eski atildi
    assert store.get(s2)["n"] == 2                 # erisim tazeler
    s4 = store.put({"n": 4})
    assert store.get(s3) is None and store.get(s2) is not None
    assert store.get(s4) is not None


# ---------------- isaretli cikti ----------------
def _entry(boxes, w=300, h=200):
    return {"pil": _pil(w, h), "width": w, "height": h, "mode": "grid",
            "window": 96, "display_scale": 1.0, "boxes": boxes,
            "heatmap_png": None, "next_box_id": len(boxes) + 1,
            "threshold": 0.4, "created": 0.0}


def test_render_annotated_smoke():
    boxes = [{"id": 1, "x": 20, "y": 20, "w": 60, "h": 60, "p": 0.9,
              "source": "grid", "det_conf": None},
             {"id": 2, "x": 150, "y": 100, "w": 60, "h": 60, "p": 0.05,
              "source": "grid", "det_conf": None}]
    png = scene.render_annotated(_entry(boxes), threshold=0.4)
    assert png[:4] == b"\x89PNG"
    img = np.asarray(Image.open(io.BytesIO(png)))
    assert img.shape[:2] == (200, 300)             # tam cozunurluk
    # kutu-1 kenarinda kirmizi cerceve var; kutu-2 (esik alti grid) cizilmemis
    assert (img[20, 20:80, 0] > 180).any()
    orig = np.asarray(_pil(300, 200))
    assert np.array_equal(img[100, 150:210], orig[100, 150:210])


def test_csv_bytes_bom_and_rows():
    import codecs
    boxes = [{"id": 1, "x": 1, "y": 2, "w": 30, "h": 40, "p": 0.87,
              "source": "yolo", "det_conf": 0.5},
             {"id": 2, "x": 5, "y": 6, "w": 30, "h": 40, "p": 0.1,
              "source": "manual", "det_conf": None}]
    raw = scene.csv_bytes(_entry(boxes), threshold=0.4)
    assert raw.startswith(codecs.BOM_UTF8)
    text = raw[len(codecs.BOM_UTF8):].decode("utf-8")
    lines = [l for l in text.splitlines() if l]
    assert lines[0].startswith("id;kaynak;x;y")
    assert len(lines) == 3
    assert "HASARLI" in lines[1] and "SAGLAM" in lines[2]
    assert lines[2].split(";")[1] == "manuel"


# ---------------- analyze_scene ----------------
def test_analyze_scene_grid_mocked_end_to_end(monkeypatch):
    monkeypatch.setattr(inference, "predict_pils_fast",
                        lambda pils, batch_size=256: np.full(len(pils), 0.6, np.float32))
    monkeypatch.setattr(inference, "predict_pils",
                        lambda pils: np.full(len(pils), 0.8, np.float32))
    d = scene.analyze_scene(_pil(320, 240), mode="grid", threshold=0.4, window=96)
    assert set(d) >= {"scene_id", "width", "height", "mode", "display",
                      "detector_available", "buildings", "heatmap", "timings"}
    assert d["mode"] == "grid" and d["window"] == 96
    assert d["heatmap"].startswith("data:image/png;base64,")
    assert d["display"]["scale"] == 1.0            # kucuk sahne kucultulmez
    assert all(b["source"] == "grid" for b in d["buildings"])
    assert all(0 <= b["p"] <= 1 for b in d["buildings"])
    assert scene.STORE.get(d["scene_id"]) is not None
    assert d["timings"]["n_windows"] > 0


def test_analyze_scene_yolo_offsets_and_chips(monkeypatch):
    # her tile icin sabit tek kutu (tile koordinatinda 10,10,60,60)
    def fake_tiles(pils, conf=0.25):
        return [np.array([[10, 10, 60, 60, 0.7]], dtype=np.float32) for _ in pils]
    monkeypatch.setattr(detector, "predict_tiles", fake_tiles)
    monkeypatch.setattr(detector, "availability", lambda: (True, None))
    monkeypatch.setattr(inference, "predict_pils_fast",
                        lambda pils, batch_size=256: np.full(len(pils), 0.5, np.float32))
    monkeypatch.setattr(inference, "predict_pils",
                        lambda pils: np.full(len(pils), 0.9, np.float32))
    d = scene.analyze_scene(_pil(700, 300), mode="yolo", threshold=0.4, window=96)
    assert d["mode"] == "yolo" and d["heatmap"] is None
    tiles = scene.tile_grid(700, 300)
    expected_xs = sorted({t[0] + 10 for t in tiles})   # tile ofseti + kutu x'i
    assert len(d["buildings"]) == len(expected_xs)     # NMS sonrasi tile basina 1 kutu
    assert sorted(b["x"] for b in d["buildings"]) == expected_xs
    assert all(b["source"] == "yolo" and b["det_conf"] == 0.7 for b in d["buildings"])


def test_analyze_scene_yolo_unavailable_propagates(monkeypatch):
    def boom(pils, conf=0.25):
        raise detector.DetectorUnavailable("ultralytics kurulu değil")
    monkeypatch.setattr(detector, "predict_tiles", boom)
    with pytest.raises(detector.DetectorUnavailable):
        scene.analyze_scene(_pil(), mode="yolo", threshold=0.4, window=96)


def test_analyze_scene_rejects_bad_mode_and_size(monkeypatch):
    with pytest.raises(inference.UploadError, match="Geçersiz mod"):
        scene.analyze_scene(_pil(), mode="foo", threshold=0.4, window=96)
    monkeypatch.setattr(scene, "MAX_SIDE", 100)
    with pytest.raises(inference.UploadError, match="büyük"):
        scene.analyze_scene(_pil(320, 240), mode="grid", threshold=0.4, window=96)
