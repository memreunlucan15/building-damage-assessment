"""make_mosaic + evaluate_scene testleri (sentetik veri seti, model yok)."""
import csv

import pytest
from PIL import Image

from evaluate_scene import match_boxes
from make_mosaic import build_mosaic


def test_build_mosaic_counts_and_no_overlap(synth_dataset, tmp_path):
    out_png = tmp_path / "m.png"
    out_csv = tmp_path / "m_truth.csv"
    info = build_mosaic(synth_dataset, n_buildings=5, damaged_frac=0.4,
                        size=(600, 400), bg_kind="soil", seed=1,
                        out_png=out_png, out_csv=out_csv)
    assert info["n"] == 5 and info["damaged"] == 2 and info["intact"] == 3
    img = Image.open(out_png)
    assert img.size == (600, 400)

    with open(out_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert len(rows) == 5
    rects = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in rows]
    for x, y, w, h in rects:                       # sinir ici
        assert 0 <= x and 0 <= y and x + w <= 600 and y + h <= 400
    for i in range(len(rects)):                    # ikili kesisim alani 0
        for j in range(i + 1, len(rects)):
            ax, ay, aw, ah = rects[i]
            bx, by, bw, bh = rects[j]
            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            assert ix * iy == 0


def test_build_mosaic_too_small_raises(synth_dataset, tmp_path):
    with pytest.raises(ValueError, match="kucuk"):
        build_mosaic(synth_dataset, n_buildings=5, damaged_frac=0.4,
                     size=(70, 60), bg_kind="gray", seed=1,
                     out_png=tmp_path / "x.png", out_csv=tmp_path / "x.csv")


def test_match_boxes_greedy():
    pred = [{"x": 0, "y": 0, "w": 100, "h": 100},      # truth-0 ile tam eslesir
            {"x": 300, "y": 300, "w": 50, "h": 50},    # IoU dusuk: eslesmez
            {"x": 5, "y": 5, "w": 100, "h": 100}]      # truth-0'a ikinci aday
    truth = [{"x": 0, "y": 0, "w": 100, "h": 100},
             {"x": 500, "y": 500, "w": 40, "h": 40}]
    m = match_boxes(pred, truth, iou_thr=0.5)
    assert m == [(0, 0)]                               # en yuksek IoU kazanir, tek eslesme
