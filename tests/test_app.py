"""app.py rota testleri: inference siniri monkeypatch'lenir, gercek agirlik yuklenmez."""
import io

import numpy as np
import pytest
from PIL import Image

import config
import inference
from conftest import write_opt_mat, write_sar_mat


def _png_file(w=32, h=32, color=(120, 60, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def client(monkeypatch):
    def fake_predict(pils):
        return np.array(([0.9, 0.1] * len(pils))[:len(pils)])

    monkeypatch.setattr(inference, "predict_pils", fake_predict)
    monkeypatch.setattr(inference, "gradcam_pils",
                        lambda pils: [np.zeros((224, 224), np.float32)] * len(pils))
    import app as app_module
    a = app_module.create_app()
    a.testing = True
    return a.test_client()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Deprem Hasar Tespiti" in r.get_data(as_text=True)


def test_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    d = r.get_json()
    for key in ("threshold", "device", "n_fold_files", "model_ready",
                "max_files", "samples_available", "tta"):
        assert key in d
    assert 0.0 < d["threshold"] < 1.0     # lokalde 0.40, CI'da 0.5 fallback


def test_predict_png(client):
    r = client.post("/api/predict", data={
        "files": (_png_file(), "bina.png"),
        "threshold": "0.5", "gradcam": "1",
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["threshold"] == 0.5
    res = d["results"][0]
    assert res["ok"] is True
    assert res["p_damaged"] == pytest.approx(0.9)
    assert res["prediction"] == "damaged"
    assert res["source"] == "upload"
    assert res["thumb"].startswith("data:image/jpeg")
    assert res["cam"].startswith("data:image/png")


def test_predict_mat_and_gradcam_off(client, tmp_path):
    path = tmp_path / "a_opt.mat"
    write_opt_mat(path)
    r = client.post("/api/predict", data={
        "files": (io.BytesIO(path.read_bytes()), "a_opt.mat"),
        "gradcam": "0",
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    res = r.get_json()["results"][0]
    assert res["ok"] is True
    assert res["cam"] is None


def test_predict_batch_partial_failure(client, tmp_path):
    sar = tmp_path / "a_SAR.mat"
    write_sar_mat(sar)
    r = client.post("/api/predict", data={
        "files": [(_png_file(), "iyi.png"),
                  (io.BytesIO(sar.read_bytes()), "a_SAR.mat")],
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    res = r.get_json()["results"]
    assert len(res) == 2
    assert res[0]["name"] == "iyi.png" and res[0]["ok"] is True
    assert res[1]["ok"] is False and "x3" in res[1]["error"]


def test_predict_empty(client):
    r = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_too_many(client):
    files = [(_png_file(), f"f{i}.png") for i in range(17)]
    r = client.post("/api/predict", data={"files": files},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_bad_threshold(client):
    r = client.post("/api/predict", data={
        "files": (_png_file(), "a.png"), "threshold": "2",
    }, content_type="multipart/form-data")
    assert r.status_code == 400


def test_models_missing_503(client, monkeypatch):
    def boom(pils):
        raise FileNotFoundError("fold yok")
    monkeypatch.setattr(inference, "predict_pils", boom)
    r = client.post("/api/predict", data={"files": (_png_file(), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 503
    assert "Model dosyaları" in r.get_json()["error"]


def test_samples_missing_dataset(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "yok")
    r = client.get("/api/samples")
    assert r.status_code == 200
    assert r.get_json() == {"available": False, "samples": []}


def test_samples_and_sample_predict(client, monkeypatch, synth_dataset):
    monkeypatch.setattr(config, "DATA_DIR", synth_dataset)
    r = client.get("/api/samples")
    d = r.get_json()
    assert d["available"] is True
    assert 0 < len(d["samples"]) <= 8
    ref = d["samples"][0]["ref"]
    assert ref.split("/")[0] in ("damaged", "intact")
    assert d["samples"][0]["thumb"].startswith("data:image/jpeg")

    r = client.post("/api/predict", data={"sample_refs": ref},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    res = r.get_json()["results"][0]
    assert res["ok"] is True
    assert res["source"] == "sample"
    assert res["true_cls"] == ref.split("/")[0]


def test_sample_traversal_rejected(client):
    r = client.post("/api/predict", data={"sample_refs": "damaged/../../gizli"},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    res = r.get_json()["results"][0]
    assert res["ok"] is False
    assert "Geçersiz örnek referansı" in res["error"]


def test_413_json(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 1000
    big = io.BytesIO(b"0" * 5000)
    r = client.post("/api/predict", data={"files": (big, "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 413
    assert "error" in r.get_json()
