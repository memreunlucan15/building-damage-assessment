# Deprem Hasar Tespit — Uydu Görüntülerinden Bina Hasarı Sınıflandırma

![tests](https://github.com/memreunlucan15/derin-ogrenme-deprem-hasar-tespit/actions/workflows/tests.yml/badge.svg)

Deprem sonrası uydu görüntülerinden (optik RGB + SAR) bina bazında **ikili hasar
sınıflandırması**: `intact` (sağlam) / `damaged` (hasarlı). Mühendislik tasarımı
bitirme projesi.

Öncelik, hasarlı binayı **kaçırmamak** (recall): eşik seçimi F2 / precision-tabanlı
yapılır, sınıf dengesizliği dengeli örnekleme (WeightedRandomSampler) ve Focal Loss
ile ele alınır.

## Veri Seti

`earthquake_building_dataset/` altında her bina örneği için `.mat` dosyaları
(klasör adı = etiket):

| Sonek       | Anahtar | İçerik                              |
|-------------|---------|-------------------------------------|
| `_opt.mat`  | `x3`    | Optik RGB, (3,H,W) uint8            |
| `_optftp.mat` | `x4`  | Optik bina ayak izi maskesi {0,1}   |
| `_SAR.mat`  | `x1`    | SAR (dB), (H,W) float               |
| `_SARftp.mat` | `x2`  | SAR bina ayak izi maskesi {0,1}     |

~4029 örnek, ~169 `damaged` (≈%4) — ciddi sınıf dengesizliği. Veri seti ve
`outputs/` git'e dahil değildir.

## Kurulum

```bash
pip install -r requirements.txt
```

SAR kolu için SAR-HUB ön-eğitimli ağırlık (yalnızca `--modalities SAR` veya füzyon
denemeleri için gerekli; `config.SAR_PRETRAINED = "sarhub"`):

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1D8zA4unMK6ROvUKNMawbL3V3GbdnDK0Y" -O weights/sarhub
# Kullanılan dosya: weights/sarhub/ResNet18_TSX.pth (TerraSAR-X, X-band)
```

## Kullanım (güncel hat: çok-modlu CV)

Ana sonuç üretim aracı 5-fold çapraz doğrulamadır; modalite seçilir, çıktılar
`outputs/cv_<tag>/` altına yazılır:

```bash
python dataset.py                 # veri indeksi smoke testi
python cross_validate_mm.py --modalities opt     --tag opt     --folds 5 --epochs 25
python cross_validate_mm.py --modalities SAR     --tag SAR     --folds 5 --epochs 25
python cross_validate_mm.py --modalities opt,SAR --tag opt_SAR --folds 5 --epochs 25
```

Dağıtım (yeni görüntü için tahmin): 5 optik fold modelinin ortalaması —

```bash
python ensemble.py                # demo; API için dosya başındaki docstring'e bak
python predict.py klasor/ --csv tahminler.csv   # CLI: *_opt.mat dosyalarına tahmin
```

Testler (veri seti ve checkpoint gerektirmez, ~10 sn):

```bash
pip install pytest
python -m pytest tests/ -q
```

### Eski hat (Faz 1, tek sabit split)

`split.py → train.py → evaluate.py` tek bir train/val/test bölmesiyle çalışan ilk
sürümdür; Grad-CAM görselleri `evaluate.py`'den gelir. Raporlanan nihai sonuçlar
CV hattından alınmıştır.

## Sonuçlar (5-fold CV, ortalama)

| Deneme            | Recall | Precision | F1    | ROC-AUC | PR-AUC |
|-------------------|--------|-----------|-------|---------|--------|
| **opt (ResNet18)**| 0.686  | 0.565     | **0.616** | 0.946 | 0.673  |
| SAR (SAR-HUB)     | 0.321  | 0.286     | 0.202 | 0.802   | 0.240  |
| opt+SAR           | 0.681  | 0.549     | 0.603 | 0.944   | 0.667  |
| opt+SAR (SAR-HUB) | 0.729  | 0.499     | 0.590 | 0.944   | 0.687  |
| opt + footprint   | 0.639  | 0.568     | 0.599 | 0.944   | 0.685  |
| opt (ResNet34)    | 0.663  | 0.573     | 0.608 | 0.947   | 0.682  |

Özet: tek başına optik en iyi F1'i veriyor; SAR füzyonu recall'u artırıyor ama
precision'dan yiyor. ResNet34, footprint ve füzyon denemeleri anlamlı kazanım
sağlamadı → dağıtım modeli **opt / ResNet18 5-fold ensemble**.

### Görseller (opt, OOF)

| PR / ROC | Confusion Matrix |
|----------|------------------|
| ![PR ve ROC eğrileri](assets/pr_roc.png) | ![Confusion matrix](assets/confusion_matrix.png) |

![Grad-CAM örnekleri](assets/gradcam.png)

Grad-CAM: model kararlarının bina ve enkaz bölgelerine odaklandığının nitel kontrolü.

## Dosya Haritası

| Dosya | Görev |
|-------|-------|
| `config.py` | Tüm yollar/hiperparametreler/seed (tek gerçek kaynağı) |
| `dataset.py`, `dataset_mm.py` | .mat okuyucular, indeks, Dataset sınıfları |
| `model.py`, `model_mm.py` | ResNet18/34 ve çok-kollu füzyon modeli (SAR-HUB yükleme) |
| `losses.py` | Focal Loss |
| `engine.py` | **Ortak motor**: run_epoch, iki aşamalı fit, TTA, eşik seçimi |
| `engine_mm.py` | Ortak motorun çok-modlu bağlaması |
| `split.py`, `train.py`, `evaluate.py`, `cross_validate.py` | Faz-1 (tek modalite) hattı |
| `cross_validate_mm.py` | Güncel 5-fold CV aracı (opt/SAR/füzyon) |
| `ensemble.py` | Dağıtım: 5 fold modelinin olasılık ortalaması |
| `predict.py` | CLI: dosya/klasörden ensemble tahmini (+CSV çıktı) |
| `tests/` | pytest paketi (sentetik veriyle, eğitimsiz) |
| `build_report.py`, `build_slides.py`, `render_pptx.py` | Rapor/sunum üretim yardımcıları (tek seferlik) |

## Notlar

- `NUM_WORKERS = 0`: Windows + h5py kombinasyonunda çoklu işçi sorun çıkarabilir.
- Eşik (`pick_threshold`): önce `precision ≥ 0.50` sağlayan eşikler içinde recall
  maksimize edilir; sağlanamazsa F2-optimal eşiğe düşülür. Eşik daima iç-validation
  üzerinde seçilir, fold testine öyle uygulanır (sızıntı yok).
- Tekrarlanabilirlik: `config.set_seed()` — cudnn determinizmi zorlanmaz, küçük
  farklar normaldir.
