"""Merkezi yapilandirma: yollar, hiperparametreler, cihaz ve tekrarlanabilirlik."""
import os
import random
from pathlib import Path

import numpy as np
import torch

# --- Yollar ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "earthquake_building_dataset"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Siniflar ---
# Pozitif sinif = damaged (label 1). Hasarli binayi kacirmamak oncelik.
CLASSES = ["intact", "damaged"]
CLASS_TO_IDX = {"intact": 0, "damaged": 1}
POSITIVE_CLASS = "damaged"
POSITIVE_IDX = 1

# --- Modalite (Faz 1: optik RGB) ---
MODALITY = "opt"          # opt -> anahtar x3, (3,H,W) uint8 RGB
MAT_KEY = {"opt": "x3", "optftp": "x4", "SAR": "x1", "SARftp": "x2"}[MODALITY]

# --- Goruntu / egitim hiperparametreleri ---
IMG_SIZE = 224            # ImageNet on-egitimli ResNet icin standart
BATCH_SIZE = 32
NUM_WORKERS = 0           # Windows + h5py icin guvenli baslangic
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7   # val damaged-F1 iyilesmezse durdur

# --- Loss secimi ---
LOSS = "focal"            # "focal" veya "ce" (sinif-agirlikli CrossEntropy)
FOCAL_GAMMA = 2.0         # focal odak parametresi

# --- Esik secimi (recall odakli) ---
# F-beta ile esik secilir; beta>1 recall'a precision'dan daha cok agirlik verir.
# beta=2 -> F2 skoru: hasarli binayi kacirmamak oncelikli.
THRESHOLD_BETA = 2.0
# Saglam operasyon noktasi: precision >= PRECISION_FLOOR iken recall'i maksimize et.
# Hicbir esik bu tabani saglamazsa F-beta'ya geri don.
PRECISION_FLOOR = 0.50

# --- Backbone ---
BACKBONE = "resnet18"     # "resnet18" veya "resnet34" (denendi: resnet34 kazanim yok)

# --- Cok-modlu (Tier 2) ---
# Kullanilacak modaliteler: ["opt"], ["SAR"] veya ["opt","SAR"] (fuzyon).
# cross_validate_mm.py icinde --modalities ile gecersiz kilinabilir.
MODALITIES = ["opt"]
USE_FOOTPRINT = False      # True ise footprint maskesi ek kanal olarak eklenir (denendi: kazanim yok)
# Her modalitenin temel kanal sayisi (footprint hariç). SAR 3 kanala cogaltilir
# (z-score sonrasi) -> 3-kanalli on-egitimli conv1 (ImageNet veya SAR-HUB) dogrudan kullanilir.
MODALITY_CHANNELS = {"opt": 3, "SAR": 3}
MAT_KEYS = {"opt": "x3", "optftp": "x4", "SAR": "x1", "SARftp": "x2"}
FOOTPRINT_SUFFIX = {"opt": "optftp", "SAR": "SARftp"}

# SAR kolu on-egitimi: "imagenet" (varsayilan torchvision) veya "sarhub" (SAR-domain).
# SAR-HUB sadece resnet18 icin mevcut; ResNet18_TSX = TerraSAR-X (X-band, Capella'ya uygun).
SAR_PRETRAINED = "sarhub"
SARHUB_WEIGHTS = ROOT / "weights" / "sarhub" / "ResNet18_TSX.pth"
# SAR normalizasyonu: True -> SAR-HUB'a uygun dB min-max [SAR_DB_MIN,SAR_DB_MAX]->[0,1]
# (per-image z-score yerine). Sinirlar veri seti dB persentillerinden (~1-99) secildi.
SAR_DB_NORM = True
SAR_DB_MIN = 24.0
SAR_DB_MAX = 38.0

# --- Tier 1 egitim recetesi ---
USE_SAMPLER = True        # WeightedRandomSampler ile dengeli batch'ler
FREEZE_EPOCHS = 3         # once sadece siniflandirici basligi egit (govde dondurulur)
LR_HEAD = 1e-3            # dondurulmus asama icin baslik LR'si
USE_TTA = True            # cikarimda flip tabanli test-time augmentation

# --- Capraz dogrulama (CV) ---
N_FOLDS = 5
CV_EPOCHS = 25
INNER_VAL_FRACTION = 0.15  # her fold'un egitim kismindan ayrilan ic-validation
CV_DIR = OUTPUT_DIR / "cv"
CV_DIR.mkdir(exist_ok=True)

# --- Veri bolme oranlari ---
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# --- ImageNet normalizasyonu ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Cihaz ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Cikti dosyalari ---
SPLIT_PATH = OUTPUT_DIR / "split.json"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "history.json"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
