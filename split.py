"""Stratified train/val/test bolmesi; tekrarlanabilirlik icin JSON'a kaydedilir."""
import json

from sklearn.model_selection import train_test_split

import config
from dataset import build_index


def make_split(save: bool = True):
    samples = build_index()
    labels = [s["label"] for s in samples]

    # Once test'i ayir, sonra kalandan val ayir (her ikisi de stratified).
    train_val, test = train_test_split(
        samples, test_size=config.TEST_FRACTION,
        stratify=labels, random_state=config.SEED,
    )
    tv_labels = [s["label"] for s in train_val]
    val_rel = config.VAL_FRACTION / (1.0 - config.TEST_FRACTION)
    train, val = train_test_split(
        train_val, test_size=val_rel,
        stratify=tv_labels, random_state=config.SEED,
    )

    split = {
        "train": [s["id"] for s in train],
        "val": [s["id"] for s in val],
        "test": [s["id"] for s in test],
    }
    if save:
        with open(config.SPLIT_PATH, "w") as f:
            json.dump(split, f, indent=2)
    return split


def load_split():
    with open(config.SPLIT_PATH) as f:
        split = json.load(f)
    id_sets = {k: set(v) for k, v in split.items()}
    samples = build_index()
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        for part in ("train", "val", "test"):
            if s["id"] in id_sets[part]:
                out[part].append(s)
                break
    return out


def _summary(name, items):
    n_dmg = sum(1 for s in items if s["label"] == 1)
    print(f"{name:5s}: {len(items):5d} ornek | damaged {n_dmg:4d} | intact {len(items)-n_dmg:5d}")


if __name__ == "__main__":
    make_split(save=True)
    parts = load_split()
    for name in ("train", "val", "test"):
        _summary(name, parts[name])
    print(f"\nKaydedildi: {config.SPLIT_PATH}")
