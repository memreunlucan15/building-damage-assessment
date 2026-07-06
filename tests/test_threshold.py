"""pick_threshold: precision tabanli saglam esik secimi."""
import numpy as np

from engine import pick_threshold


def test_feasible_maximizes_recall_with_precision_floor():
    # 4 pozitif p=0.9, 4 negatif p=0.1 -> t=0.5 civari mukemmel ayrim.
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    t, feasible = pick_threshold(probs, labels, precision_floor=0.5)
    assert feasible
    preds = (probs >= t).astype(int)
    assert (preds == labels).all()          # secilen esik tam ayrimi korur


def test_feasible_prefers_recall_over_precision():
    # Dusuk esik: recall 1.0 / precision 0.5 (taban tam sinirda).
    # Yuksek esik: recall 0.5 / precision 1.0. Recall onceligi dusugu secmeli.
    probs = np.array([0.9, 0.6, 0.55, 0.58, 0.1, 0.1])
    labels = np.array([1, 1, 0, 0, 0, 0])
    t, feasible = pick_threshold(probs, labels, precision_floor=0.5)
    assert feasible
    preds = (probs >= t).astype(int)
    recall = preds[labels == 1].mean()
    assert recall == 1.0


def test_infeasible_falls_back_to_fbeta():
    # Pozitifler negatiflerin altinda -> hicbir esik precision tabanini saglamaz.
    probs = np.array([0.2, 0.25, 0.8, 0.85, 0.9])
    labels = np.array([1, 1, 0, 0, 0])
    t, feasible = pick_threshold(probs, labels, precision_floor=0.99, beta=2.0)
    assert not feasible
    assert 0.05 <= t <= 0.95


def test_threshold_within_grid_bounds():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=200)
    labels = (probs + rng.normal(0, 0.3, 200) > 0.5).astype(int)
    t, _ = pick_threshold(probs, labels)
    assert 0.05 <= t <= 0.95
