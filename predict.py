"""Komut satirindan hasar tahmini: .mat dosyalari veya klasorler -> p(damaged).

5 optik fold modelinin ensemble ortalamasini kullanir (bkz. ensemble.py).
Girdi olarak *_opt.mat dosyalari beklenir; kardes modalite dosyalari ayni
klasorde aranir (su an yalniz optik kullaniliyor).

Kullanim:
    python predict.py ornek_klasoru/
    python predict.py bina1_opt.mat bina2_opt.mat --csv tahminler.csv
    python predict.py klasor/ --threshold 0.6
"""
import argparse
import csv
from pathlib import Path

import config
from ensemble import DEPLOY_THRESHOLD, load_ensemble, predict_labels


def collect_samples(paths):
    """Verilen dosya/klasor yollarindan tahmin ornekleri kurar."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob("*_opt.mat"))
        elif p.suffix == ".mat" and p.stem.endswith("_opt"):
            files.append(p)
        else:
            raise SystemExit(f"Desteklenmeyen girdi: {p} (beklenen: *_opt.mat veya klasor)")
    if not files:
        raise SystemExit("Hic *_opt.mat dosyasi bulunamadi.")
    # label alani Dataset arayuzu icin zorunlu; tahminde kullanilmaz.
    return [{"id": f.name.rsplit("_opt.mat", 1)[0], "label": 0,
             "cls": "?", "path": str(f)} for f in files]


def main():
    ap = argparse.ArgumentParser(description="Ensemble ile bina hasar tahmini")
    ap.add_argument("paths", nargs="+", help="*_opt.mat dosyalari ve/veya klasorler")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"karar esigi (varsayilan: OOF esigi {DEPLOY_THRESHOLD:.2f})")
    ap.add_argument("--csv", type=str, default=None, help="sonuclari CSV'ye yaz")
    args = ap.parse_args()

    samples = collect_samples(args.paths)
    threshold = DEPLOY_THRESHOLD if args.threshold is None else args.threshold
    print(f"Ornek: {len(samples)} | esik: {threshold:.2f} | cihaz: {config.DEVICE}")

    models = load_ensemble()
    preds, probs, _ = predict_labels(models, samples, threshold)

    name = {0: "intact", 1: "damaged"}
    print(f"\n  {'id':<20s} {'tahmin':<9s} p(damaged)")
    for s, pr, pb in zip(samples, preds, probs):
        print(f"  {s['id']:<20s} {name[int(pr)]:<9s} {pb:.3f}")
    n_dmg = int(preds.sum())
    print(f"\n  Toplam {len(samples)} bina: {n_dmg} damaged, {len(samples) - n_dmg} intact")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "path", "p_damaged", "prediction", "threshold"])
            for s, pr, pb in zip(samples, preds, probs):
                w.writerow([s["id"], s["path"], f"{pb:.4f}", name[int(pr)], f"{threshold:.2f}"])
        print(f"  CSV: {args.csv}")


if __name__ == "__main__":
    main()
