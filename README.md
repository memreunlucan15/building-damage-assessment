# Building Damage Assessment

![Tests](https://github.com/memreunlucan15/building-damage-assessment/actions/workflows/tests.yml/badge.svg)

A computer engineering capstone project that classifies buildings as **intact** or **damaged** from post-earthquake satellite imagery. It compares optical RGB, synthetic aperture radar (SAR), and multimodal models, with a focus on detecting damaged buildings in a heavily imbalanced dataset.

Built with **Python, PyTorch, and Flask**. The web interface supports batch prediction, Grad-CAM visualizations, adjustable thresholds, and CSV export. Larger scenes can be explored with grid scanning or optional YOLO building detection.

## Results

The dataset contains approximately **4,029 building samples**, including **169 damaged buildings (~4%)**. Training uses balanced sampling and focal loss.

Selected results, averaged across five cross-validation folds:

| Model | Recall | Precision | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| **Optical — ResNet18** | 0.686 | 0.565 | **0.616** | 0.673 |
| SAR — SAR-HUB | 0.321 | 0.286 | 0.202 | 0.240 |
| Optical + SAR — SAR-HUB | **0.729** | 0.499 | 0.590 | **0.687** |

Optical ResNet18 achieved the highest F1 among the evaluated configurations. SAR-HUB fusion improved recall at the cost of precision. The application averages predictions from the five optical fold models; the table reports cross-validation performance, not a separate evaluation of this ensemble.

![Grad-CAM examples](assets/gradcam.png)

## Setup

Use Python 3.10+ in a virtual environment:

```bash
git clone https://github.com/memreunlucan15/building-damage-assessment.git
cd building-damage-assessment
python -m pip install -r requirements.txt
```

**Data and trained checkpoints are not included.** Training expects the dataset under `earthquake_building_dataset/`, with class-labelled folders and optical `*_opt.mat` files containing the `x3` RGB array. Paths and training settings are defined in [`config.py`](config.py).

Train the optical model with five-fold cross-validation:

```bash
python cross_validate_mm.py --modalities opt --tag opt --folds 5 --epochs 25
```

This produces the checkpoints and results in `outputs/cv_opt/`. Prediction requires the trained `fold*.pt` files; keep `cv_results.json` alongside them to use the selected decision threshold.

## Usage

Start the web interface:

```bash
python app.py
```

Open [localhost:5000](http://127.0.0.1:5000) and upload PNG, JPG, TIF, or optical MAT files. The classifier was trained on building-centred crops; use the cropping or scene-analysis tools for larger images.

For command-line prediction on optical MAT files:

```bash
python predict.py path/to/samples/ --csv predictions.csv
```

## Tests

Tests use synthetic inputs and do not require the dataset or trained checkpoints.

```bash
python -m pip install pytest
python -m pytest tests/ -q
```
