# Fairness & Interpretability Analysis for Medical Image AI

Reference implementation for CV8502 Assignment 2 (Fairness & Interpretability in Medical AI). The repository provides:

- A DenseNet-121 baseline for binary medical image classification.
- Subgroup fairness evaluation (DP, EO, EOds) with calibration and threshold analysis.
- Two mitigation strategies (group reweighting and GroupDRO).
- Post-hoc explainability via Grad-CAM and Integrated Gradients, including basic sanity and stability checks.

## Environment

```bash
conda create -n cv8502_a2 python=3.9 -y
conda activate cv8502_a2
pip install -r requirements.txt
```

Point `--csv` and `--img-root` to wherever your dataset lives (e.g., NIH ChestXray14). You do not need to copy images into this repo; relative paths in the CSV are supported.

## Usage

The core entry point is `main.py`, which exposes subcommands for training, evaluation, stress testing, calibration, selective prediction, and explanations.

### Example: baseline training and evaluation

```bash
conda activate cv8502_a2

# Train baseline DenseNet-121 on Effusion with a group column `sex`
python main.py train \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --group-col sex \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/baseline

# Evaluate on test split and export predictions
python main.py eval \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion --split test \
  --group-col sex \
  --weights outputs/baseline/best.pt \
  --outdir outputs/baseline_eval
```

### Fairness metrics and mitigation

```bash
# Fairness metrics and gaps (DP/EO/EOds) from predictions
python scripts/fairness_report.py \
  --preds outputs/baseline_eval/preds_test.csv \
  --group-col sex \
  --label-col Effusion \
  --prob-col prob_Effusion \
  --threshold 0.5 \
  --split test \
  --outdir outputs/fairness_sex

# Mitigation 1: inverse-frequency group sampler
python main.py train \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --group-col sex --reweight-by-group \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/mitigation_reweight

# Mitigation 2: GroupDRO (worst-group loss per batch)
python main.py train \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --group-col sex --group-dro \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/mitigation_groupdro
```

### Explainability, calibration, and robustness

```bash
# Explanations: Grad-CAM + Integrated Gradients overlays
python main.py explain \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion --split test \
  --group-col sex \
  --weights outputs/baseline/best.pt \
  --target-label Effusion --samples 16 \
  --outdir outputs/explain_baseline

# Calibration (temperature scaling + reliability diagrams)
python main.py calibrate \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --val-split val --test-split test \
  --weights outputs/baseline/best.pt \
  --outdir outputs/calibration_baseline

# Mild perturbation stability (noise + brightness/contrast)
python scripts/perturb_eval.py \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --weights outputs/baseline/best.pt \
  --split test \
  --outdir outputs/perturb_baseline
```

## Outputs

- `preds_<split>.csv`: image path, split, label columns, `prob_<label>`, and group column.
- `group_metrics_<split>.json`: per-group macro metrics (AUROC, AUPRC, F1, TPR@95%Spec).
- `fairness_summary.json` + `per_group_metrics.csv`: fairness gaps (DP, EO, EOds) and subgroup metrics.
- `exp_<split>_####.png` + `explain_summary.csv`: Grad-CAM and IG overlays for selected cases.
- `calibration_*/reliability_*.png` and `calibration_summary.json`: calibration metrics and reliability diagrams.
- `perturb_*/clean_metrics.json` and `perturb_*/perturbed_metrics.json`: robustness of the baseline under mild perturbations.

## Mitigation flags

- `--reweight-by-group`: inverse-frequency group sampler over the specified group column.
- `--group-dro`: worst-group loss per batch (GroupDRO), requires `--group-col`.

## Notes

- Keep a lightweight working copy (e.g., on a GPU node or shared filesystem) and, if needed, a separate bare repo for syncing across machines.
- Ensure that `outputs/` remains untracked to avoid committing large artifacts.
