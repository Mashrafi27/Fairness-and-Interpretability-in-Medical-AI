# CV8502 A2 — Fairness & Interpretability

Standalone starter for Assignment 2. Baseline: DenseNet-121 (binary classification), with fairness metrics/gaps, two mitigation options, and explainability (Grad-CAM + Integrated Gradients).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Point `--csv` and `--img-root` to wherever your data already lives on HPC. You don’t need to copy it into this repo. (If you prefer short paths, you can create symlinks under `data/`.)

## Typical workflow
Choose one binary label (e.g., `Effusion`) and one demographic column (e.g., `sex`) for subgroup/fairness analysis.

```bash
# 0) Activate env
source .venv/bin/activate

# 1) Train baseline
python main.py train \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/baseline

# 2) Evaluate + export predictions (creates preds_test.csv)
python main.py eval \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion --split test \
  --group-col sex \
  --weights outputs/baseline/best.pt \
  --outdir outputs/baseline_eval

# 3) Fairness gaps (DP/EO/EOds) + per-group metrics
python scripts/fairness_report.py \
  --preds outputs/baseline_eval/preds_test.csv \
  --group-col sex \
  --label-col Effusion \
  --prob-col prob_Effusion \
  --threshold 0.5 \
  --split test \
  --outdir outputs/fairness_sex

# 4) Mitigation method 1: inverse-frequency group sampler
python main.py train \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --group-col sex --reweight-by-group \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/mitigation_reweight

# 5) Mitigation method 2: GroupDRO (worst-group loss per batch)
python main.py train \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion \
  --group-col sex --group-dro \
  --epochs 10 --batch-size 16 --lr 1e-4 --seed 1337 \
  --outdir outputs/mitigation_groupdro

# 6) Repeat eval + fairness_report for each mitigation run (update weights/outdir)

# 7) Explainability: Grad-CAM + Integrated Gradients overlays
python main.py explain \
  --csv /path/to/your.csv --img-root /path/to/images_root \
  --labels Effusion --split test \
  --group-col sex \
  --weights outputs/baseline/best.pt \
  --target-label Effusion --samples 16 \
  --outdir outputs/explain_baseline

# 8) Repeat explainability after mitigation for comparison/case studies
```

## Outputs
- `preds_<split>.csv`: image_path, split, label columns, `prob_<label>`, and group column (for fairness_report).
- `group_metrics_<split>.json`: per-group macro metrics (AUROC/AUPRC/F1/Sens@95%Spec).
- `fairness_summary.json` + `per_group_metrics.csv`: DP/EO/EOds gaps and subgroup metrics.
- `exp_<split>_####.png` + `explain_summary.csv`: Grad-CAM and IG overlays.

## Mitigation flags
- `--reweight-by-group`: inverse-frequency group sampler.
- `--group-dro`: worst-group loss per batch (requires `--group-col`).

## Deliverables checklist
- Report (8 pages): data audit + CIs, subgroup metrics, DP/EO/EOds gaps, mitigation effects, calibration/thresholds, Grad-CAM + attribution before/after mitigation, two case studies.
- Code/notebooks + one-command repro.
- Contribution note; (bonus) eval CLI.

## HPC note
- Bare repo: `/home/mashrafi.monon/repos/cv8502_a2.git`
- Working copy: `/l/users/mashrafi.monon/cv8502_a2_workdir`
- Point `--csv/--img-root` to your dataset location to avoid duplicates.
