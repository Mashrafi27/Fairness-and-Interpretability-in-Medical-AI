#!/usr/bin/env python3
"""
Generate summary figures for the CV8502 A2 report:
- Fairness gaps (DP, EO, EOds) across methods
- Subgroup TPR@95% specificity (F vs M) across methods
- Baseline robustness to mild perturbations (clean vs perturbed)

Assumes the following layout under a root outputs directory:
  root/
    fairness_sex/fairness_summary.json
    fairness_reweight_sex/fairness_summary.json
    fairness_groupdro_sex/fairness_summary.json
    fairness_sex/per_group_metrics.csv
    fairness_reweight_sex/per_group_metrics.csv
    fairness_groupdro_sex/per_group_metrics.csv
    perturb_baseline/clean_metrics.json
    perturb_baseline/perturbed_metrics.json

Usage (from repo root, after copying HPC outputs):
  python scripts/make_figures.py --root outputs/outputs
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_fairness_summary(path: Path):
    data = json.loads(path.read_text())
    gaps = data["fairness_gaps"]
    return gaps["dp_gap"], gaps["eo_gap"], gaps["eods_gap"]


def plot_fairness_gaps(root: Path, outdir: Path):
    methods = ["Baseline", "Reweight", "GroupDRO"]
    dirs = ["fairness_sex", "fairness_reweight_sex", "fairness_groupdro_sex"]
    dp, eo, eods = [], [], []
    for d in dirs:
        jpath = root / d / "fairness_summary.json"
        g_dp, g_eo, g_eods = load_fairness_summary(jpath)
        dp.append(g_dp)
        eo.append(g_eo)
        eods.append(g_eods)

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(6, 4))
    plt.bar(x - width, dp, width, label="DP")
    plt.bar(x, eo, width, label="EO")
    plt.bar(x + width, eods, width, label="EOds")
    plt.xticks(x, methods)
    plt.ylabel("Gap (absolute difference)")
    plt.title("Fairness gaps across methods")
    plt.legend()
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "fig_fairness_gaps.png", dpi=200)
    plt.close()


def plot_subgroup_tpr(root: Path, outdir: Path):
    methods = ["Baseline", "Reweight", "GroupDRO"]
    dirs = ["fairness_sex", "fairness_reweight_sex", "fairness_groupdro_sex"]
    groups = ["F", "M"]
    tpr = {m: [] for m in methods}

    for m, d in zip(methods, dirs):
        df = pd.read_csv(root / d / "per_group_metrics.csv")
        for g in groups:
            row = df[df["group"] == g].iloc[0]
            tpr[m].append(row["tpr_at_95spec"])

    x = np.arange(len(groups))
    width = 0.25

    plt.figure(figsize=(6, 4))
    plt.bar(x - width, tpr["Baseline"], width, label="Baseline")
    plt.bar(x, tpr["Reweight"], width, label="Reweight")
    plt.bar(x + width, tpr["GroupDRO"], width, label="GroupDRO")
    plt.xticks(x, ["F", "M"])
    plt.ylabel("TPR @ 95% specificity")
    plt.title("Subgroup sensitivity across methods")
    plt.legend()
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "fig_subgroup_tpr.png", dpi=200)
    plt.close()


def plot_perturbation(root: Path, outdir: Path):
    clean = json.loads((root / "perturb_baseline" / "clean_metrics.json").read_text())
    pert = json.loads((root / "perturb_baseline" / "perturbed_metrics.json").read_text())
    m_clean = clean["macro"]
    m_pert = pert["macro"]

    metrics = ["auroc", "auprc", "f1@0.5", "sens@95spec"]
    labels = ["AUROC", "AUPRC", "F1@0.5", "TPR@95%Spec"]
    clean_vals = [m_clean[k] for k in metrics]
    pert_vals = [m_pert[k] for k in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, clean_vals, width, label="Clean")
    plt.bar(x + width / 2, pert_vals, width, label="Perturbed")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Score")
    plt.title("Baseline robustness to mild perturbations")
    plt.legend()
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "fig_perturbation.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Make summary figures for CV8502 A2.")
    ap.add_argument("--root", type=Path, default=Path("outputs/outputs"),
                    help="Root directory containing fairness_* and perturb_baseline.")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Directory to write figures (default: <root>/figures).")
    args = ap.parse_args()

    root = args.root
    outdir = args.outdir or (root / "figures")

    plot_fairness_gaps(root, outdir)
    plot_subgroup_tpr(root, outdir)
    plot_perturbation(root, outdir)
    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()

