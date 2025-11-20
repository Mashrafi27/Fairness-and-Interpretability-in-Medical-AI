#!/usr/bin/env python3
"""
Compute subgroup metrics and fairness gaps (DP, EO, EOds) from a predictions CSV.

Expected CSV columns:
- label column (ground truth), e.g., "label" (configurable via --label-col)
- probability column for the positive class, e.g., "prob" (configurable via --prob-col)
- group column (demographic attribute), provided via --group-col
- optional: "split" to filter train/val/test with --split
- optional: discrete prediction column (configurable via --pred-col). If absent, we
  threshold probabilities using --threshold.

Outputs:
- <outdir>/per_group_metrics.csv : per-group metrics
- <outdir>/fairness_summary.json : gap metrics and metadata
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return AUROC or NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return AUPRC or NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def tpr_at_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float
) -> Tuple[float, Optional[float]]:
    """
    Compute TPR at the highest threshold that still achieves target specificity.
    Returns (tpr, threshold). If not achievable, returns (nan, None).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return float("nan"), None
    best_idx = idx[-1]  # lowest threshold before dropping below target specificity
    return float(tpr[best_idx]), float(thresholds[best_idx])


def ppv_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Positive predictive value at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    if y_pred.sum() == 0:
        return float("nan")
    return float(precision_score(y_true, y_pred))


def group_stats(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    prob_col: str,
    pred_col: str,
    target_spec: float,
    threshold: float,
) -> List[Dict]:
    rows = []
    for group_value, g in df.groupby(group_col):
        y_true = g[label_col].to_numpy()
        y_prob = g[prob_col].to_numpy()
        y_pred = g[pred_col].to_numpy()

        tpr_95, thr_95 = tpr_at_specificity(y_true, y_prob, target_spec=target_spec)
        rows.append(
            {
                "group": group_value,
                "count": int(len(g)),
                "prevalence": float(y_true.mean()) if len(g) else float("nan"),
                "auroc": safe_auc(y_true, y_prob),
                "auprc": safe_auprc(y_true, y_prob),
                "brier": float(brier_score_loss(y_true, y_prob)),
                "tpr_at_{:.0f}spec".format(target_spec * 100): tpr_95,
                "threshold_at_{:.0f}spec".format(target_spec * 100): thr_95,
                "ppv_at_threshold": ppv_at_threshold(y_true, y_prob, threshold),
            }
        )
    return rows


def fairness_gaps(
    df: pd.DataFrame, group_col: str, label_col: str, pred_col: str
) -> Dict[str, float]:
    """
    Compute DP, EO, EOds gaps (max pairwise absolute differences across groups).
    - DP: P(y_hat=1 | g)
    - EO: TPR differences conditioned on y=1
    - EOds: max of TPR and FPR differences
    """
    groups = []
    positive_rates = []
    tprs = []
    fprs = []

    for group_value, g in df.groupby(group_col):
        y_true = g[label_col].to_numpy()
        y_pred = g[pred_col].to_numpy()
        groups.append(group_value)

        positive_rate = y_pred.mean() if len(y_pred) else float("nan")
        positive_rates.append(positive_rate)

        pos_mask = y_true == 1
        neg_mask = y_true == 0

        tpr = y_pred[pos_mask].mean() if pos_mask.any() else float("nan")
        fpr = y_pred[neg_mask].mean() if neg_mask.any() else float("nan")

        tprs.append(tpr)
        fprs.append(fpr)

    def max_gap(values: List[float]) -> float:
        clean = [v for v in values if not np.isnan(v)]
        if len(clean) < 2:
            return float("nan")
        diffs = [abs(a - b) for i, a in enumerate(clean) for b in clean[i + 1 :]]
        return max(diffs)

    return {
        "groups": groups,
        "dp_gap": max_gap(positive_rates),
        "eo_gap": max_gap(tprs),
        "eods_gap": max(max_gap(tprs), max_gap(fprs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute subgroup metrics and fairness gaps from predictions."
    )
    parser.add_argument("--preds", required=True, type=Path, help="Predictions CSV")
    parser.add_argument(
        "--group-col", required=True, help="Demographic/group column in the CSV"
    )
    parser.add_argument(
        "--label-col", default="label", help="Ground truth column (binary)"
    )
    parser.add_argument(
        "--prob-col",
        default="prob",
        help="Probability column for positive class (float in [0,1])",
    )
    parser.add_argument(
        "--pred-col",
        default=None,
        help="Optional binary prediction column; if missing, will threshold prob-col",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to binarize prob-col when pred-col is absent",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="If set, filter to this split column value (expects 'split' column)",
    )
    parser.add_argument(
        "--target-specificity",
        type=float,
        default=0.95,
        help="Target specificity for TPR@Spec metric",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory; defaults to preds parent / fairness_<group-col>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.preds)

    if args.split is not None:
        if "split" not in df.columns:
            raise ValueError("split column not found in predictions CSV")
        df = df[df["split"] == args.split]

    for col in (args.group_col, args.label_col, args.prob_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from predictions CSV")

    pred_col = args.pred_col or "__y_pred_tmp"
    if args.pred_col and args.pred_col not in df.columns:
        raise ValueError(f"Column '{args.pred_col}' missing from predictions CSV")
    if not args.pred_col:
        df[pred_col] = (df[args.prob_col] >= args.threshold).astype(int)

    outdir = (
        args.outdir
        if args.outdir is not None
        else args.preds.parent / f"fairness_{args.group_col}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    per_group = group_stats(
        df=df,
        group_col=args.group_col,
        label_col=args.label_col,
        prob_col=args.prob_col,
        pred_col=pred_col,
        target_spec=args.target_specificity,
        threshold=args.threshold,
    )
    per_group_df = pd.DataFrame(per_group)
    per_group_path = outdir / "per_group_metrics.csv"
    per_group_df.to_csv(per_group_path, index=False)

    gaps = fairness_gaps(df, group_col=args.group_col, label_col=args.label_col, pred_col=pred_col)
    summary = {
        "preds_file": str(args.preds),
        "group_col": args.group_col,
        "label_col": args.label_col,
        "prob_col": args.prob_col,
        "pred_col": pred_col if args.pred_col else None,
        "threshold": args.threshold,
        "target_specificity": args.target_specificity,
        "fairness_gaps": gaps,
        "groups": gaps["groups"],
    }
    summary_path = outdir / "fairness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote per-group metrics to {per_group_path}")
    print(f"Wrote fairness summary to {summary_path}")


if __name__ == "__main__":
    main()
