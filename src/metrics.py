"""Evaluation helpers: macro AUROC, patient-level bootstrap CIs, ECE."""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import SEED


def class_aucs(y, p):
    return np.array([roc_auc_score(y[:, k], p[:, k]) if 0 < y[:, k].sum() < len(y) else np.nan
                     for k in range(y.shape[1])])


def macro_auc(y, p):
    a = class_aucs(y, p)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def patient_bootstrap(y, p, groups, n=1000, seed=SEED):
    """Resample PATIENTS (not records) with replacement. Returns 95% CI for macro and per-class AUROC."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(ug))]
    macro, per = [], []
    for _ in range(n):
        s = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by[i] for i in s])
        a = class_aucs(y[idx], p[idx])
        per.append(a)
        macro.append(np.nanmean(a))
    per = np.array(per)
    return np.percentile(macro, [2.5, 97.5]), np.nanpercentile(per, [2.5, 97.5], axis=0)


def ece(y, p, bins=10):
    y, p = y.ravel(), p.ravel()
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))
