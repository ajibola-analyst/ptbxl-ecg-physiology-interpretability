"""Torch-free statistics helpers for the external validation (kept separate so tests run without PyTorch)."""
import numpy as np

from src.config import SEED
from src.external_mapping import CLASSES, MIN_POSITIVES, label_record
from src.metrics import class_aucs, ece


def build_labels(code_lists, strict=False):
    inc, Y = [], []
    for codes in code_lists:
        ok, y = label_record(codes, strict)
        inc.append(ok)
        Y.append(y)
    return np.array(inc, dtype=bool), np.array(Y, dtype=np.float32)


def evaluable_classes(Y):
    n, pos = len(Y), Y.sum(0)
    return [k for k in range(Y.shape[1]) if pos[k] >= MIN_POSITIVES and n - pos[k] >= MIN_POSITIVES]


def bootstrap_ci(y, p, n=1000, seed=SEED):
    """Record-level bootstrap (the Georgia files have no patient IDs)."""
    rng = np.random.default_rng(seed)
    N = len(y)
    per = np.array([class_aucs(y[i], p[i]) for i in (rng.integers(0, N, N) for _ in range(n))])
    return np.percentile(np.nanmean(per, axis=1), [2.5, 97.5]), np.nanpercentile(per, [2.5, 97.5], axis=0)


def evaluate_probs(Y, P, cols, n_boot=1000):
    y, p = Y[:, cols], P[:, cols]
    aucs = class_aucs(y, p)
    (mlo, mhi), (plo, phi) = bootstrap_ci(y, p, n_boot)
    names = [CLASSES[k] for k in cols]
    return {"per_class_auroc": {c: float(a) for c, a in zip(names, aucs)},
            "per_class_ci95": {c: [float(plo[i]), float(phi[i])] for i, c in enumerate(names)},
            "macro_auroc": float(np.mean(aucs)), "macro_ci95": [float(mlo), float(mhi)], "ece": float(ece(y, p))}
