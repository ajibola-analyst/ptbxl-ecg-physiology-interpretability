import numpy as np
import pandas as pd

from src.data import add_superclass, labels_matrix
from src.features import FEATURE_NAMES, extract
from src.metrics import macro_auc, patient_bootstrap


def synthetic_ecg(hr=75, fs=100, seconds=10):
    t = np.arange(seconds * fs) / fs
    period = 60 / hr
    sig = np.zeros_like(t)
    for c in np.arange(0.5, seconds, period):
        sig += np.exp(-((t - c) ** 2) / (2 * 0.01 ** 2))               # sharp "R wave"
        sig += 0.3 * np.exp(-((t - c - 0.28) ** 2) / (2 * 0.04 ** 2))  # broad "T wave"
    return np.tile(sig[:, None], (1, 12)).astype(np.float32)


def test_features_recover_heart_rate():
    f = dict(zip(FEATURE_NAMES, extract(synthetic_ecg(hr=75))))
    assert abs(f["hr_bpm"] - 75) < 3
    assert len(extract(synthetic_ecg())) == len(FEATURE_NAMES)


def test_superclass_aggregation():
    scp = pd.DataFrame({"diagnostic": [1, 1, 1], "diagnostic_class": ["NORM", "MI", "CD"]},
                       index=["NORM", "IMI", "CLBBB"])
    db = pd.DataFrame({"scp_codes": [{"NORM": 100.0}, {"IMI": 50.0, "CLBBB": 0.0}, {"XYZ": 1.0}]})
    db = add_superclass(db, scp)
    assert db["superclasses"].tolist() == [["NORM"], ["CD", "MI"], []]
    Y = labels_matrix(db)
    assert Y[1].tolist() == [0, 1, 0, 1, 0]


def test_metrics_and_patient_bootstrap():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 5)) > 0.5).astype(float)
    p = np.clip(y * 0.6 + rng.random((300, 5)) * 0.4, 0, 1)
    assert macro_auc(y, p) > 0.9
    (lo, hi), _ = patient_bootstrap(y, p, np.arange(300) // 2, n=50)
    assert lo <= hi
