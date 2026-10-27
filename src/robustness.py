"""Step 5: robustness to synthetic noise on the test fold (both models).
Run: python -m src.robustness"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from src.config import CACHE, FIGURES, MODELS, RESULTS, SEED
from src.data import load_processed
from src.features import extract
from src.metrics import macro_auc
from src.model import ECGResNet, normalize, predict


def gaussian_noise(X, snr_db, rng):
    Xc = X - X.mean(axis=1, keepdims=True)
    power = (Xc ** 2).mean(axis=1, keepdims=True)
    return X + rng.standard_normal(X.shape).astype(np.float32) * np.sqrt(power / 10 ** (snr_db / 10))


def baseline_wander(X, rng, freq=0.3, fs=100):
    t = np.arange(X.shape[1]) / fs
    amp = 0.5 * X.std(axis=1, keepdims=True)
    ph = rng.uniform(0, 2 * np.pi, (X.shape[0], 1, 1))
    return X + (amp * np.sin(2 * np.pi * freq * t[None, :, None] + ph)).astype(np.float32)


def main():
    X, Y, meta = load_processed()
    te = (meta["strat_fold"] == 10).values
    Xt, yt = X[te], Y[te]
    norm = np.load(CACHE / "norm.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ECGResNet().to(dev)
    net.load_state_dict(torch.load(MODELS / "resnet.pt", map_location=dev))
    clfs = joblib.load(MODELS / "classical.joblib")
    rng = np.random.default_rng(SEED)
    conds = [("clean", Xt)] + [(f"gaussian SNR {s} dB", gaussian_noise(Xt, s, rng)) for s in (20, 10, 5, 0)] \
            + [("baseline wander 0.3 Hz", baseline_wander(Xt, rng))]
    rows = []
    for name, Xc in conds:
        p_res = predict(net, normalize(Xc, norm["mean"], norm["std"]), dev)
        Fm = np.vstack(Parallel(n_jobs=-1, batch_size=64)(delayed(extract)(x) for x in Xc))
        p_cl = np.stack([c.predict_proba(Fm)[:, 1] for c in clfs], 1)
        rows.append({"condition": name, "resnet_macro_auroc": macro_auc(yt, p_res),
                     "classical_macro_auroc": macro_auc(yt, p_cl)})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "robustness.csv", index=False)
    ax = df.set_index("condition")[["resnet_macro_auroc", "classical_macro_auroc"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Macro AUROC (test fold 10)")
    ax.set_ylim(0.4, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "robustness.png", dpi=200)


if __name__ == "__main__":
    main()
