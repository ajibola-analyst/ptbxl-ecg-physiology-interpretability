"""Step 7 (optional): package a small demo (25 correctly-classified test ECGs + model) for the Streamlit app.
PTB-XL is CC BY 4.0, so a small sample can be redistributed with attribution.
Run: python -m src.make_demo_samples"""
import shutil

import numpy as np

from src.config import CACHE, MODELS, RESULTS, ROOT, SUPERCLASSES
from src.data import load_processed


def main():
    X, Y, meta = load_processed()
    te = np.where((meta["strat_fold"] == 10).values)[0]
    probs = np.load(RESULTS / "probs_resnet_test.npy")
    keep = []
    for k in range(len(SUPERCLASSES)):
        keep += list(np.where((Y[te][:, k] == 1) & (probs[:, k] > 0.5))[0][:5])
    sel = te[keep]
    assets = ROOT / "app" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(assets / "demo_samples.npz", X=X[sel], Y=Y[sel], ecg_id=meta.loc[sel, "ecg_id"].values)
    shutil.copy(MODELS / "resnet.pt", assets / "resnet.pt")
    shutil.copy(CACHE / "norm.npz", assets / "norm.npz")
    print("Demo assets written to", assets)


if __name__ == "__main__":
    main()
