"""Step 2: hand-crafted features + gradient boosting (one binary model per superclass).
Run: python -m src.train_classical"""
import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.config import CACHE, MODELS, RESULTS, SEED, SUPERCLASSES
from src.data import load_processed
from src.features import extract


def main():
    X, Y, meta = load_processed()
    fpath = CACHE / "features.npy"
    if fpath.exists():
        Fm = np.load(fpath)
    else:
        print("Extracting features (a few minutes)...")
        Fm = np.vstack(Parallel(n_jobs=-1, batch_size=64)(delayed(extract)(x) for x in X))
        np.save(fpath, Fm)
    fold = meta["strat_fold"].values
    tr, va, te = fold <= 8, fold == 9, fold == 10
    clfs, pv, pt = [], [], []
    for k, name in enumerate(SUPERCLASSES):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, early_stopping=False, random_state=SEED)
        clf.fit(Fm[tr], Y[tr, k])
        clfs.append(clf)
        pv.append(clf.predict_proba(Fm[va])[:, 1])
        pt.append(clf.predict_proba(Fm[te])[:, 1])
        print(f"{name}: val AUROC = {roc_auc_score(Y[va, k], pv[-1]):.3f}")
    pv, pt = np.stack(pv, 1), np.stack(pt, 1)
    print(f"Validation macro AUROC: {np.mean([roc_auc_score(Y[va, k], pv[:, k]) for k in range(5)]):.3f}")
    np.save(RESULTS / "probs_classical_test.npy", pt)
    joblib.dump(clfs, MODELS / "classical.joblib")


if __name__ == "__main__":
    main()
