"""Step 3: train the 1D ResNet on folds 1-8, select on fold 9, predict fold 10 (test, touched once).
Run: python -m src.train_resnet [--epochs 25]"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import CACHE, MODELS, RESULTS, SEED
from src.data import load_processed
from src.model import ECGResNet, normalize, predict


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", dev)
    X, Y, meta = load_processed()
    fold = meta["strat_fold"].values
    tr, va, te = fold <= 8, fold == 9, fold == 10
    mean, std = X[tr].mean(axis=(0, 1)), X[tr].std(axis=(0, 1)) + 1e-6   # train-fold statistics only
    np.savez(CACHE / "norm.npz", mean=mean, std=std)
    Xn = normalize(X, mean, std)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xn[tr]), torch.from_numpy(Y[tr])),
                    batch_size=a.bs, shuffle=True)
    model = ECGResNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    best, log = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        v = macro_auc(Y[va], predict(model, Xn[va], dev))
        log.append({"epoch": ep, "train_loss": tot / tr.sum(), "val_macro_auroc": v})
        print(f"epoch {ep:02d} loss {tot / tr.sum():.4f} val macro AUROC {v:.4f}")
        if v > best:
            best = v
            torch.save(model.state_dict(), MODELS / "resnet.pt")
    pd.DataFrame(log).to_csv(RESULTS / "resnet_training_log.csv", index=False)
    model.load_state_dict(torch.load(MODELS / "resnet.pt", map_location=dev))
    np.save(RESULTS / "probs_resnet_test.npy", predict(model, Xn[te], dev))
    print(f"Best validation macro AUROC: {best:.4f}. Test predictions saved.")


if __name__ == "__main__":
    main()
