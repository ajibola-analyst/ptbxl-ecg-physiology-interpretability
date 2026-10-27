"""Step 4: test-fold (fold 10) evaluation: AUROC + patient-level bootstrap CIs, AUPRC, calibration, subgroups.
Run: python -m src.evaluate"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import FIGURES, RESULTS, SUPERCLASSES
from src.data import load_processed
from src.metrics import class_aucs, ece, macro_auc, patient_bootstrap

MODELS = {"Classical (hand-crafted + gradient boosting)": "probs_classical_test.npy",
          "1D ResNet": "probs_resnet_test.npy"}


def main():
    _, Y, meta = load_processed()
    te = (meta["strat_fold"] == 10).values
    y, m = Y[te], meta[te].reset_index(drop=True)
    groups = m["patient_id"].values
    out, probs = {}, {}
    for name, f in MODELS.items():
        p = np.load(RESULTS / f)
        probs[name] = p
        (lo, hi), (plo, phi) = patient_bootstrap(y, p, groups)
        pc = class_aucs(y, p)
        out[name] = {
            "macro_auroc": macro_auc(y, p), "macro_auroc_ci95": [float(lo), float(hi)],
            "per_class_auroc": {c: float(pc[k]) for k, c in enumerate(SUPERCLASSES)},
            "per_class_auroc_ci95": {c: [float(plo[k]), float(phi[k])] for k, c in enumerate(SUPERCLASSES)},
            "macro_auprc": float(np.mean([average_precision_score(y[:, k], p[:, k]) for k in range(5)])),
            "mean_brier": float(np.mean([brier_score_loss(y[:, k], p[:, k]) for k in range(5)])),
            "ece_pooled": ece(y, p),
        }
        sub = {}
        for sx in sorted(m["sex"].unique()):
            mk = (m["sex"] == sx).values
            sub[f"sex={sx}"] = {"n": int(mk.sum()), "macro_auroc": macro_auc(y[mk], p[mk])}
        bins = pd.cut(m["age_clean"], [-1, 39, 59, 74, 200], labels=["<40", "40-59", "60-74", ">=75"])
        for b in bins.cat.categories:
            mk = (bins == b).values
            sub[f"age {b}"] = {"n": int(mk.sum()), "macro_auroc": macro_auc(y[mk], p[mk])}
        out[name]["subgroups"] = sub
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))

    md = ["| Model | Macro AUROC (95% CI) | Macro AUPRC | ECE |", "|---|---|---|---|"]
    for n, r in out.items():
        md.append(f"| {n} | {r['macro_auroc']:.3f} ({r['macro_auroc_ci95'][0]:.3f}-{r['macro_auroc_ci95'][1]:.3f}) "
                  f"| {r['macro_auprc']:.3f} | {r['ece_pooled']:.3f} |")
    md += ["", "| Model | " + " | ".join(SUPERCLASSES) + " |", "|---|" + "---|" * 5]
    for n, r in out.items():
        md.append(f"| {n} | " + " | ".join(
            f"{r['per_class_auroc'][c]:.3f} ({r['per_class_auroc_ci95'][c][0]:.3f}-{r['per_class_auroc_ci95'][c][1]:.3f})"
            for c in SUPERCLASSES) + " |")
    md += ["", "| Subgroup | n | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    first = list(MODELS)[0]
    for g in out[first]["subgroups"]:
        md.append(f"| {g} | {out[first]['subgroups'][g]['n']} | " + " | ".join(
            f"{out[n]['subgroups'][g]['macro_auroc']:.3f}" for n in MODELS) + " |")
    (RESULTS / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    for i, (n, r) in enumerate(out.items()):
        ax.bar(np.arange(5) + i * w, [r["per_class_auroc"][c] for c in SUPERCLASSES], w, label=n)
    ax.set_xticks(np.arange(5) + w / 2)
    ax.set_xticklabels(SUPERCLASSES)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (test fold 10)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_auroc.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for n, p in probs.items():
        fr, mp = calibration_curve(y.ravel(), p.ravel(), n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", label=n)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
