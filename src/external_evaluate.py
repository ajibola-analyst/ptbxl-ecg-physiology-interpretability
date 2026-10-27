"""External validation, step 3: test the PTB-XL-trained models, unchanged, on the Georgia database.
Nothing is retrained or re-tuned. Classes with fewer than MIN_POSITIVES positives are not evaluated (MI in Georgia).
Run: python -m src.external_evaluate"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CACHE, FIGURES, MODELS, RESULTS, ROOT
from src.external_mapping import CLASSES, MIN_POSITIVES
from src.external_stats import build_labels, evaluable_classes, evaluate_probs
from src.metrics import class_aucs, ece

EXT = ROOT / "data" / "external" / "processed"
MODEL_NAMES = ("1D ResNet", "Classical")
PROB_FILES = {"1D ResNet": "probs_resnet_test.npy", "Classical": "probs_classical_test.npy"}


def predict_all(Xi):
    """Run both trained models on external ECGs (n, 1000, 12). Heavy imports live here on purpose."""
    import joblib
    import torch
    from joblib import Parallel, delayed

    from src.features import extract
    from src.model import ECGResNet, normalize, predict

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = np.load(CACHE / "norm.npz")   # PTB-XL training-fold statistics; nothing is re-fitted
    net = ECGResNet().to(dev)
    net.load_state_dict(torch.load(MODELS / "resnet.pt", map_location=dev))
    p_res = predict(net, normalize(Xi, norm["mean"], norm["std"]), dev)
    print("Extracting features for the classical model (a few minutes)...")
    Fm = np.vstack(Parallel(n_jobs=-1, batch_size=64)(delayed(extract)(x) for x in Xi))
    p_cl = np.stack([c.predict_proba(Fm)[:, 1] for c in joblib.load(MODELS / "classical.joblib")], 1)
    return {"1D ResNet": p_res, "Classical": p_cl}


def ptbxl_reference(cols, probs_file):
    """Same classes, PTB-XL test fold 10, from the saved test-set probabilities."""
    Y = np.load(CACHE / "Y.npy")
    meta = pd.read_csv(CACHE / "meta.csv")
    te = (meta["strat_fold"] == 10).values
    y, p = Y[te][:, cols], np.load(RESULTS / probs_file)[:, cols]
    aucs = class_aucs(y, p)
    return {"per_class_auroc": {CLASSES[k]: float(a) for k, a in zip(cols, aucs)},
            "macro_auroc": float(np.mean(aucs)), "ece": float(ece(y, p))}


def amplitude_report(Xext):
    """Sanity check that units and lead order match PTB-XL: median per-lead signal spread (mV), Georgia vs PTB-XL."""
    meta = pd.read_csv(CACHE / "meta.csv")
    idx = np.where((meta["strat_fold"] == 10).values)[0]
    Xp = np.load(CACHE / "X.npy", mmap_mode="r")[idx]
    out = {"ptbxl_lead_std_mV": [float(v) for v in np.median(Xp.std(axis=1), axis=0)],
           "georgia_lead_std_mV": [float(v) for v in np.median(Xext.std(axis=1), axis=0)]}
    ratio = np.array(out["georgia_lead_std_mV"]) / np.array(out["ptbxl_lead_std_mV"])
    out["ratio_min"], out["ratio_max"] = float(ratio.min()), float(ratio.max())
    out["looks_consistent"] = bool(0.4 < ratio.min() and ratio.max() < 2.5)
    return out


def run_variant(Y, P, n_boot):
    cols = evaluable_classes(Y)
    res = {"evaluated_classes": [CLASSES[k] for k in cols]}
    for name, p in P.items():
        res[name] = {"external": evaluate_probs(Y, p, cols, n_boot),
                     "ptbxl_fold10": ptbxl_reference(cols, PROB_FILES[name])}
    return res


def fmt(v, ci=None):
    return f"{v:.3f}" if ci is None else f"{v:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


def results_table(variant):
    head = " | ".join(f"{m}: PTB-XL | {m}: Georgia (95% CI) | {m}: change" for m in MODEL_NAMES)
    rows = [f"| Class | {head} |", "|---|" + "---|---|---|" * len(MODEL_NAMES)]
    for c in variant["evaluated_classes"] + ["Macro average"]:
        cells = []
        for m in MODEL_NAMES:
            e, r = variant[m]["external"], variant[m]["ptbxl_fold10"]
            if c == "Macro average":
                a, ci, b = e["macro_auroc"], e["macro_ci95"], r["macro_auroc"]
            else:
                a, ci, b = e["per_class_auroc"][c], e["per_class_ci95"][c], r["per_class_auroc"][c]
            cells += [fmt(b), fmt(a, ci), f"{a - b:+.3f}"]
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(n_boot=1000):
    X = np.load(EXT / "X.npy")
    meta = pd.read_csv(EXT / "meta.csv", dtype=str, keep_default_na=False)
    codes = [c.split("|") if c else [] for c in meta["codes"]]
    inc_main, Y_main_all = build_labels(codes, strict=False)
    inc_strict, Y_strict_all = build_labels(codes, strict=True)
    assert (inc_strict <= inc_main).all()
    amp = amplitude_report(X[inc_main])
    print(f"Amplitude check (Georgia / PTB-XL, per-lead ratio): {amp['ratio_min']:.2f} to {amp['ratio_max']:.2f}")
    if not amp["looks_consistent"]:
        print("WARNING: amplitudes differ a lot from PTB-XL. Check units, lead order and resampling before trusting results.")
    P = predict_all(X[inc_main])
    Ym, within = Y_main_all[inc_main], inc_strict[inc_main]
    Ys = Y_strict_all[inc_strict]
    main_v = run_variant(Ym, P, n_boot)
    strict_v = run_variant(Ys, {k: v[within] for k, v in P.items()}, n_boot)
    counts = {c: int(Ym[:, k].sum()) for k, c in enumerate(CLASSES)}
    out = {"n_records_converted": int(len(X)), "n_included": int(inc_main.sum()),
           "n_excluded": int((~inc_main).sum()), "n_included_strict": int(inc_strict.sum()),
           "positives_per_class_main": counts, "min_positives_rule": MIN_POSITIVES,
           "not_evaluated": [c for k, c in enumerate(CLASSES) if k not in evaluable_classes(Ym)],
           "amplitude_check": amp, "main": main_v, "strict_sensitivity": strict_v}
    (RESULTS / "external_georgia.json").write_text(json.dumps(out, indent=2))

    md = ["### Georgia sample", "", "| Item | Count |", "|---|---|",
          f"| Records converted | {out['n_records_converted']:,} |",
          f"| Included (status for the five classes is known) | {out['n_included']:,} |",
          f"| Excluded (unmapped codes only, or prolonged QT) | {out['n_excluded']:,} |", "",
          "| Class | Positives | Evaluated? |", "|---|---|---|"]
    for k, c in enumerate(CLASSES):
        md.append(f"| {c} | {counts[c]:,} | {'yes' if k in evaluable_classes(Ym) else f'no (fewer than {MIN_POSITIVES} positives)'} |")
    md += ["", "### Main analysis (all mapped codes)", "", results_table(main_v), "",
           "### Sensitivity: strict mapping (direct PTB-XL equivalents only)", "", results_table(strict_v)]
    (RESULTS / "external_georgia.md").write_text("\n".join(md))
    print("\n".join(md))

    cls = main_v["evaluated_classes"]
    series = [(m, t) for m in MODEL_NAMES for t in ("ptbxl_fold10", "external")]
    vals = [[main_v[m][t]["per_class_auroc"][c] for c in cls] for m, t in series]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.2
    for i, ((m, t), v) in enumerate(zip(series, vals)):
        ax.bar(np.arange(len(cls)) + (i - 1.5) * w, v, w,
               label=f"{m} - {'PTB-XL (internal)' if t == 'ptbxl_fold10' else 'Georgia (external)'}")
    ax.set_xticks(np.arange(len(cls)))
    ax.set_xticklabels(cls)
    ax.set_ylim(min(0.5, min(map(min, vals)) - 0.05), 1.0)
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_validation.png", dpi=200)


if __name__ == "__main__":
    main()
