"""Step 6: Grad-CAM + beat-aligned attention analysis (does the network look where cardiac physiology says it should?).
Run: python -m src.gradcam"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import CACHE, FIGURES, LEADS, MODELS, RESULTS, SUPERCLASSES
from src.data import load_processed
from src.features import bandpass, find_r_peaks, orient
from src.model import ECGResNet, gradcam_1d, normalize

PRE, POST = 20, 60  # samples around each R peak at 100 Hz: -200 ms .. +600 ms
T = np.arange(-PRE, POST) * 10  # ms relative to R
WINDOWS = {"QRS (-40 to +60 ms)": (T >= -40) & (T < 60),
           "ST segment / early T (+60 to +200 ms)": (T >= 60) & (T < 200),
           "T wave (+200 to +400 ms)": (T >= 200) & (T < 400)}
WINDOWS["Rest of cycle (PR/TP)"] = ~np.any(list(WINDOWS.values()), axis=0)


def main(max_per_class=200):
    X, Y, meta = load_processed()
    te = np.where((meta["strat_fold"] == 10).values)[0]
    norm = np.load(CACHE / "norm.npz")
    dev = "cpu"
    net = ECGResNet()
    net.load_state_dict(torch.load(MODELS / "resnet.pt", map_location=dev))
    Xte = X[te]
    Xn = normalize(Xte, norm["mean"], norm["std"])
    probs = np.load(RESULTS / "probs_resnet_test.npy")
    yt = Y[te]
    summary, profiles, examples = {}, {}, {}
    for k, cls in enumerate(SUPERCLASSES):
        idx = np.where((yt[:, k] == 1) & (probs[:, k] > 0.5))[0][:max_per_class]
        cams, ecgs = [], []
        for i in idx:
            _, cam = gradcam_1d(net, Xn[i], k, dev)
            lead2 = orient(bandpass(Xte[i].astype(np.float64))[:, 1])
            for p in find_r_peaks(lead2):
                if p - PRE >= 0 and p + POST <= len(cam):
                    cams.append(cam[p - PRE:p + POST])
                    ecgs.append(Xn[i][1][p - PRE:p + POST])
            if k not in examples:
                examples[k] = (Xn[i], cam)
        if not cams:
            continue
        cams = np.array(cams)
        prof = cams.mean(0)
        profiles[cls] = (prof, np.mean(ecgs, 0))
        mass = prof / prof.sum()
        summary[cls] = {"n_records": int(len(idx)), "n_beats": int(len(cams)),
                        "attention_share": {w: float(mass[m].sum()) for w, m in WINDOWS.items()},
                        "enrichment_vs_uniform": {w: float(mass[m].sum() / m.mean()) for w, m in WINDOWS.items()}}
    (RESULTS / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, len(profiles), figsize=(3.2 * len(profiles), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (cls, (prof, ecg)) in zip(axes, profiles.items()):
        ax.plot(T, prof, color="crimson")
        ax.set_title(cls)
        ax.set_xlabel("ms from R peak")
        ax2 = ax.twinx()
        ax2.plot(T, ecg, color="gray", alpha=0.6, lw=1)
        ax2.set_yticks([])
        ax.axvspan(-40, 60, color="gold", alpha=0.15)
        ax.axvspan(60, 400, color="royalblue", alpha=0.08)
    axes[0].set_ylabel("Mean attention")
    fig.suptitle("Beat-aligned attention (gold = QRS, blue = ST-T; lead II in grey)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "gradcam_beat_aligned.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(len(examples), 2, figsize=(11, 2.2 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, (k, (xn, cam)) in enumerate(sorted(examples.items())):
        for col, lead in enumerate(["II", "V2"]):
            a = axes[row, col]
            a.plot(xn[LEADS.index(lead)], color="black", lw=0.8)
            a.imshow(cam[None], aspect="auto", cmap="Reds", alpha=0.45, extent=[0, len(cam), *a.get_ylim()])
            a.set_title(f"{SUPERCLASSES[k]} - lead {lead}", fontsize=8)
            a.set_xticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "gradcam_examples.png", dpi=200)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
