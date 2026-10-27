"""Redraw the robustness and second-dataset figures with plain-English labels.
Reads only saved result files (results/robustness.csv, results/external_georgia.json). No retraining, runs in seconds.
Run: python -m src.make_figures"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES, RESULTS

BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT = "#1f77b4", "#9ecae1", "#ff7f0e", "#fdd0a2"
NOISE_LABELS = {"clean": "Clean", "gaussian SNR 20 dB": "Mild noise\n(20 dB)", "gaussian SNR 10 dB": "Medium noise\n(10 dB)",
                "gaussian SNR 5 dB": "Strong noise\n(5 dB)", "gaussian SNR 0 dB": "Very strong\nnoise (0 dB)",
                "baseline wander 0.3 Hz": "Slow drift\n(0.3 Hz)"}


def _bars(ax, groups, series):
    """series: list of (label, colour, values). Draws grouped bars with the value printed on each bar."""
    n = len(series)
    w = 0.8 / n
    for i, (label, colour, vals) in enumerate(series):
        xs = np.arange(len(groups)) + (i - (n - 1) / 2) * w
        ax.bar(xs, vals, w, label=label, color=colour, edgecolor="white", linewidth=0.5)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score (AUROC, higher is better)\naxis starts at 0.5 = guessing", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)


def robustness_figure():
    df = pd.read_csv(RESULTS / "robustness.csv")
    groups = [NOISE_LABELS.get(c, c) for c in df["condition"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _bars(ax, groups, [("1D ResNet (AI model)", BLUE, df["resnet_macro_auroc"].tolist()),
                       ("Simple method", ORANGE, df["classical_macro_auroc"].tolist())])
    fig.savefig(FIGURES / "robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def external_figure():
    res = json.loads((RESULTS / "external_georgia.json").read_text())["main"]
    classes = res["evaluated_classes"]
    groups = classes + ["Average"]

    def vals(model, key):
        d = res[model][key]
        return [d["per_class_auroc"][c] for c in classes] + [d["macro_auroc"]]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    _bars(ax, groups, [("1D ResNet: PTB-XL (home dataset)", BLUE, vals("1D ResNet", "ptbxl_fold10")),
                       ("1D ResNet: Georgia (new dataset)", BLUE_LIGHT, vals("1D ResNet", "external")),
                       ("Simple method: PTB-XL (home dataset)", ORANGE, vals("Classical", "ptbxl_fold10")),
                       ("Simple method: Georgia (new dataset)", ORANGE_LIGHT, vals("Classical", "external"))])
    fig.savefig(FIGURES / "external_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Each figure is drawn only if its result file exists, so this is safe to run at any point in the pipeline."""
    if (RESULTS / "robustness.csv").exists():
        robustness_figure()
        print("Saved figures/robustness.png")
    else:
        print("Skipped robustness figure (results/robustness.csv not found).")
    if (RESULTS / "external_georgia.json").exists():
        external_figure()
        print("Saved figures/external_validation.png")
    else:
        print("Skipped second-dataset figure (results/external_georgia.json not found).")


if __name__ == "__main__":
    main()
