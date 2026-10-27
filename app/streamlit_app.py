import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LEADS, SUPERCLASSES  # noqa: E402
from src.model import ECGResNet, gradcam_1d, normalize  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="ECG interpretability demo", layout="wide")
st.title("12-lead ECG classifier with physiological interpretation")
st.warning("Research demonstration only - NOT a medical device and not for clinical use.")


@st.cache_resource
def load():
    net = ECGResNet()
    net.load_state_dict(torch.load(ASSETS / "resnet.pt", map_location="cpu"))
    net.eval()
    return net, np.load(ASSETS / "norm.npz"), np.load(ASSETS / "demo_samples.npz")


net, norm, demo = load()
X, Y, ids = demo["X"], demo["Y"], demo["ecg_id"]
i = st.selectbox("Choose a test ECG (PTB-XL fold 10)", range(len(X)),
                 format_func=lambda j: f"ecg_id {ids[j]} - true: {', '.join(c for c, v in zip(SUPERCLASSES, Y[j]) if v)}")
xn = normalize(X[i:i + 1], norm["mean"], norm["std"])[0]
with torch.no_grad():
    probs = torch.sigmoid(net(torch.from_numpy(xn[None]))).numpy()[0]
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Predicted probabilities")
    for c, p in zip(SUPERCLASSES, probs):
        st.progress(float(p), text=f"{c}: {p:.2f}")
with c2:
    cls = st.selectbox("Explain class", SUPERCLASSES, index=int(np.argmax(probs)))
    _, cam = gradcam_1d(net, xn, SUPERCLASSES.index(cls))
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    for ax, lead in zip(axes, ["II", "V2"]):
        ax.plot(np.arange(1000) / 100, xn[LEADS.index(lead)], color="black", lw=0.8)
        ax.imshow(cam[None], aspect="auto", cmap="Reds", alpha=0.45, extent=[0, 10, *ax.get_ylim()])
        ax.set_ylabel(lead)
    axes[1].set_xlabel("seconds")
    st.pyplot(fig)
st.caption("Red shading = Grad-CAM attention for the selected class. Data: PTB-XL (CC BY 4.0), Wagner et al. 2020/2022.")
