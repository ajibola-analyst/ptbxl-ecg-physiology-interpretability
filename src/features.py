"""Hand-crafted ECG features (classical baseline).
NOTE: these are simple, transparent proxies computed at 100 Hz - not a clinical delineation algorithm."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_widths
from scipy.stats import kurtosis, skew

from src.config import FS, LEADS

_STATS = ["mean", "std", "min", "max", "skew", "kurt"]
FEATURE_NAMES = (
    ["n_beats", "hr_bpm", "sdnn_ms", "rmssd_ms", "r_width_ms"]
    + [f"{s}_{l}" for s in _STATS for l in LEADS]
    + [f"{s}_{l}" for s in ["r_amp", "st_dev", "tw_dev"] for l in LEADS]
)


def bandpass(x, fs=FS, lo=0.5, hi=40.0):
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def orient(lead):
    x = lead - np.median(lead)
    return -x if abs(x.min()) > abs(x.max()) else x


def find_r_peaks(x, fs=FS):
    """R-peak detection on an oriented single-lead signal."""
    top = x.max()
    if top <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), height=0.4 * top, prominence=0.3 * top)
    return peaks


def extract(sig, fs=FS):
    """sig: (1000, 12) array -> feature vector ordered as FEATURE_NAMES."""
    x = bandpass(sig.astype(np.float64), fs)
    lead2 = orient(x[:, 1])
    peaks = find_r_peaks(lead2, fs)
    d = {n: np.nan for n in FEATURE_NAMES}
    d["n_beats"] = len(peaks)
    rr = np.diff(peaks) / fs
    if len(rr) >= 1:
        d["hr_bpm"] = 60.0 / rr.mean()
    if len(rr) >= 3:
        d["sdnn_ms"] = 1000 * rr.std(ddof=1)
        d["rmssd_ms"] = 1000 * np.sqrt(np.mean(np.diff(rr) ** 2))
    if len(peaks) >= 1:
        d["r_width_ms"] = float(np.median(peak_widths(lead2, peaks, rel_height=0.5)[0]) * 1000 / fs)
    for j, l in enumerate(LEADS):
        v = x[:, j]
        d[f"mean_{l}"], d[f"std_{l}"] = v.mean(), v.std()
        d[f"min_{l}"], d[f"max_{l}"] = v.min(), v.max()
        d[f"skew_{l}"], d[f"kurt_{l}"] = skew(v), kurtosis(v)
    # beat-locked amplitudes (relative to a PR-segment baseline 100-200 ms before R)
    ok = [p for p in peaks if p - 20 >= 0 and p + 40 <= len(x)]
    if ok:
        for j, l in enumerate(LEADS):
            v = x[:, j]
            base = np.array([v[p - 20:p - 10].mean() for p in ok])
            d[f"r_amp_{l}"] = float(np.mean(np.array([v[p] for p in ok]) - base))
            d[f"st_dev_{l}"] = float(np.mean(np.array([v[p + 8:p + 20].mean() for p in ok]) - base))   # ~80-200 ms after R
            d[f"tw_dev_{l}"] = float(np.mean(np.array([v[p + 20:p + 40].mean() for p in ok]) - base))  # ~200-400 ms after R
    return np.array([d[n] for n in FEATURE_NAMES], dtype=np.float32)
