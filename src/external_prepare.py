"""External validation, step 2: convert Georgia .hea/.mat files into the same format the model was trained on:
100 Hz, 1000 samples, 12 leads in PTB-XL order, units of millivolts.
Run: python -m src.external_prepare"""
import math

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

from src.config import FS, LEADS, N_SAMPLES, ROOT

RAW = ROOT / "data" / "external" / "georgia"
OUT = ROOT / "data" / "external" / "processed"


def parse_header(text):
    """Parse a Challenge .hea file with robust handling for header formatting and spacing."""
    lines = [ln.rstrip("\r") for ln in text.split("\n") if ln.strip()]
    first = lines[0].split()
    n_leads, fs, n_samples = int(first[1]), float(first[2]), int(first[3])
    names, gains, bases = [], [], []
    
    for ln in lines[1:1 + n_leads]:
        e = ln.split()
        names.append(e[-1])
        # Robustly parse gain and baseline even if formatted like 1000.0(0)/mV
        gain_str = e[2].split("/")[0].split("(")[0]
        base_str = e[4].split("/")[0]
        gains.append(float(gain_str))
        bases.append(float(base_str))
        
    dx = []
    for ln in lines:
        if ln.startswith("#"):
            clean = ln.lstrip("#").strip()
            # Match 'Dx:', 'Dx :', or 'dx:' regardless of spaces after '#'
            if clean.lower().startswith("dx"):
                parts = clean.split(":", 1)
                if len(parts) > 1:
                    dx += [c.strip() for c in parts[1].split(",") if c.strip()]
                    
    return {
        "n_leads": n_leads, 
        "fs": fs, 
        "n_samples": n_samples, 
        "leads": names,
        "gains": np.array(gains), 
        "baselines": np.array(bases), 
        "dx": dx
    }


def convert(val, hdr):
    """val: (n_leads, n_samples) raw integers -> (1000, 12) float32 in mV at 100 Hz, or (None, reason)."""
    if val.shape[0] != hdr["n_leads"] or (hdr["gains"] == 0).any():
        return None, "bad_shape_or_gain"
    up_names = [n.upper() for n in hdr["leads"]]
    if any(lead not in up_names for lead in LEADS):
        return None, "missing_lead"
    x = (val.astype(np.float64) - hdr["baselines"][:, None]) / hdr["gains"][:, None]   # millivolts
    x = x[[up_names.index(lead) for lead in LEADS]].T                                   # (T, 12) in PTB-XL order
    g = math.gcd(FS, int(round(hdr["fs"])))
    x = resample_poly(x, FS // g, int(round(hdr["fs"])) // g, axis=0)
    if x.shape[0] < N_SAMPLES:
        return None, "shorter_than_10s"
    x = x[:N_SAMPLES].astype(np.float32)
    if not np.isfinite(x).all():
        return None, "non_finite"
    return x, "ok"


def main():
    heas = sorted(RAW.glob("*.hea"))
    if not heas:
        raise SystemExit(f"No .hea files in {RAW}. Run: python -m src.external_download")
    OUT.mkdir(parents=True, exist_ok=True)
    X, rows, skipped = [], [], {}
    for i, hp in enumerate(heas):
        mp = hp.with_suffix(".mat")
        if not mp.exists():
            skipped["missing_mat"] = skipped.get("missing_mat", 0) + 1
            continue
        hdr = parse_header(hp.read_text(encoding="utf-8", errors="ignore"))
        x, reason = convert(loadmat(str(mp))["val"], hdr)
        if x is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        X.append(x)
        rows.append({"record": hp.stem, "fs": hdr["fs"], "codes": "|".join(hdr["dx"])})
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(heas)}")
    np.save(OUT / "X.npy", np.stack(X))
    pd.DataFrame(rows).to_csv(OUT / "meta.csv", index=False)
    print(f"Converted {len(X)} of {len(heas)} records. Skipped: {skipped or 'none'}")
    print("Saved to", OUT)


if __name__ == "__main__":
    main()