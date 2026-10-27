"""Step 1: read 100 Hz waveforms + labels, keep records with >=1 diagnostic superclass, cache to .npy.
Run:  python -m src.prepare_data"""
import numpy as np
import wfdb
from tqdm import tqdm

from src.config import CACHE, DATA_DIR, FS, N_SAMPLES
from src.data import add_superclass, labels_matrix, load_metadata


def main():
    db, scp = load_metadata()
    print(f"Records in metadata: {len(db)}")
    db = add_superclass(db, scp)
    db = db[db["superclasses"].map(len) > 0].copy()
    print(f"Records with >=1 diagnostic superclass: {len(db)}")
    Y = labels_matrix(db)
    X = np.zeros((len(db), N_SAMPLES, 12), dtype=np.float32)
    for i, fn in enumerate(tqdm(db["filename_lr"].values, desc="Reading ECGs")):
        sig, meta = wfdb.rdsamp(str(DATA_DIR / fn))
        assert sig.shape == (N_SAMPLES, 12), (fn, sig.shape)
        if i == 0:
            assert meta["fs"] == FS, meta["fs"]
            print("Lead order:", meta["sig_name"])
            assert meta["sig_name"][1].upper() == "II", "Lead II is expected at index 1"
        X[i] = sig
    assert np.isfinite(X).all(), "Non-finite values found in signals"
    out = db[["patient_id", "age", "sex", "strat_fold"]].copy()
    out["age_clean"] = out["age"].clip(upper=90)  # ages >89 are coded ~300 in PTB-XL (HIPAA)
    out["superclasses"] = db["superclasses"].map("|".join)
    out.reset_index().to_csv(CACHE / "meta.csv", index=False)
    np.save(CACHE / "X.npy", X)
    np.save(CACHE / "Y.npy", Y)
    print("Class counts:", dict(zip(["NORM", "MI", "STTC", "CD", "HYP"], Y.sum(0).astype(int))))
    print("Saved to", CACHE)


if __name__ == "__main__":
    main()
