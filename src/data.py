"""Label handling for PTB-XL (diagnostic superclasses) and cached-array loading."""
import ast

import numpy as np
import pandas as pd

from src.config import CACHE, DATA_DIR, SUPERCLASSES


def load_metadata(data_dir=DATA_DIR):
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]  # diagnostic statements only
    return db, scp


def add_superclass(db, scp):
    """Map each record's SCP codes to diagnostic superclasses (NORM, MI, STTC, CD, HYP).
    Follows the aggregation used in PTB-XL's official example_physionet.py."""
    def agg(codes):
        out = set()
        for code in codes:
            if code in scp.index:
                cls = scp.loc[code, "diagnostic_class"]
                if isinstance(cls, str):
                    out.add(cls)
        return sorted(out)

    db = db.copy()
    db["superclasses"] = db["scp_codes"].apply(agg)
    return db


def labels_matrix(db):
    Y = np.zeros((len(db), len(SUPERCLASSES)), dtype=np.float32)
    for i, classes in enumerate(db["superclasses"]):
        for c in classes:
            Y[i, SUPERCLASSES.index(c)] = 1.0
    return Y


def load_processed():
    X = np.load(CACHE / "X.npy")
    Y = np.load(CACHE / "Y.npy")
    meta = pd.read_csv(CACHE / "meta.csv")
    return X, Y, meta
