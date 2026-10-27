"""SNOMED CT -> PTB-XL diagnostic-superclass mapping used for external validation on the
PhysioNet/Computing in Cardiology Challenge 2020 Georgia database.

Codes and names were copied from the official Challenge tables (dx_mapping_scored.csv and
dx_mapping_unscored.csv, github.com/physionetchallenges/evaluation-2020). Superclass assignments follow the
PTB-XL equivalents in scp_statements.csv (e.g. 1AVB, IVCD, CRBBB -> CD; LVH, RVH, LAO/LAE -> HYP;
NST_, NDT, ISC* -> STTC; IMI, AMI ... -> MI).

tier "direct": clear PTB-XL equivalent. tier "approx": mapped by clinical meaning (subtype not specified, or a
form-type finding). The strict sensitivity analysis uses "direct" codes only.

Prolonged QT (111975006) is EXCLUDED, not mapped. PTB-XL has an LNGQT statement (class STTC), but the code is far
more common in Georgia (1,391 of 10,344 records, 13%) than in PTB-XL (118 of 21,837, 0.5%), which suggests a
different labelling practice. Any record carrying it has an unknown STTC status, so the whole record is left out.

Strict mode keeps only records whose mapped codes are ALL "direct" equivalents.
"""
import numpy as np

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NSR = "426783006"          # sinus rhythm
EXCLUDE_IF_PRESENT = {"111975006"}   # prolonged QT: unknown STTC status, see docstring
MIN_POSITIVES = 50         # a class is evaluated only if it has at least this many positives and negatives

# code: (official Dx name, PTB-XL superclass, tier)
MAPPING = {
    "270492004": ("1st degree av block", "CD", "direct"),
    "195042002": ("2nd degree av block", "CD", "direct"),
    "713427006": ("complete right bundle branch block", "CD", "direct"),
    "713426002": ("incomplete right bundle branch block", "CD", "direct"),
    "445118002": ("left anterior fascicular block", "CD", "direct"),
    "445211001": ("left posterior fascicular block", "CD", "direct"),
    "164909002": ("left bundle branch block", "CD", "direct"),
    "251120003": ("incomplete left bundle branch block", "CD", "direct"),
    "698252002": ("nonspecific intraventricular conduction disorder", "CD", "direct"),
    "59118001": ("right bundle branch block", "CD", "approx"),
    "6374002": ("bundle branch block", "CD", "approx"),
    "164873001": ("left ventricular hypertrophy", "HYP", "direct"),
    "89792004": ("right ventricular hypertrophy", "HYP", "direct"),
    "446813000": ("left atrial hypertrophy", "HYP", "direct"),
    "253352002": ("left atrial abnormality", "HYP", "direct"),
    "446358003": ("right atrial hypertrophy", "HYP", "direct"),
    "253339007": ("right atrial abnormality", "HYP", "direct"),
    "266249003": ("ventricular hypertrophy", "HYP", "approx"),
    "195126007": ("atrial hypertrophy", "HYP", "approx"),
    "428750005": ("nonspecific st t abnormality", "STTC", "direct"),
    "164934002": ("t wave abnormal", "STTC", "direct"),
    "426434006": ("anterior ischemia", "STTC", "direct"),
    "164861001": ("myocardial ischemia", "STTC", "direct"),
    "413444003": ("acute myocardial ischemia", "STTC", "direct"),
    "413844008": ("chronic myocardial ischemia", "STTC", "direct"),
    "429622005": ("st depression", "STTC", "approx"),
    "164931005": ("st elevation", "STTC", "approx"),
    "164930006": ("st interval abnormal", "STTC", "approx"),
    "59931005": ("t wave inversion", "STTC", "approx"),
    "164865005": ("myocardial infarction", "MI", "direct"),
    "164867002": ("old myocardial infarction", "MI", "direct"),
    "54329005": ("anterior myocardial infarction", "MI", "direct"),
    "57054005": ("acute myocardial infarction", "MI", "direct"),
}


def label_record(codes, strict=False):
    """Return (included, y) for one record.

    y is a length-5 vector in CLASSES order. A record is included only if its status for the five classes is
    known: it has at least one mapped code, or its only code is sinus rhythm (treated as NORM).
    Excluded: records whose codes are all unmapped (e.g. atrial fibrillation only), records carrying a code in
    EXCLUDE_IF_PRESENT, and (strict mode only) records with any "approx" code.
    """
    codes = [c.strip() for c in codes if c.strip()]
    y = np.zeros(len(CLASSES), dtype=np.float32)
    if any(c in EXCLUDE_IF_PRESENT for c in codes):
        return False, y
    mapped_codes = [c for c in codes if c in MAPPING]
    if strict and any(MAPPING[c][2] == "approx" for c in mapped_codes):
        return False, y
    for sc in {MAPPING[c][1] for c in mapped_codes}:
        y[CLASSES.index(sc)] = 1.0
    if mapped_codes:
        return True, y
    if set(codes) == {NSR}:
        y[CLASSES.index("NORM")] = 1.0
        return True, y
    return False, y
