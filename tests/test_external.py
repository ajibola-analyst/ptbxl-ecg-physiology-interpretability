import numpy as np

from src.external_download import record_urls
from src.external_mapping import CLASSES, MAPPING, NSR, label_record
from src.external_prepare import convert, parse_header
from src.external_readme import inject, key_line, results_section
from src.external_stats import bootstrap_ci, build_labels, evaluable_classes, evaluate_probs

LEADS12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
HEADER = "A0001 12 500 5000 05-Feb-2020 11:39:16\n" + "".join(
    f"A0001.mat 16+24 1000/mV 16 0 28 -1716 0 {n}\n" for n in LEADS12) + \
    "#Age: 74\n#Sex: Male\n#Dx: 164889003,59118001\n#Rx: Unknown\n"


def test_header_parsing():
    h = parse_header(HEADER)
    assert h["fs"] == 500 and h["n_samples"] == 5000 and h["n_leads"] == 12
    assert h["gains"].tolist() == [1000.0] * 12 and h["leads"][3] == "aVR"
    assert h["dx"] == ["164889003", "59118001"]


def test_convert_units_order_and_length():
    h = parse_header(HEADER)
    val = np.stack([np.full(5000, (k + 1) * 1000) for k in range(12)])   # lead k = (k+1) mV
    x, reason = convert(val, h)
    assert reason == "ok" and x.shape == (1000, 12)
    assert np.allclose(x[500], np.arange(1, 13), atol=1e-3)
    h2 = dict(h)
    h2["leads"] = list(reversed(h["leads"]))       # header lists leads in reverse: must be re-ordered
    x2, _ = convert(val, h2)
    assert np.allclose(x2[500], np.arange(12, 0, -1), atol=1e-3)


def test_baseline_offset_is_removed():
    h = parse_header(HEADER)
    h["baselines"] = np.full(12, 200.0)
    x, _ = convert(np.full((12, 5000), 1200), h)
    assert np.allclose(x[500], 1.0, atol=1e-3)


def test_short_recording_is_skipped():
    x, reason = convert(np.zeros((12, 2000)), parse_header(HEADER))
    assert x is None and reason == "shorter_than_10s"


def test_label_logic():
    ok, y = label_record([NSR])
    assert ok and y.tolist() == [1, 0, 0, 0, 0]
    ok, y = label_record([NSR, "164909002"])
    assert ok and y[CLASSES.index("CD")] == 1 and y[0] == 0
    assert not label_record(["164889003"])[0]                        # atrial fibrillation only
    ok, y = label_record(["164873001", "428750005"])
    assert ok and y[CLASSES.index("HYP")] == 1 and y[CLASSES.index("STTC")] == 1
    assert not label_record([NSR, "426177001"])[0]                   # NSR + sinus bradycardia: NORM status unclear
    assert not label_record(["111975006"])[0]                        # prolonged QT alone
    assert not label_record(["164909002", "111975006"])[0]           # prolonged QT anywhere -> excluded
    assert label_record(["429622005"])[0]                            # approx code allowed in normal mode
    assert not label_record(["429622005"], strict=True)[0]
    assert not label_record(["164909002", "429622005"], strict=True)[0]
    assert label_record(["164909002", "428750005"], strict=True)[0]


def test_mapping_is_well_formed():
    assert all(v[1] in CLASSES and v[2] in ("direct", "approx") for v in MAPPING.values())
    assert NSR not in MAPPING and "111975006" not in MAPPING


def test_build_labels_strict_is_subset():
    codes = [[NSR], ["164909002"], ["429622005"], ["164889003"], ["111975006"]]
    inc, _ = build_labels(codes)
    inc_s, _ = build_labels(codes, strict=True)
    assert inc.tolist() == [True, True, True, False, False]
    assert inc_s.tolist() == [True, True, False, False, False]


def test_evaluation_helpers():
    rng = np.random.default_rng(0)
    Y = (rng.random((600, 5)) > 0.7).astype(np.float32)
    Y[:, 1] = 0
    Y[:5, 1] = 1
    assert 1 not in evaluable_classes(Y)
    P = np.clip(Y * 0.5 + rng.random((600, 5)) * 0.5, 0, 1)
    cols = evaluable_classes(Y)
    r = evaluate_probs(Y, P, cols, n_boot=50)
    assert r["macro_auroc"] > 0.8 and r["macro_ci95"][0] <= r["macro_ci95"][1]
    assert bootstrap_ci(Y[:, cols], P[:, cols], n=20)[0][0] <= bootstrap_ci(Y[:, cols], P[:, cols], n=20)[0][1]


def _fake_results():
    def var(ext_bias):
        cl = ["NORM", "STTC", "CD", "HYP"]
        v = {"evaluated_classes": cl}
        for m, base in (("1D ResNet", 0.93), ("Classical", 0.88)):
            per = {c: base - ext_bias * (i + 1) / 10 for i, c in enumerate(cl)}
            v[m] = {"external": {"per_class_auroc": per, "per_class_ci95": {c: [per[c] - .02, per[c] + .02] for c in cl},
                                 "macro_auroc": float(np.mean(list(per.values()))), "macro_ci95": [.8, .9], "ece": .03},
                    "ptbxl_fold10": {"per_class_auroc": {c: base for c in cl}, "macro_auroc": base, "ece": .02}}
        return v
    return {"n_included": 5000, "n_included_strict": 4000, "not_evaluated": ["MI"], "min_positives_rule": 50,
            "positives_per_class_main": {"MI": 7}, "main": var(0.1), "strict_sensitivity": var(0.05)}


def test_readme_generation_and_marker_injection():
    res = _fake_results()
    sec, key = results_section(res), key_line(res)
    assert "MI could not be tested" in sec and "Only 7 Georgia records" in sec
    assert "| **Average** |" in sec and "four classes" in key
    text = "a\n<!-- EXTERNAL_KEY_START -->\nold\n<!-- EXTERNAL_KEY_END -->\nb"
    new, ok = inject(text, "EXTERNAL_KEY", key)
    assert ok and "old" not in new and new.startswith("a\n<!-- EXTERNAL_KEY_START -->")
    assert inject("no markers", "EXTERNAL_KEY", key)[1] is False


def test_record_urls_built_from_records_files():
    fake = {f"https://physionet.org/files/challenge-2020/1.0.2/training/georgia/g{g}/RECORDS": "E%05d\nE%05d\n" % (g * 10, g * 10 + 1)
            for g in range(1, 12)}
    urls = record_urls(fetch_fn=lambda u: fake.get(u))
    assert len(urls) == 11 * 2 * 2
    assert urls[0].endswith("/georgia/g1/E00010.hea") and urls[1].endswith("/georgia/g1/E00010.mat")


REAL_HEADER = "E00001 12 500 5000\n" + "".join(
    f"E00001.mat 16x1+24 1000.0(0)/mV 16 0 {v} 0 0 {n}\n"
    for v, n in zip([136, 87, -48, -112, 92, 19, -39, 58, 87, 97, 87, 78], LEADS12)
) + "# Age: NaN\n# Sex: Female\n# Dx: 426783006\n# Rx: Unknown\n# Hx: Unknown\n"


def test_real_georgia_header_format():
    """Exact layout of the downloaded Georgia files: gain written as 1000.0(0)/mV, and '# Dx:' with a space."""
    h = parse_header(REAL_HEADER)
    assert h["fs"] == 500 and h["n_samples"] == 5000 and h["n_leads"] == 12
    assert h["gains"].tolist() == [1000.0] * 12
    assert h["leads"] == LEADS12
    assert h["dx"] == ["426783006"]
    x, reason = convert(np.full((12, 5000), 1000), h)
    assert reason == "ok" and np.allclose(x[500], 1.0, atol=1e-3)
