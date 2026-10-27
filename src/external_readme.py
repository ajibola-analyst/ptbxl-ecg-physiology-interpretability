"""OPTIONAL helper (not part of run_external.ps1): drafts the second-dataset README section from the result files.
The published README section was written by hand from these same results, so do not run this on the final README.
Fills the text between the two marker pairs in README.md:
  <!-- EXTERNAL_KEY_START --> ... <!-- EXTERNAL_KEY_END -->
  <!-- EXTERNAL_RESULTS_START --> ... <!-- EXTERNAL_RESULTS_END -->
Run: python -m src.external_readme"""
import json
import re

from src.config import RESULTS, ROOT

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def inject(text, name, content):
    pat = re.compile(rf"(<!-- {name}_START -->)(.*?)(<!-- {name}_END -->)", re.DOTALL)
    if not pat.search(text):
        return text, False
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text, count=1), True


def _pm(v):
    return f"{v:+.3f}"


def compact_table(variant, model):
    e, r = variant[model]["external"], variant[model]["ptbxl_fold10"]
    rows = ["| Class | PTB-XL fold 10 | Georgia (95% CI) | Change |", "|---|---|---|---|"]
    for c in variant["evaluated_classes"]:
        a, ci, b = e["per_class_auroc"][c], e["per_class_ci95"][c], r["per_class_auroc"][c]
        rows.append(f"| {c} | {b:.3f} | {a:.3f} ({ci[0]:.3f} to {ci[1]:.3f}) | {_pm(a - b)} |")
    a, ci, b = e["macro_auroc"], e["macro_ci95"], r["macro_auroc"]
    rows.append(f"| **Average** | **{b:.3f}** | **{a:.3f}** ({ci[0]:.3f} to {ci[1]:.3f}) | **{_pm(a - b)}** |")
    return "\n".join(rows)


def deltas(variant, model):
    e, r = variant[model]["external"], variant[model]["ptbxl_fold10"]
    return {c: e["per_class_auroc"][c] - r["per_class_auroc"][c] for c in variant["evaluated_classes"]}


def key_line(res):
    m = res["main"]
    k = WORDS.get(len(m["evaluated_classes"]), str(len(m["evaluated_classes"])))
    a, b = m["1D ResNet"]["external"]["macro_auroc"], m["1D ResNet"]["ptbxl_fold10"]["macro_auroc"]
    c, d = m["Classical"]["external"]["macro_auroc"], m["Classical"]["ptbxl_fold10"]["macro_auroc"]
    s = (f"**Second-dataset test:** on ECGs from a US collection (Georgia), the ResNet scored {a:.3f} on the {k} classes "
         f"that could be tested, against {b:.3f} on the same classes in PTB-XL (change {_pm(a - b)}). "
         f"The simple method changed by {_pm(c - d)}.")
    if res["not_evaluated"]:
        s += f" {', '.join(res['not_evaluated'])} could not be tested there."
    return s


def results_section(res):
    m, st = res["main"], res["strict_sensitivity"]
    lines = ["### 6. Does it still work in another dataset?", "",
             f"Both models were built only on PTB-XL. I then tested them, unchanged, on {res['n_included']:,} ECGs "
             "from the Georgia database (a collection from the Southeastern United States). Nothing was retrained or tuned on it.", ""]
    if res["not_evaluated"]:
        for c in res["not_evaluated"]:
            lines.append(f"**{c} could not be tested.** Only {res['positives_per_class_main'][c]} Georgia records carry a "
                         f"matching label (at least {res['min_positives_rule']} are needed). "
                         f"So the {c} result stays untested outside PTB-XL.")
        lines.append("")
    lines += ["**1D ResNet**", "", compact_table(m, "1D ResNet"), "", "**Simple method**", "",
              compact_table(m, "Classical"), "", "![External validation](figures/external_validation.png)", "",
              "**What this shows**", ""]
    dr = deltas(m, "1D ResNet")
    er, ec = m["1D ResNet"]["external"], m["Classical"]["external"]
    worst, best = min(dr, key=dr.get), min(dr, key=lambda c: abs(dr[c]))
    lines.append(f"- The ResNet's average score changed from {m['1D ResNet']['ptbxl_fold10']['macro_auroc']:.3f} to "
                 f"{er['macro_auroc']:.3f} ({_pm(er['macro_auroc'] - m['1D ResNet']['ptbxl_fold10']['macro_auroc'])}).")
    lines.append(f"- The simple method changed from {m['Classical']['ptbxl_fold10']['macro_auroc']:.3f} to "
                 f"{ec['macro_auroc']:.3f} ({_pm(ec['macro_auroc'] - m['Classical']['ptbxl_fold10']['macro_auroc'])}).")
    lines.append(f"- On Georgia, the ResNet's average score was "
                 f"{'higher' if er['macro_auroc'] > ec['macro_auroc'] else 'not higher'} than the simple method's "
                 f"({er['macro_auroc']:.3f} vs {ec['macro_auroc']:.3f}).")
    lines.append(f"- The ResNet changed most for {worst} ({_pm(dr[worst])}) and least for {best} ({_pm(dr[best])}).")
    lines.append(f"- Calibration error (ECE) of the ResNet went from {m['1D ResNet']['ptbxl_fold10']['ece']:.3f} to "
                 f"{er['ece']:.3f}. No recalibration was done.")
    sr = st["1D ResNet"]
    lines += ["", "**Check with a stricter label set.** Using only records whose labels have a direct PTB-XL equivalent "
              f"({res['n_included_strict']:,} records), the ResNet's average score was {sr['external']['macro_auroc']:.3f} "
              f"(change {_pm(sr['external']['macro_auroc'] - sr['ptbxl_fold10']['macro_auroc'])}).", "",
              "**Reminder.** The two datasets label ECGs differently, so part of any change comes from labelling, "
              "not only from the model. The 95% ranges here come from resampling records, not patients."]
    return "\n".join(lines)


def main():
    res = json.loads((RESULTS / "external_georgia.json").read_text())
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text, ok1 = inject(text, "EXTERNAL_KEY", key_line(res))
    text, ok2 = inject(text, "EXTERNAL_RESULTS", results_section(res))
    if not (ok1 and ok2):
        raise SystemExit("Markers not found in README.md. Use the new README.md from the guide (it contains them).")
    path.write_text(text, encoding="utf-8")
    print("README.md updated. Open it and read the new section before you commit.")


if __name__ == "__main__":
    main()
