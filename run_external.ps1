# Second-dataset test (Georgia). Run from the repo root with the virtual environment activated,
# AFTER run_all.ps1 has finished. Needs about 1.3 GB of extra disk space (plus 0.5 GB for converted arrays).
# Safe to re-run: finished downloads are skipped.
$ErrorActionPreference = "Stop"
python -m pytest -q tests
python -m src.external_download
python -m src.external_prepare
python -m src.external_evaluate
python -m src.make_figures
