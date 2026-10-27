# Runs the main pipeline (PowerShell). Run from the repo root with the virtual environment activated.
$ErrorActionPreference = "Stop"
python -m pytest -q tests
python -m src.prepare_data
python -m src.train_classical
python -m src.train_resnet --epochs 25
python -m src.evaluate
python -m src.robustness
python -m src.gradcam
python -m src.make_demo_samples
python -m src.make_figures
