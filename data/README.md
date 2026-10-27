# Data

Raw data are **not** stored in this repository (they are large, and each dataset has its own licence).

## Main data: PTB-XL

Expected layout (only the 100 Hz version is needed):

```
data/ptbxl/
  ptbxl_database.csv
  scp_statements.csv
  records100/00000/00001_lr.dat and .hea ... (21,799 records)
```
Download PTB-XL v1.0.3 (CC BY 4.0): https://doi.org/10.13026/kfzx-aw45
Everything in `data/processed/` is created by `python -m src.prepare_data`.

## Second-dataset test data: Georgia 12-Lead ECG Challenge Database

Source: PhysioNet/Computing in Cardiology Challenge 2020, version 1.0.2, folder `training/georgia/`.
https://doi.org/10.13026/dvyd-kd57 (read the dataset page for its licence terms).

```powershell
python -m src.external_download
```

- Files go to `data/external/georgia/` (10,344 records, each a `.hea` and a `.mat` file).
- In my run, 10,334 of the 10,344 records downloaded; 10 did not, even after repeated attempts.
- Converted arrays go to `data/external/processed/` (created by `python -m src.external_prepare`).
- Both folders are ignored by git.
