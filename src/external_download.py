"""External validation, step 1: download the Georgia 12-lead ECG database (PhysioNet/CinC Challenge 2020, v1.0.2).
Files go to data/external/georgia/ (flat folder with matching .hea and .mat files).
Re-run at any time: files already downloaded are skipped.
Run: python -m src.external_download"""
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

from src.config import ROOT

BASE = "https://physionet.org/files/challenge-2020/1.0.2/training/georgia/"
OUT = ROOT / "data" / "external" / "georgia"
HREF = re.compile(r'href="([^"?#]+\.(?:hea|mat))"', re.IGNORECASE)


def fetch(url, binary=False, tries=5):
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            return data if binary else data.decode("utf-8", "ignore")
        except Exception:
            time.sleep(2)
    return None


def list_files(folder_url):
    html = fetch(folder_url)
    if html is None:
        return []
    return sorted({urljoin(folder_url, h) for h in HREF.findall(html)})


def record_urls(fetch_fn=None):
    """Build the file list from PhysioNet's own RECORDS file in each folder (g1 ... g11).
    Falls back to reading the folder's web listing if a RECORDS file cannot be fetched."""
    fetch_fn = fetch_fn or fetch
    urls = []
    for g in range(1, 12):
        folder = f"{BASE}g{g}/"
        txt = fetch_fn(folder + "RECORDS")
        ids = [ln.strip() for ln in txt.splitlines() if ln.strip()] if txt else []
        found = [f"{folder}{i}.{ext}" for i in ids for ext in ("hea", "mat")] if ids else list_files(folder)
        print(f"g{g}: {len(found) // 2} records listed")
        urls += found
    return urls


def get_one(url):
    dest = OUT / Path(url).name
    if dest.exists() and dest.stat().st_size > 0:
        return True
    data = fetch(url, binary=True)
    if not data:
        return False
    tmp = dest.with_name(dest.name + ".part")   # write-then-rename: an interrupted run never leaves a half file
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    urls = record_urls()
    if not urls:
        raise SystemExit("No files could be listed. Use the fallback in the guide (AWS CLI or browser download).")
    print(f"Downloading {len(urls)} files to {OUT} (expected 20,688: 10,344 records x 2 files) ...")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ok in ex.map(get_one, urls):
            done += 1
            failed += 0 if ok else 1
            if done % 1000 == 0:
                print(f"  {done}/{len(urls)}")
    if failed:
        print(f"WARNING: {failed} files failed. Just run the command again; finished files are skipped.")
    heas = len(list(OUT.glob("*.hea")))
    mats = len(list(OUT.glob("*.mat")))
    print(f"Done. .hea files: {heas}, .mat files: {mats} (expected 10,344 each). Re-run if lower.")


if __name__ == "__main__":
    main()
