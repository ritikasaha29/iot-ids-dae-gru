"""
download_data.py — Dataset acquisition helper
===============================================
Prints download links and organises data directories.
Actual downloads require manual steps (datasets require registration).

Supported:
  1. NSL-KDD    — General anomaly detection baseline
  2. N-BaIoT    — Real IoT botnet traffic (Mirai, BASHLITE)
  3. Kitsune     — Online network intrusion (ARP, fuzzing, etc.)
"""

import os
from pathlib import Path

DATASETS = {
    "NSL-KDD": {
        "url": "https://www.unb.ca/cic/datasets/nsl.html",
        "direct": "https://github.com/defcom17/NSL_KDD/archive/refs/heads/master.zip",
        "files": ["KDDTrain+.txt", "KDDTest+.txt"],
        "dir": "data/NSL-KDD/",
        "notes": "No registration required. Direct GitHub download available."
    },
    "N-BaIoT": {
        "url": "https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot",
        "direct": "https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip",
        "files": ["*.csv (one per device/attack combination)"],
        "dir": "data/N-BaIoT/",
        "notes": "Free UCI download. ~1.3GB. Organise into benign/ and attack/ subdirs."
    },
    "Kitsune": {
        "url": "https://github.com/ymirsky/Kitsune-py",
        "direct": "https://zenodo.org/record/2600520",
        "files": ["*_dataset.csv", "*_labels.csv"],
        "dir": "data/Kitsune/",
        "notes": "Multiple attack scenarios available on Zenodo."
    }
}

def print_instructions():
    print("=" * 65)
    print("  DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    for name, info in DATASETS.items():
        Path(info["dir"]).mkdir(parents=True, exist_ok=True)
        print(f"\n  {name}")
        print(f"  {'─' * (len(name) + 2)}")
        print(f"  Info page : {info['url']}")
        print(f"  Direct URL: {info['direct']}")
        print(f"  Save to   : {info['dir']}")
        print(f"  Files     : {', '.join(info['files'])}")
        print(f"  Notes     : {info['notes']}")

    print("\n" + "=" * 65)
    print("  QUICK SETUP (after downloading):")
    print("=" * 65)
    print("""
  NSL-KDD (recommended for quick start):
    mkdir -p data/NSL-KDD
    # Download KDDTrain+.txt and KDDTest+.txt
    # Place them in data/NSL-KDD/
    python train.py --data_dir data/NSL-KDD/

  N-BaIoT:
    mkdir -p data/N-BaIoT/benign data/N-BaIoT/mirai data/N-BaIoT/bashlite
    # Extract and organise CSVs into subfolders by attack type
    python train.py --data_dir data/N-BaIoT/ --top_k 25

  Kitsune:
    mkdir -p data/Kitsune
    # Download attack scenario CSVs + label files
    python train.py --data_dir data/Kitsune/ --seq_len 15
""")

def try_auto_download():
    """Attempt to download NSL-KDD (no auth required) via requests."""
    try:
        import urllib.request
        import zipfile
        out_dir = Path("data/NSL-KDD")
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "nslkdd.zip"
        print("[Auto-download] Fetching NSL-KDD from GitHub...")
        url = "https://github.com/defcom17/NSL_KDD/archive/refs/heads/master.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        # Move txt files to root of data/NSL-KDD/
        for f in out_dir.rglob("*.txt"):
            f.rename(out_dir / f.name)
        zip_path.unlink()
        print(f"  Saved to {out_dir}/")
        return True
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        return False

if __name__ == "__main__":
    print_instructions()
    import sys
    if "--auto" in sys.argv:
        try_auto_download()
