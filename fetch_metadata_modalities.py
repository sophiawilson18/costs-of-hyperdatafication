#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch all Hugging Face dataset IDs grouped by modality tag.

For each modality (text, image, audio, video, tabular, document, time-series, 3d, geospatial),
the script queries the Hugging Face Hub using the official search API and stores:
  - one text file per modality
  - one combined Parquet file with columns: id, modality

Usage:
    python fetch_all_modalities.py --out-dir modality --token $HF_TOKEN
"""

import os
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fetch Hugging Face datasets grouped by modality.")
    parser.add_argument("--out-dir", default="metadata", help="Output directory for lists and combined parquet")
    parser.add_argument("--token", default=None, help="Hugging Face token (or env HF_TOKEN)")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    modalities = [
        "text", "image", "audio", "video",
        "tabular", "document", "timeseries", "3d", "geospatial"
    ]

    api = HfApi()
    all_rows = []

    for mod in modalities:
        tag = f"modality:{mod}"
        print(f"[info] Fetching datasets for {tag} ...")
        try:
            results = list(api.list_datasets(filter=[tag], full=False, token=args.token))
            ids = [r.id for r in results]
            print(f"  → found {len(ids)} datasets")
        except Exception as e:
            print(f"  [error] {mod}: {e}")
            ids = []

        # save to text file
        #txt_path = OUT_DIR / f"ids_{mod}.txt"
        #with open(txt_path, "w", encoding="utf-8") as f:
        #    for dsid in ids:
        #        f.write(dsid + "\n")

        # add to combined table
        all_rows.extend([{"id": dsid, "modality": mod} for dsid in ids])

    # save combined parquet
    if all_rows:
        df = pd.DataFrame(all_rows)
        df = (
            df.groupby("id")["modality"]
            .apply(lambda s: ";".join(sorted(set(s))))
            .reset_index(name="modalities_final")
        )

        out_parquet = OUT_DIR / "metadata_modalities.parquet"
        df.to_parquet(out_parquet, index=False)

        # --- summary ---
        print(f"\n[✓] Saved combined {len(df)} datasets → {out_parquet}\n")
        print("Modality combinations:\n")
        counts = df["modalities_final"].value_counts()
        print(counts.to_string())

    else:
        print("[warn] No datasets retrieved for any modality.")

if __name__ == "__main__":
    main()