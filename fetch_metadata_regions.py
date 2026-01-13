#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch all Hugging Face dataset IDs and their region tags.

For each dataset:
  - reads its tags
  - extracts all tags of the form "region:XXX"
  - stores a combined Parquet file with columns:
        id, regions_final

Usage:
    python fetch_all_regions.py --out-dir metadata --token $HF_TOKEN
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi


def extract_regions_from_tags(tags):
    """Return a sorted list of region codes from a list of tags."""
    if not tags:
        return []
    regs = []
    for t in tags:
        if t.startswith("region:"):
            val = t.split("region:", 1)[1]
            if val:
                regs.append(val)
    return sorted(set(regs))


def main():
    parser = argparse.ArgumentParser(description="Fetch Hugging Face datasets and their region tags.")
    parser.add_argument("--out-dir", default="metadata",
                        help="Output directory for Parquet and optional txt files")
    parser.add_argument("--token", default=None, help="Hugging Face token (or env HF_TOKEN)")
    parser.add_argument("--write-per-region", action="store_true",
                        help="Also write one txt file per region")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    print("[info] Fetching all datasets with full metadata ...")
    results = list(api.list_datasets(full=True, token=args.token, limit=None))
    print(f"[info] Retrieved {len(results)} datasets")

    all_rows = []
    region_to_ids = defaultdict(list)

    for r in results:
        regs = extract_regions_from_tags(getattr(r, "tags", None))
        if not regs:
            continue

        all_rows.append({
            "id": r.id,
            "regions_final": ";".join(regs),
        })

        for reg in regs:
            region_to_ids[reg].append(r.id)

    if not all_rows:
        print("[warn] No datasets with region tags found.")
        return

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"])
    out_parquet = out_dir / "metadata_regions.parquet"
    df.to_parquet(out_parquet, index=False)

    print(f"\n[✓] Saved combined {len(df)} datasets → {out_parquet}\n")

    # summary
    print("Top regions (by number of datasets):\n")
    region_counts = (
        df["regions_final"]
        .str.split(";", expand=True)
        .stack()
        .value_counts()
    )
    print(f"\nTotal number of unique regions: {len(region_counts)}")
    print(region_counts.head(50).to_string())

    # optional per-region txts
    if args.write_per_region:
        print("\n[info] Writing per-region id lists ...")
        for reg, ids in region_to_ids.items():
            txt_path = out_dir / f"ids_region_{reg}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for dsid in sorted(set(ids)):
                    f.write(dsid + "\n")
        print(f"[✓] Wrote {len(region_to_ids)} region files in {out_dir}")


if __name__ == "__main__":
    main()