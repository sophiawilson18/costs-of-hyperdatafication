#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch all Hugging Face dataset IDs and their language tags.

For each dataset, the script:
  - reads its tags
  - extracts all tags of the form "language:XXX"
  - stores a combined Parquet file with columns:
        id, languages_final

Usage:
    python fetch_all_languages.py --out-dir metadata --token $HF_TOKEN
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi


def extract_languages_from_tags(tags):
    """Return a sorted list of language codes from a list of tags."""
    if not tags:
        return []
    langs = []
    for t in tags:
        if t.startswith("language:"):
            val = t.split("language:", 1)[1]
            if val:
                langs.append(val)
    return sorted(set(langs))


def main():
    parser = argparse.ArgumentParser(description="Fetch Hugging Face datasets and their language tags.")
    parser.add_argument("--out-dir", default="metadata", help="Output directory for Parquet and optional txt files")
    parser.add_argument("--token", default=None, help="Hugging Face token (or env HF_TOKEN)")
    parser.add_argument("--write-per-language", action="store_true",
                        help="Also write one txt file per language (may create many files)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    print("[info] Fetching all datasets with full metadata ...")
    # limit=None → fetch all datasets
    results = list(api.list_datasets(full=True, token=args.token, limit=None))
    print(f"[info] Retrieved {len(results)} datasets")

    all_rows = []
    lang_to_ids = defaultdict(list)

    for r in results:
        langs = extract_languages_from_tags(getattr(r, "tags", None))
        if not langs:
            continue

        all_rows.append({
            "id": r.id,
            "languages_final": ";".join(langs),
        })

        for lang in langs:
            lang_to_ids[lang].append(r.id)

    if not all_rows:
        print("[warn] No datasets with language tags found.")
        return

    # Combined Parquet (one row per dataset with all its languages)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"])
    out_parquet = out_dir / "metadata_languages.parquet"
    df.to_parquet(out_parquet, index=False)

    print(f"\n[✓] Saved combined {len(df)} datasets → {out_parquet}\n")

    # Simple summary of language counts
    print("Top languages (by number of datasets):\n")
    lang_counts = (
        df["languages_final"]
        .str.split(";", expand=True)
        .stack()
        .value_counts()
    )
    print(f"\nTotal number of unique languages: {len(lang_counts)}")
    print(lang_counts.head(50).to_string())
    

    # Optional: one txt file per language (can be 8k+ files)
    if args.write_per_language:
        print("\n[info] Writing per-language id lists ...")
        for lang, ids in lang_to_ids.items():
            txt_path = out_dir / f"ids_language_{lang}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for dsid in sorted(set(ids)):
                    f.write(dsid + "\n")
        print(f"[✓] Wrote {len(lang_to_ids)} language files in {out_dir}")


if __name__ == "__main__":
    main()