#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch all Hugging Face dataset IDs grouped by task category AND attach a high-level category label.

Outputs:
  - One text file per task (ids_<task>.txt)
  - One combined Parquet with columns:
      id, tasks_final, categories_final

Usage:
    python fetch_all_tasks.py --out-dir tasks --token $HF_TOKEN
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi

# ----------------------------- Task catalogue -----------------------------
# Keep tasks exactly as used by the Hub (Datasets Explorer "task_categories" filter)
CATEGORIES = {
    "MM": [
        "image-text-to-text",
        "video-text-to-text",
        "any-to-any",
        "visual-question-answering",
        "visual-document-retrieval",
    ],
    "CV": [
        "depth-estimation",
        "image-classification",
        "object-detection",
        "image-segmentation",
        "text-to-image",
        "image-to-text",
        "image-to-image",
        "image-to-video",
        "unconditional-image-generation",
        "video-classification",
        "text-to-video",
        "zero-shot-image-classification",
        "mask-generation",
        "zero-shot-object-detection",
        "text-to-3d",
        "image-to-3d",
        "image-feature-extraction",
    ],
    "NLP": [
        "text-classification",
        "token-classification",
        "table-question-answering",
        "question-answering",
        "zero-shot-classification",
        "translation",
        "summarization",
        "feature-extraction",
        "text-generation",
        "fill-mask",
        "sentence-similarity",
        "table-to-text",
        "multiple-choice",
        "text-ranking",
        "text-retrieval",
    ],
    "AS": [
        "text-to-speech",
        "text-to-audio",
        "automatic-speech-recognition",
        "audio-to-audio",
        "audio-classification",
        "voice-activity-detection",
    ],
    "TAB": [
        "tabular-classification",
        "tabular-regression",
        "tabular-to-text",
        "time-series-forecasting",
    ],
    "RL": [
        "reinforcement-learning",
        "robotics",
    ],
    "Other": [
        "graph-machine-learning",
    ],
}

# Reverse map task -> category (first hit wins; keep it simple)
TASK_TO_CATEGORY = {
    task: category
    for category, tasks in CATEGORIES.items()
    for task in tasks
}

def main():
    parser = argparse.ArgumentParser(description="Fetch Hugging Face datasets grouped by task, with category column.")
    parser.add_argument("--out-dir", default="metadata", help="Output directory for lists and combined parquet.")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (or set HF_TOKEN env var).")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    all_rows = []  # long-form rows: {id, task, category}

    # Iterate tasks by category so we can attach the category immediately
    for category, tasks in CATEGORIES.items():
        print(f"\n=== {category} ===")
        for task in tasks:
            tag = f"task_categories:{task}"
            print(f"[info] Fetching datasets for {tag} ...")
            try:
                results = list(api.list_datasets(filter=[tag], full=False, token=args.token))
                ids = [r.id for r in results]
                print(f"  → found {len(ids)} datasets")
            except Exception as e:
                print(f"  [error] {task}: {e}")
                ids = []

            # per-task text file
            #txt_path = OUT_DIR / f"ids_{task.replace('/', '_')}.txt"
            #with open(txt_path, "w", encoding="utf-8") as f:
            #    for dsid in ids:
            #        f.write(dsid + "\n")

            # add to long-form table
            all_rows.extend({"id": dsid, "task": task, "category": category} for dsid in ids)

    if not all_rows:
        print("\n[warn] No datasets retrieved for any task category.")
        return

    # Build combined wide table: id -> semicolon-joined tasks & categories
    df_long = pd.DataFrame(all_rows)
    # Aggregate tasks
    tasks_wide = (
        df_long.groupby("id")["task"]
        .apply(lambda s: ";".join(sorted(set(s))))
        .rename("tasks_final")
    )
    # Aggregate categories
    cats_wide = (
        df_long.groupby("id")["category"]
        .apply(lambda s: ";".join(sorted(set(s))))
        .rename("categories_final")
    )
    df_wide = pd.concat([tasks_wide, cats_wide], axis=1).reset_index()

    out_parquet = OUT_DIR / "metadata_tasks.parquet"
    df_wide.to_parquet(out_parquet, index=False)

    print(f"\n[✓] Saved combined {len(df_wide)} datasets → {out_parquet}")

    # --- summaries ---
    print("\nTask combination counts (top 20):")
    print(df_wide["tasks_final"].value_counts().head(20).to_string())

    print("\nCategory coverage:")
    # Count datasets per category (multi-category datasets counted in each)
    cat_counts = (
        df_long[["id", "category"]].drop_duplicates()
        .groupby("category")["id"].nunique()
        .sort_values(ascending=False)
    )
    print(cat_counts.to_string())

if __name__ == "__main__":
    main()