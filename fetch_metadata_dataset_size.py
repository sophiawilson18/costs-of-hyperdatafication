#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch dataset size for Hugging Face datasets from an ID list.

- Uses datasets-server /info endpoint
- Aggregates dataset_size across configs
- Parallel, resumable, checkpointed
"""

import os, re, json, time, argparse, random, socket, getpass, datetime as dt
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="Fetch HF dataset sizes via datasets-server.")
p.add_argument("--ids-file", required=True, help="TXT file with one dataset ID per line.")
p.add_argument("--out", default="dataset_sizes.parquet", help="Merged output parquet.")
p.add_argument("--parts-dir", default="parts", help="Checkpoint directory.")
p.add_argument("--part-prefix", default=None, help="Prefix for part files.")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--sleep", type=float, default=0.25)
p.add_argument("--batch-size", type=int, default=2000)
p.add_argument("--token", default=os.getenv("HF_TOKEN"))
p.add_argument("--ua", default="Datalifecycle/0.1")
args = p.parse_args()

IDS_FILE   = Path(args.ids_file)
OUT_PATH  = Path(args.out)
PARTS_DIR = Path(args.parts_dir)
PARTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PART_PREFIX = args.part_prefix or f"{getpass.getuser()}@{socket.gethostname()}"
INFO_API = "https://datasets-server.huggingface.co/info?dataset="

UA = {"User-Agent": args.ua}
if args.token:
    UA["Authorization"] = f"Bearer {args.token}"

http = requests.Session()

# ---------------- Helpers ----------------
def polite_sleep():
    if args.sleep > 0:
        time.sleep(args.sleep * (0.9 + 0.4 * random.random()))

def read_ids(p):
    return [l.strip() for l in open(p) if l.strip()]

def list_done_ids(parts_dir, final_path):
    done = set()
    if final_path.exists():
        try:
            done |= set(pd.read_parquet(final_path, columns=["id"])["id"])
        except Exception:
            pass
    for p in parts_dir.glob("*.parquet"):
        try:
            done |= set(pd.read_parquet(p, columns=["id"])["id"])
        except Exception:
            pass
    return done

def next_part_index():
    idxs = []
    for p in PARTS_DIR.glob(f"{PART_PREFIX}_part_*.parquet"):
        m = re.search(r"_part_(\d+)\.parquet$", p.name)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) + 1 if idxs else 1

def write_part(rows, idx):
    out = PARTS_DIR / f"{PART_PREFIX}_part_{idx:06d}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[checkpoint] wrote {len(rows)} rows -> {out}")

def merge_parts():
    dfs = []
    for p in PARTS_DIR.glob("*.parquet"):
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            pass
    if not dfs:
        return
    df = (
        pd.concat(dfs, ignore_index=True)
          .sort_values("id")
          .drop_duplicates("id", keep="last")
    )
    df.to_parquet(OUT_PATH, index=False)
    print(f"[merge] wrote {OUT_PATH} (rows={len(df)})")

# ---------------- Fetch ----------------
def fetch_size(repo_id):
    url = INFO_API + quote(repo_id, safe="")
    backoff = 1.0
    for _ in range(5):
        try:
            r = http.get(url, headers=UA, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            r.raise_for_status()
            data = r.json().get("dataset_info", {})
            polite_sleep()

            # Aggregate dataset_size across configs
            if "dataset_size" in data:
                size = data.get("dataset_size", None)
            else:
                size = sum(
                    cfg.get("dataset_size", 0) or 0
                    for cfg in data.values()
                    if isinstance(cfg, dict)
                )

            return {
                "id": repo_id,
                "dataset_size_bytes": int(size) if size else None,
                "status": "ok",
                "fetch_timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
            }

        except Exception as e:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    return {
        "id": repo_id,
        "dataset_size_bytes": None,
        "status": "error",
        "fetch_timestamp": dt.datetime.now(dt.timezone.utc).isoformat()
    }

# ---------------- Main ----------------
ids_all = read_ids(IDS_FILE)
done = list_done_ids(PARTS_DIR, OUT_PATH)
todo = [i for i in ids_all if i not in done]

print(f"[ids] total={len(ids_all)} done={len(done)} todo={len(todo)}")

rows = []
part_idx = next_part_index()

with ThreadPoolExecutor(max_workers=args.threads) as ex:
    futures = {ex.submit(fetch_size, i): i for i in todo}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        rows.append(fut.result())
        if len(rows) >= args.batch_size:
            write_part(rows, part_idx)
            part_idx += 1
            rows = []

if rows:
    write_part(rows, part_idx)

merge_parts()
print("[done]")