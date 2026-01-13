#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REST-only fetch of (createdAt, lastModified, downloadsAllTime, downloads) for dataset IDs.

- Uses https://huggingface.co/api/datasets/<id> (single REST call per ID)
- Parallel threads, jittered delay, retries & backoff
- Resumable: skips IDs already present in parts/ or --out
- Safe parallel runs with different --part-prefix (or auto user@host)
"""

import os, re, json, time, argparse, random, socket, getpass, datetime as dt
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="Fetch createdAt/lastModified/downloads(/AllTime) via REST for ID list.")
p.add_argument("--ids-file", required=True, help="TXT with one dataset ID per line.")
p.add_argument("--parts-dir", default="parts", help="Directory for checkpoint parts.")
p.add_argument("--part-prefix", default=None, help="Filename prefix for parts (default: user@host).")
p.add_argument("--out", default="core_rest.parquet", help="Merged parquet (rewritten during merges).")
p.add_argument("--threads", type=int, default=4, help="Worker threads.")
p.add_argument("--sleep", type=float, default=0.25, help="Polite per-request delay (seconds, jittered).")
p.add_argument("--batch-size", type=int, default=2000, help="Rows per checkpoint write.")
p.add_argument("--merge-every-parts", type=int, default=5, help="Merge final after every N parts.")
p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (or env HF_TOKEN).")
p.add_argument("--ua", default="Datalifecycle/0.1 (contact: you@example.com)", help="Custom User-Agent.")
args = p.parse_args()

IDS_FILE   = Path(args.ids_file)
PARTS_DIR  = Path(args.parts_dir); PARTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = Path(args.out);       OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
PART_PREFIX = args.part_prefix or f"{getpass.getuser()}@{socket.gethostname()}"

THREADS = max(1, args.threads)
DELAY   = max(0.0, args.sleep)
BATCH   = max(1, args.batch_size)
HF_TOKEN = args.token
UA = {"User-Agent": args.ua}
http = requests.Session()

# ---------------- Helpers ----------------
def polite_sleep():
    if DELAY > 0:
        time.sleep(DELAY * (0.9 + 0.4 * random.random()))

def read_ids(pth: Path) -> list[str]:
    with open(pth, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def list_existing_ids(parts_dir: Path, final_path: Path) -> set[str]:
    done = set()
    if final_path.exists():
        try:
            df = pd.read_parquet(final_path, columns=["id"])
            done |= set(df["id"].astype(str))
        except Exception:
            pass
    for pth in parts_dir.glob("*.parquet"):
        try:
            dfp = pd.read_parquet(pth, columns=["id"])
            done |= set(dfp["id"].astype(str))
        except Exception:
            continue
    return done

def next_part_index(parts_dir: Path, prefix: str) -> int:
    idxs = []
    for pth in parts_dir.glob(f"{prefix}_part_*.parquet"):
        m = re.search(rf"{re.escape(prefix)}_part_(\d+)\.parquet$", pth.name)
        if m: idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 1

def write_part(rows: list[dict], parts_dir: Path, prefix: str, idx: int) -> Path:
    out = parts_dir / f"{prefix}_part_{idx:06d}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[checkpoint] wrote {len(rows)} rows -> {out}")
    return out

def merge_parts(parts_dir: Path, final_path: Path):
    files = sorted(parts_dir.glob("*.parquet"))
    if not files:
        print("[merge] no parts to merge.")
        return
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        print("[merge] no readable parts.")
        return
    big = (pd.concat(dfs, ignore_index=True)
             .sort_values("id")
             .drop_duplicates("id", keep="last"))
    big.to_parquet(final_path, index=False)
    print(f"[merge] wrote {final_path}  (rows={len(big)})")

# ---------------- Fetch (REST) ----------------
def fetch_rest(repo_id: str) -> dict:
    url = f"https://huggingface.co/api/datasets/{quote(repo_id, safe='')}"
    headers = {**UA, **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {})}
    params = {
        "expand": ["downloads","downloadsAllTime","createdAt","lastModified", "trendingScore", "likes", "usedStorage"] #, "tags"]
    }
    backoff = 1.0
    for _ in range(6):
        try:
            r = http.get(url, headers=headers, params=params, timeout=25)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff = min(backoff*2, 30); continue
            r.raise_for_status()
            js = r.json()
            polite_sleep()

            tags = js.get("tags") or []
            # remove language tags (start with "language:")
            tags = [t for t in tags if not t.startswith("language:")]
            tags.sort()  # optional alphabetical sort
            return {
                "id": repo_id,
                "created_at": js.get("createdAt"),
                "last_modified": js.get("lastModified"),
                "downloads_30d": js.get("downloads"),
                "downloads_all_time": js.get("downloadsAllTime"),
                "trending_score": js.get("trendingScore"),
                "likes": js.get("likes"),
                "used_storage": js.get("usedStorage"),
                #"tags": tags,
                "status": "ok",
                "fetch_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
            }
        except requests.RequestException:
            time.sleep(backoff); backoff = min(backoff*2, 30)
    return {
        "id": repo_id, "created_at": None, "last_modified": None,
        "downloads_30d": None, "downloads_all_time": None,
        "trending_score": None, "likes": None, 
        "used_storage": None, #"tags": None,
        "status": "error",
        "error_message": f"REST failed after retries for {repo_id}",
        "fetch_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
    }

# ---------------- Main ----------------
ids_all = read_ids(IDS_FILE)
done = list_existing_ids(PARTS_DIR, OUT_PATH)
todo = [i for i in ids_all if i not in done]
print(f"[ids] total={len(ids_all)}  already_saved={len(done)}  to_process={len(todo)}")

rows_buffer = []
part_idx = next_part_index(PARTS_DIR, PART_PREFIX) - 1
parts_written = 0

def flush():
    global rows_buffer, part_idx, parts_written
    if not rows_buffer: return
    part_idx += 1
    write_part(rows_buffer, PARTS_DIR, PART_PREFIX, part_idx)
    parts_written += 1
    rows_buffer = []
    if parts_written % max(1, args.merge_every_parts) == 0:
        merge_parts(PARTS_DIR, OUT_PATH)

if not todo:
    print("[done] nothing to do; merging existing parts…")
    merge_parts(PARTS_DIR, OUT_PATH)
    raise SystemExit(0)

with ThreadPoolExecutor(max_workers=THREADS) as ex:
    futures = {ex.submit(fetch_rest, ds_id): ds_id for ds_id in todo}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching (REST)", unit="ds", dynamic_ncols=True):
        rows_buffer.append(fut.result())
        if len(rows_buffer) >= BATCH:
            flush()

flush()
merge_parts(PARTS_DIR, OUT_PATH)
print(f"[done] final: {OUT_PATH}")