"""Combine the scan parts into one flag per benchmark question.

Reads every <hits>/hits.partK.npz + summary.partK.json written by scan_shards.py and the test
index (index.npz + test_items.json from build_test_index.py). Refuses to run unless every part
finished, all parts used the same index and file list, and every shard was scanned exactly once.

Writes to --out:
    flags.jsonl       one line per test question, in lm-eval order (joins RQ 4.1 on task, doc_id)
    summary.json      counts per benchmark, totals, the rule, and fingerprints
    top_matches.txt   the most frequent matching 13-grams, decoded, to spot stock phrasing

Rule: a question is flagged if any of its 13-grams appears anywhere in the scanned shards.
"frac_matched" (share of the question's 13-grams found) is reported too, so a stricter
threshold can be applied later without rescanning. Overlap is a warning, not proof.

To ignore 13-grams judged to be stock phrasing, list their hashes (one per line, as printed in
top_matches.txt) in a file and pass --drop-hashes; the summary records how many were dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_parts(hits_dir: Path, index_sha: str):
    summaries = [json.loads(f.read_text()) for f in sorted(hits_dir.glob("summary.part*.json"))]
    if not summaries:
        raise SystemExit(f"No summary.part*.json in {hits_dir}")
    nparts = {s["nparts"] for s in summaries}
    if len(nparts) != 1 or sorted(s["part"] for s in summaries) != list(range(nparts.pop())):
        raise SystemExit("Missing or mixed parts: rerun the parts that failed")
    if not all(s["complete"] for s in summaries):
        raise SystemExit("A part was a timing run (--limit-files): rerun it in full")
    if {s["index_sha256"] for s in summaries} != {index_sha}:
        raise SystemExit("Parts used a different index than --index")
    files = summaries[0]["all_files"]
    if any(s["all_files"] != files for s in summaries):
        raise SystemExit("Parts saw different shard lists")
    scanned = sorted(f["file"] for s in summaries for f in s["scanned"])
    if scanned != list(range(len(files))):
        raise SystemExit("Not every shard was scanned exactly once")
    h, f, p = [], [], []
    for s in summaries:
        z = np.load(hits_dir / f"hits.part{s['part']}.npz")
        h.append(z["hashes"]); f.append(z["files"]); p.append(z["positions"])
    return summaries, files, np.concatenate(h), np.concatenate(f), np.concatenate(p)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="folder with index.npz and test_items.json")
    ap.add_argument("--hits", required=True, help="folder the scan parts wrote to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="", help="class .model file, to decode top_matches.txt")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--drop-hashes", default="", help="file of 13-gram hashes to ignore (stock phrasing)")
    a = ap.parse_args(argv)

    index_dir, out = Path(a.index), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.npz"
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    z = np.load(index_path)
    H, O = z["hashes"], z["owners"]                     # sorted; one row per (hash, owning item)
    test = json.loads((index_dir / "test_items.json").read_text())
    items = test["items"]

    summaries, files, hh, ff, pp = load_parts(Path(a.hits), index_sha)

    dropped = set()
    if a.drop_hashes:
        dropped = {int(x) for x in Path(a.drop_hashes).read_text().split()}
        keep = ~np.isin(hh, np.array(sorted(dropped), dtype=np.uint64))
        hh, ff, pp = hh[keep], ff[keep], pp[keep]

    # one entry per distinct matched hash: corpus count, files, first occurrence
    uh, first, counts = np.unique(hh, return_index=True, return_counts=True)
    pair = np.unique(np.stack([np.searchsorted(uh, hh), ff.astype(np.int64)]), axis=1)
    files_of = defaultdict(list)
    for j, fi in zip(pair[0].tolist(), pair[1].tolist()):
        files_of[j].append(fi)

    lo = np.searchsorted(H, uh, side="left")
    hi = np.searchsorted(H, uh, side="right")
    matched = Counter(); hits = Counter(); shards = defaultdict(set); example = {}
    owners_of = {}
    for j in range(len(uh)):
        owners = sorted(set(O[lo[j]:hi[j]].tolist()))
        owners_of[j] = owners
        for it in owners:
            matched[it] += 1
            hits[it] += int(counts[j])
            shards[it].update(files_of[j])
            if it not in example:
                example[it] = (int(ff[first[j]]), int(pp[first[j]]))

    by_task = defaultdict(lambda: Counter())
    with open(out / "flags.jsonl", "w") as fo:
        for i, m in enumerate(items):
            n = int(m["n_ngrams"]); k = matched.get(i, 0)
            row = {"task": m["task"], "doc_id": m["doc_id"], "n_ngrams": n, "matched_ngrams": k,
                   "frac_matched": round(k / n, 4) if n else 0.0, "n_hits": hits.get(i, 0),
                   "flagged": k > 0, "short": n == 0,
                   "shards": [Path(files[s]).name for s in sorted(shards.get(i, ()))]}
            if i in example:
                row["example"] = {"shard": Path(files[example[i][0]]).name, "pos": example[i][1]}
            fo.write(json.dumps(row) + "\n")
            t = by_task[m["task"]]
            t["items"] += 1; t["short"] += n == 0; t["flagged"] += k > 0
            t["flagged_half"] += n > 0 and k / n >= 0.5

    per_task = {}
    for task, t in sorted(by_task.items()):
        checkable = t["items"] - t["short"]
        per_task[task] = {**dict(t), "flagged_pct": round(100 * t["flagged"] / checkable, 2) if checkable else None}
    summary = {
        "rule": "flagged = any shared 13-token run with the scanned shards; frac_matched also reported",
        "n": test["n"], "index_sha256": index_sha, "total_items": len(items),
        "shards": len(files), "tokens_scanned": sum(s["tokens"] for s in summaries),
        "matches_total": int(len(hh)), "distinct_matched_ngrams": int(len(uh)),
        "dropped_hashes": len(dropped), "per_task": per_task,
        "scan_seconds": sum(s["seconds"] for s in summaries),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    tok = None
    if a.tokenizer:
        from gw1b.tokenizer import Tokenizer
        tok = Tokenizer(a.tokenizer)
    from gw1b.data import read_shard
    order = np.argsort(-counts, kind="stable")[:a.top]
    with open(out / "top_matches.txt", "w") as fo:
        fo.write("count\towners\thash\texample_owner\ttext\n")
        for j in order.tolist():
            fi, pos = int(ff[first[j]]), int(pp[first[j]])
            ids = np.asarray(read_shard(files[fi])[pos:pos + test["n"]]).tolist()
            text = tok.decode(ids) if tok else " ".join(map(str, ids))
            own = items[owners_of[j][0]]
            fo.write(f"{int(counts[j])}\t{len(owners_of[j])}\t{int(uh[j])}\t"
                     f"{own['task']}:{own['doc_id']}\t{text!r}\n")

    for task, t in per_task.items():
        print(f"{task:20s} items {t['items']:6d}  short {t['short']:5d}  flagged {t['flagged']:5d}"
              f"  ({t['flagged_pct']}%)")
    print(f"wrote {out}/flags.jsonl, summary.json, top_matches.txt")


if __name__ == "__main__":
    main()
