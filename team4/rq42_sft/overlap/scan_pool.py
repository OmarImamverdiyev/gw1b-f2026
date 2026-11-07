"""Step 2: stream SFT / preference JSONL and flag every example sharing a 13-gram with any test item.

  gw1b run --cpu --cpus 8 --mem 32G --time 4:00:00 overlap/scan_pool.py --tokenizer TOK \
      --index $T4/data/overlap/test_index --pool $T4/data/sft/raw/*.jsonl \
      --out $T4/data/overlap/scan_nov7 --shard 0/4        # run shards 0..3, then:
  python overlap/scan_pool.py --merge $T4/data/overlap/scan_nov7

Scans EVERY text field (messages[*].content, system, prompt, response, chosen, rejected), each tokenized
separately so no 13-gram spans two fields. Writes flagged rows, removal IDs and counts per source x test set.
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ngram import N, ngram_hashes, normalize

FIELDS = ("system", "prompt", "response", "chosen", "rejected")


def texts_of(row):
    for f in FIELDS:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            yield f, v
        elif isinstance(v, list):  # chosen/rejected may be message lists
            for j, m in enumerate(v):
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    yield f"{f}[{j}]", m["content"]
    for j, m in enumerate(row.get("messages") or []):
        if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip():
            yield f"messages[{j}]", m["content"]


def scan_tokens(tokens, H, O):
    """-> (n_matched_ngrams, longest_run_tokens, set(test item indices))"""
    h = ngram_hashes(tokens)
    if h.size == 0 or H.size == 0:
        return 0, 0, set()
    pos = np.minimum(np.searchsorted(H, h), len(H) - 1)
    hit = H[pos] == h
    if not hit.any():
        return 0, 0, set()
    idx = np.flatnonzero(hit)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    items = set()
    for hh in np.unique(h[hit]):
        items.update(O[np.searchsorted(H, hh, "left"):np.searchsorted(H, hh, "right")].tolist())
    return int(hit.sum()), max(len(r) for r in runs) + N - 1, items


def scan_rows(rows, tok, H, O, meta, group_of):
    """rows: iterable of (example_id, source, row). Yields flag records; also returns counters."""
    flagged, counts = [], defaultdict(Counter)
    totals = Counter()
    for ex_id, source, row in rows:
        totals[source] += 1
        n_match, longest, items, fields = 0, 0, set(), []
        for field, text in texts_of(row):
            m, l, it = scan_tokens(tok.encode(normalize(text)), H, O)
            if m:
                n_match += m; longest = max(longest, l); items |= it; fields.append(field)
        if n_match:
            tests = sorted({f'{meta[i]["task"]}:{meta[i]["doc_id"]}' for i in items})
            groups = sorted({group_of(meta[i]["task"]) for i in items})
            for g in groups:
                counts[source][g] += 1
            totals[f"flagged::{source}"] += 1
            flagged.append({"id": ex_id, "source": source, "fields": fields, "n_matched": n_match,
                            "longest_run": longest, "test_items": tests[:20], "n_test_items": len(tests),
                            "test_sets": groups})
    return flagged, counts, totals


def group_of(task):
    return "mmlu" if task.startswith("mmlu") else task


def read_pool(files, shard, nshards):
    k = 0
    for f in files:
        stem = Path(f).stem
        with open(f, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                if not line.strip():
                    continue
                if k % nshards == shard:
                    row = json.loads(line)
                    yield f"{stem}:{line_no}", str(row.get("source") or row.get("original_source") or stem), row
                k += 1


def merge(out):
    out = Path(out)
    flagged, counts, totals = [], defaultdict(Counter), Counter()
    for f in sorted(out.glob("flags.shard*.jsonl")):
        flagged += [json.loads(l) for l in open(f) if l.strip()]
    for f in sorted(out.glob("summary.shard*.json")):
        s = json.loads(f.read_text())
        totals.update(s["totals"])
        for src, c in s["counts"].items():
            counts[src].update(c)
    ids = sorted({r["id"] for r in flagged})
    (out / "remove_ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    summary = {"n": N, "rule": "any shared 13-gram (INTERIM scanner; cross-check against Yusra's)",
               "examples_scanned": sum(v for k, v in totals.items() if not k.startswith("flagged::")),
               "examples_flagged": len(ids), "per_source_totals": {k: v for k, v in totals.items() if not k.startswith("flagged::")},
               "per_source_flagged": {k[9:]: v for k, v in totals.items() if k.startswith("flagged::")},
               "flagged_by_source_and_test_set": {k: dict(v) for k, v in counts.items()}}
    (out / "overlap_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merge")
    p.add_argument("--tokenizer"); p.add_argument("--index"); p.add_argument("--pool", nargs="+")
    p.add_argument("--out"); p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    if a.merge:
        return merge(a.merge)
    from gw1b.tokenizer import Tokenizer
    tok = Tokenizer(a.tokenizer)
    z = np.load(Path(a.index) / "index.npz")
    meta = json.loads((Path(a.index) / "test_items.json").read_text())["items"]
    files = sorted(f for pat in a.pool for f in glob.glob(pat))
    s, n = map(int, a.shard.split("/"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    flagged, counts, totals = scan_rows(read_pool(files, s, n), tok, z["hashes"], z["owners"], meta, group_of)
    (out / f"flags.shard{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in flagged))
    (out / f"summary.shard{s}.json").write_text(json.dumps(
        {"totals": dict(totals), "counts": {k: dict(v) for k, v in counts.items()}, "files": files}))
    print(f"shard {s}/{n}: scanned {sum(v for k, v in totals.items() if not k.startswith('flagged::'))}, flagged {len(flagged)}")


if __name__ == "__main__":
    main()
